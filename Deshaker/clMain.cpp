/*
 * This file is part of CUVISTA - Cuda Video Stabilizer
 * Copyright (c) 2023 Rainer Bitschi cuvista@a1.net
 *
 * This program is free software : you can redistribute it and /or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.If not, see < http://www.gnu.org/licenses/>.
 */

#include "clFunctions.hpp"
#include "clMain.hpp"
#include "clKernels.hpp"
#include "clKernelCompute.hpp"
#include "Util.hpp"
#include "DeviceInfo.hpp"
#include "Affine2D.hpp"
#include "MovieFrame.hpp"

#include <format>
#include <algorithm>
#include <regex>


using namespace cl;

static bool operator == (const ImageFormat& a, const ImageFormat& b) {
	return a.image_channel_data_type == b.image_channel_data_type && a.image_channel_order == b.image_channel_order;
}

OpenClFrame::OpenClFrame(CoreData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool) :
	FrameExecutor(data, deviceInfo, frame, pool) {}

//check available devices
std::vector<DeviceInfoOpenCl> cl::probeRuntime() {
	std::vector<DeviceInfoOpenCl> out;

	try {
		cl_uint n;
		std::vector<Platform> platforms;

		//first check number of platforms without raising an exception
		cl_int err = clGetPlatformIDs(0, nullptr, &n);
		if (err == CL_SUCCESS) {
			Platform::get(&platforms);
		}

		for (Platform& platform : platforms) {
			std::vector<Device> devices;
			platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
			for (Device& dev : devices) {
				cl_bool avail = dev.getInfo<CL_DEVICE_AVAILABLE>();
				if (avail == false) continue;

				cl_int prefWidth = dev.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>();
				if (prefWidth == 0) continue;

				cl_int nativeWidth = dev.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE>();
				if (nativeWidth == 0) continue;

				cl_bool hasImage = dev.getInfo<CL_DEVICE_IMAGE_SUPPORT>();
				if (hasImage == false) continue;

				int64_t maxImageArraySize = dev.getInfo<CL_DEVICE_IMAGE_MAX_ARRAY_SIZE>();
				if (maxImageArraySize < 2048) continue;

				//pixel dimensions
				int64_t maxPixelWidth = dev.getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>();
				int64_t maxPixelHeight = dev.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>();

				cl_device_fp_config doubleConfig = dev.getInfo< CL_DEVICE_DOUBLE_FP_CONFIG>();
				cl_int minDoubleConfig = CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN | CL_FP_DENORM;
				if ((doubleConfig & minDoubleConfig) == 0) continue;

				cl_ulong localMemSize = dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
				int64_t maxPixel = localMemSize / sizeof(float);
				if (maxPixelWidth < maxPixel) maxPixel = maxPixelWidth;
				if (maxPixelHeight < maxPixel) maxPixel = maxPixelHeight;

				//find device version
				int versionDevice = 0;
				int versionC = 0;
				std::regex pattern("OpenCL (\\d)\\.(\\d) .*");
				std::smatch matches;
				std::string deviceVersion = dev.getInfo<CL_DEVICE_VERSION>();
				if (std::regex_match(deviceVersion, matches, pattern) && matches.size() == 3) {
					versionDevice = std::stoi(matches[1]) * 1000 + std::stoi(matches[2]);
				}

				//find C version dependent on device version
				if (versionDevice < 3000) {
					std::string str = dev.getInfo<CL_DEVICE_OPENCL_C_VERSION>();
					std::regex pattern("OpenCL C (\\d)\\.(\\d) .*");
					if (std::regex_match(str, matches, pattern) && matches.size() == 3) {
						versionC = std::stoi(matches[1]) * 1000 + std::stoi(matches[2]);
					}

				} else {
					auto versionList = dev.getInfo<CL_DEVICE_OPENCL_C_ALL_VERSIONS>();
					auto func = [] (cl_name_version a, cl_name_version b) { return a.version < b.version; };
					auto maxVersion = std::max_element(versionList.begin(), versionList.end(), func);
					versionC = 1000 * (maxVersion->version >> 22) + (maxVersion->version >> 12 & 0x3FF);
				}

				//accept only devices of at least version 2.0
				if (versionDevice < 2000) continue;
				if (versionC < 2000) continue;

				//images from buffer not supported on Nvidia on OpenCL 3.0
				cl_uint pitch = dev.getInfo<CL_DEVICE_IMAGE_PITCH_ALIGNMENT>();

				//list of device extensions
				string str = dev.getInfo<CL_DEVICE_EXTENSIONS>();
				std::regex delimiter(" ");
				auto iter = std::sregex_token_iterator(str.begin(), str.end(), delimiter, -1);
				std::vector<std::string> extensions(iter, {});
				//std::vector<cl_name_version> extensions = dev.getInfo<CL_DEVICE_EXTENSIONS_WITH_VERSION>(); //Missing before version 3.0.

				//we have a valid device
				DeviceInfoOpenCl devInfo(maxPixel);
				devInfo.device = std::make_shared<Device>(dev);
				devInfo.versionDevice = versionDevice;
				devInfo.versionC = versionC;
				devInfo.pitch = pitch;
				devInfo.extensions = extensions;
				devInfo.platformVersion = platform.getInfo<CL_PLATFORM_VERSION>();

				out.push_back(devInfo);
			}
		}

	} catch (const Error& err) {
		DeviceInfoOpenCl::warning = std::format("OpenCL init error: {}", err.what());

	} catch (...) {
		DeviceInfoOpenCl::warning = "unknown error loading Open CL";
	}

	return out;
}

//set up device to use
void OpenClFrame::init() {
	assert(mDeviceInfo.getType() == DeviceType::OPEN_CL && "device type must be OpenCL here");
	const DeviceInfoOpenCl* devInfo = static_cast<const DeviceInfoOpenCl*>(&mDeviceInfo);

	try {
		clData.context = Context(*devInfo->device);
		clData.queue = CommandQueue(clData.context, *devInfo->device);
		clData.secondQueue = CommandQueue(clData.context, *devInfo->device);

		//supported image formats
		std::vector<ImageFormat> fmts;
		clData.context.getSupportedImageFormats(CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, &fmts);

		//constant core data structure
		KernelData data = { 
			mData.compMaxTol, mData.deps, mData.dmin, mData.dmax, mData.dnan,
			mData.w, mData.h, mData.ir, mData.iw, mData.zMin, mData.zMax, 
			mData.compMaxIter, mData.pyramidRowCount 
		};
		clData.core = Buffer(clData.context, CL_MEM_READ_ONLY, sizeof(KernelData));
		clData.queue.enqueueWriteBuffer(clData.core, CL_FALSE, 0, sizeof(KernelData), &data);

		//allocate ayuv data
		ImageFormat fmtAyuv(CL_RGBA, CL_UNSIGNED_INT8);
		for (int idx = 0; idx < mData.bufferCount; idx++) {
			Image2D ayuv(clData.context, CL_MEM_READ_ONLY, fmtAyuv, mData.w, mData.h);
			clData.ayuv.push_back(ayuv);
		}

		clData.backgroundColorAyuv = { mData.bgcolorAyuv[0], mData.bgcolorAyuv[1], mData.bgcolorAyuv[2], mData.bgcolorAyuv[3] };

		//output images
		ImageFormat outFmt(CL_RGBA, CL_FLOAT);
		for (Image2D& im : clData.out) {
			im = Image2D(clData.context, CL_MEM_READ_WRITE, outFmt, mData.w, mData.h);
			clData.queue.enqueueFillImage(im, clData.backgroundColorAyuv, Size2(), Size2(mData.w, mData.h));
		}
		clData.ayuvOut = Buffer(clData.context, CL_MEM_WRITE_ONLY, mData.stride4 * mData.h);
		clData.rgbaOut = Buffer(clData.context, CL_MEM_WRITE_ONLY, mData.stride4 * mData.h);

		//allocate input image
		mInputFrame = ImageAyuv(mData.w, mData.h, mData.stride4);

		//image format gray single channel float
		ImageFormat fmt32(CL_DEPTH, CL_FLOAT);

		//buffer pyramid for filtering
		for (int z = 0; z < mData.pyramidLevels; z++) {
			int hh = mData.h >> z;
			int ww = mData.w >> z;
			BufferImages buf = {
				Image2D(clData.context, CL_MEM_READ_WRITE, fmt32, ww, hh),
				Image2D(clData.context, CL_MEM_READ_WRITE, fmt32, ww, hh),
				Image2D(clData.context, CL_MEM_READ_WRITE, fmt32, ww, hh)
			};
			clData.buffer.push_back(buf);
		}

		//allocate pyramid array, one image holds all levels, kernel type image2d_array_depth_t
		clData.pyramid = Image2DArray(clData.context, CL_MEM_READ_WRITE, fmt32, mData.pyramidCount, mData.w, mData.pyramidRowCount, 0, 0);

		//point results
		clData.results = Buffer(clData.context, CL_MEM_WRITE_ONLY, sizeof(cl_PointResult) * mData.resultCount);
		clData.cl_results.resize(mData.resultCount);

		//compile device code
		std::string kernelNames[] = { kernelsInputOutput, luinvFunction, norm1Function, kernelCompute };
		Program::Sources sources;
		for (const std::string& str : kernelNames) {
			sources.emplace_back(str.c_str(), str.size());
		}

		//build program to latest C version
		std::string compilerFlag = std::format("-cl-std=CL{}.{}", devInfo->versionC / 1000, devInfo->versionC % 1000);
		Program program(clData.context, sources);
		program.build(compilerFlag.c_str());

		//assign kernels
		clData.kernels.scale_8u32f_1 = Kernel(program, "scale_8u32f_1");
		clData.kernels.scale_8u32f_3 = Kernel(program, "scale_8u32f_3");
		clData.kernels.scale_32f8u_3 = Kernel(program, "scale_32f8u_3");
		clData.kernels.filter_32f_1 = Kernel(program, "filter_32f_1");
		clData.kernels.filter_32f_3 = Kernel(program, "filter_32f_3");
		clData.kernels.remap_downsize_32f = Kernel(program, "remap_downsize_32f");
		clData.kernels.warp_back = Kernel(program, "warp_back");
		clData.kernels.unsharp = Kernel(program, "unsharp");
		clData.kernels.yuv8u_to_rgba = Kernel(program, "yuv8u_to_rgba");
		clData.kernels.yuv32f_to_rgba = Kernel(program, "yuv32f_to_rgba");
		clData.kernels.yuv32f_to_nv12 = Kernel(program, "yuv32f_to_nv12");
		clData.kernels.scrap = Kernel(program, "scrap");
		clData.kernels.compute = Kernel(program, "compute");

	} catch (const BuildError& err) {
		for (auto& data : err.getBuildLog()) {
			Device dev = data.first;
			std::string msg = data.second;
			size_t maxLen = 150;
			if (msg.length() > maxLen) {
				msg = msg.substr(0, maxLen) + "...\n[total " + std::to_string(msg.length()) + " chars]";
			}
			errorLogger().logError("OpenCL build error:\n", msg);
		}

	} catch (const Error& err) {
		errorLogger().logError("OpenCL init error: ", err.what());
	}
}

//----------------------------------
//-------- INPUT AYUV DATA ----------
//----------------------------------

Image8& OpenClFrame::inputDestination(int64_t frameIndex) {
	return mInputFrame;
}

void OpenClFrame::inputData(int64_t frameIndex) {
	int64_t idx = frameIndex % mData.bufferCount;
	try {
		clData.queue.enqueueWriteImage(clData.ayuv[idx], CL_TRUE, Size2(), Size2(mData.w, mData.h), mInputFrame.stride(), 0, mInputFrame.data());
	
	} catch (const Error& err) {
		errorLogger().logError("OpenCL input error: ", err.what());
	}
}

//----------------------------------
//-------- CREATE PYRAMID ----------
//----------------------------------

int64_t OpenClFrame::createPyramid(int64_t frameIndex, AffineDataFloat trf, bool warp) {
	//util::ConsoleTimer ic("ocl pyramid");
	int w = mData.w;
	int h = mData.h;
	int64_t frIdx = frameIndex % mData.bufferCount;
	int64_t pyrIdx = frameIndex % mData.pyramidCount;

	try {
		//convert yuv image to first level of Y pyramid
		Image& im = clData.buffer[0].result;
		Image& buf = clData.buffer[0].filterH;

		if (warp) {
			cl_float4 bg = { 0.0f, 0.0f, 0.0f, 0.0f };
			clData.queue.enqueueFillImage(im, bg, Size2(), Size2(mData.w, mData.h));
			scale_8u32f_1(clData.ayuv[frIdx], buf, clData);
			cl_float8 cltrf = { trf.m00, trf.m01, trf.m02, trf.m10, trf.m11, trf.m12 };
			warp_back(buf, im, clData, cltrf);

		} else {
			//convert uint8_t to float
			scale_8u32f_1(clData.ayuv[frIdx], im, clData);

			//filter first level
			filter_32f_h1(im, buf, 0, clData);
			filter_32f_v1(buf, im, 0, clData);
		}
		//copy into pyramid
		clData.queue.enqueueCopyImage(im, clData.pyramid, Size3(), Size3(0, 0, pyrIdx), Size3(w, h, 1));

		//lower levels of pyramid
		size_t row = h;
		for (size_t z = 1; z < mData.pyramidLevels; z++) {
			Image& src = clData.buffer[z - 1].result;
			Image& dest = clData.buffer[z].result;
			remap_downsize_32f(src, dest, clData);
			int ww = w >> z;
			int hh = h >> z;
			clData.queue.enqueueCopyImage(dest, clData.pyramid, Size3(), Size3(0ull, row, pyrIdx), Size3(ww, hh, 1));
			row += hh;
		}

	} catch (const Error& err) {
		errorLogger().logError("OpenCL pyramid error: ", err.what());
	}

	return 0;
}

//----------------------------------
//-------- COMPUTE -----------------
//----------------------------------

//internal function to start compute kernel
void OpenClFrame::compute(int64_t frameIndex, const CoreData& core, int rowStart, int rowEnd) {
	//util::ConsoleTimer timer("ocl compute start");
	assert(frameIndex > 0 && "invalid pyramid index");

	try {
		//local memory size in bytes
		size_t iw = 2LL * core.ir + 1;
		size_t memsiz = (7 * iw * iw + 108) * sizeof(cl_double) + 6 * sizeof(cl_double*);
		//set up compute kernel
		Kernel& kernel = clData.kernels.compute;
		kernel.setArg(0, (cl_long) frameIndex);
		kernel.setArg(1, clData.pyramid);
		kernel.setArg(2, clData.results);
		kernel.setArg(3, clData.core);
		kernel.setArg(4, memsiz, nullptr);
		kernel.setArg(5, rowStart);

		//threads
		size_t nX = std::max(iw, size_t(6));
		size_t nY = 4;
		NDRange ndlocal = NDRange(nX, nY);
		NDRange ndglobal = NDRange(ndlocal[0] * core.ixCount, ndlocal[1] * (rowEnd - rowStart));
		clData.queue.enqueueNDRangeKernel(kernel, NullRange, ndglobal, ndlocal);

	} catch (const Error& err) {
		errorLogger().logError("OpenCL compute error: ", err.what());
	}
}

void OpenClFrame::computeStart(int64_t frameIndex, std::span<PointResult> results) {
	try {
		//reset computed flag
		clData.queue.enqueueFillBuffer<cl_char>(clData.results, 0, 0, sizeof(cl_PointResult) * mData.resultCount);

	} catch (const Error& err) {
		errorLogger().logError("OpenCL compute error: ", err.what());
	}
	//compute first part of points
	compute(frameIndex, mData, 0, mData.iyCount / 4);
}

void OpenClFrame::computeTerminate(int64_t frameIndex, std::span<PointResult> results) {
	//util::ConsoleTimer timer("ocl compute end");

	//compute rest of points
	compute(frameIndex, mData, mData.iyCount / 4, mData.iyCount);
	
	try {
		//copy results from device to host buffer
		clData.queue.enqueueReadBuffer(clData.results, CL_TRUE, 0, sizeof(cl_PointResult) * mData.resultCount, clData.cl_results.data());

		//convert from cl_PointResult to PointResult
		for (size_t i = 0; i < results.size(); i++) {
			const cl_PointResult& pr = clData.cl_results[i];
			double x0 = pr.xm - mData.w / 2.0 + pr.u * pr.direction;
			double y0 = pr.ym - mData.h / 2.0 + pr.v * pr.direction;
			double fdir = 1.0 - 2.0 * pr.direction;
			results[i] = { pr.idx, pr.ix0, pr.iy0, x0, y0, pr.u * fdir, pr.v * fdir, PointResultType(pr.result), pr.zp, pr.direction, pr.length };
		}

	} catch (const Error& err) {
		errorLogger().logError("OpenCL compute error: ", err.what());
	}
}

//----------------------------------
//-------- OUTPUT STABILIZED -------
//----------------------------------

void OpenClFrame::outputData(int64_t frameIndex, AffineDataFloat trf) {
	//util::ConsoleTimer timer("ocl output");
	int64_t frIdx = frameIndex % mData.bufferCount;
	auto& [outStart, outWarped, outFilterH, outFilterV, outFinal] = clData.out;

	try {
		//convert input ayuv to float image
		scale_8u32f_3(clData.ayuv[frIdx], outStart, clData);
		//fill static background when requested
		if (mData.bgmode == BackgroundMode::COLOR) {
			clData.queue.enqueueFillImage(outWarped, clData.backgroundColorAyuv, Size2(), Size2(mData.w, mData.h));
		}
		//warp input on top of background
		cl_float8 cltrf = { trf.m00, trf.m01, trf.m02, trf.m10, trf.m11, trf.m12 };
		warp_back(outStart, outWarped, clData, cltrf);

		//filtering
		filter_32f_h3(outWarped, outFilterH, clData);
		filter_32f_v3(outFilterH, outFilterV, clData);

		//unsharp mask
		cl_float4 factor = { 0.0f, mData.unsharp.y, mData.unsharp.u, mData.unsharp.v };
		unsharp(outWarped, outFinal, outFilterV, clData, factor);

	} catch (const Error& err) {
		errorLogger().logError("OpenCL output error: ", err.what());
	}
}

//utility function to read from image
void OpenClFrame::readImage(Image src, size_t destPitch, void* dest, CommandQueue queue) const {
	size_t w = src.getImageInfo<CL_IMAGE_WIDTH>();
	size_t h = src.getImageInfo<CL_IMAGE_HEIGHT>();
	queue.enqueueReadImage(src, CL_TRUE, Size2(), Size2(w, h), destPitch, 0, dest);
}

//utility function to read from image array
void OpenClFrame::readImage(Image2DArray src, int idx, size_t destPitch, void* dest, cl::CommandQueue queue) const {
	size_t w = src.getImageInfo<CL_IMAGE_WIDTH>();
	size_t h = src.getImageInfo<CL_IMAGE_HEIGHT>();
	queue.enqueueReadImage(src, CL_TRUE, Size3(0, 0, idx), Size3(w, h, 1ull), destPitch, 0, dest);
}

void OpenClFrame::getInput(int64_t frameIndex, Image8& image) const {
	int64_t idx = frameIndex % mData.bufferCount;

	try {
		if (image.imageType() == ImageType::AYUV) {
			Image im = clData.ayuv[idx];
			readImage(im, image.stride(), image.data(), clData.queue);

		} else if (image.colorBase() == ColorBase::RGB) {
			yuv_to_rgba(clData.kernels.yuv8u_to_rgba, clData.ayuv[idx], image.data(), clData, image.width(), image.height(), image.stride(), image.colorIndex());
		}

	} catch (const Error& err) {
		errorLogger().logError("OpenCL get input: ", err.what());
	}
}

Matf OpenClFrame::getPyramid(int64_t frameIndex) const {
	size_t pyrIdx = frameIndex % mData.pyramidCount;
	size_t wbytes = mData.w * sizeof(float);
	Matf out = Mat<float>::zeros(mData.pyramidRowCount, mData.w);

	try {
		readImage(clData.pyramid, pyrIdx, wbytes, out.data(), clData.queue);

	} catch (const Error& err) {
		errorLogger().logError("OpenCL get pyramid: ", err.what());
	}
	return out;
}

void OpenClFrame::getOutput(int64_t frameIndex, Image8& image) const {
	try {
		if (image.imageType() == ImageType::AYUV) {
			int stride = clData.ayuvOut.getInfo<CL_MEM_SIZE>() / image.height();
			//convert to YUV444 for output
			scale_32f8u_3(clData.out[4], clData.ayuvOut, stride, clData);

			//copy to cpu memory
			Size2 region(image.stride(), mData.h);
			clData.queue.enqueueReadBufferRect(clData.ayuvOut, CL_TRUE, Size2(), Size2(), region, stride, 0, 0, 0, image.data());

		} else if (image.colorBase() == ColorBase::RGB) {
			yuv_to_rgba(clData.kernels.yuv32f_to_rgba, clData.out[4], image.data(), clData, image.width(), image.height(), image.stride(), image.colorIndex());
		}

		//forward image index
		image.setIndex(frameIndex);

	} catch (const Error& err) {
		errorLogger().logError("OpenCL output error: ", err.what());
	}
}

bool OpenClFrame::getOutput(int64_t frameIndex, Image8& image, int cudaNv12stride, unsigned char* cudaNv12ptr) const {
	try {
		yuv_to_nv12(clData.kernels.yuv32f_to_nv12, clData.out[4], image.data(), clData, image.width(), image.height(), image.stride());
		image.setIndex(frameIndex);

	} catch (const Error& err) {
		errorLogger().logError("OpenCL output error: ", err.what());
	}
	return true;
}

Matf OpenClFrame::getTransformedOutput() const {
	std::vector<cl_float4> imageData(1ull * mData.w * mData.h);
	readImage(clData.out[1], mData.w * sizeof(cl_float4), imageData.data(), clData.queue);

	auto func = [&] (size_t r, size_t c) {
		cl_float4 data = imageData[r * mData.w + c / 4];
		switch (c % 4) {
		case 0: return data.x;
		case 1: return data.y;
		case 2: return data.z;
		case 3: return data.w;
		default: return 0.0f;
		}
	};
	return Matf::generate(mData.h, mData.w * 4ull, func);
}

void OpenClFrame::getWarped(int64_t frameIndex, Image8bgr & image) {
	try {
		yuv_to_rgba(clData.kernels.yuv32f_to_rgba, clData.out[1], image.data(), clData, image.width(), image.height(), image.stride(), image.colorIndex());

	} catch (const Error& err) {
		errorLogger().logError("OpenCL get warped: ", err.what());
	}
}