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

OpenClExecutor::OpenClExecutor(CoreData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool) :
	FrameExecutor(data, deviceInfo, frame, pool) {}

//check available devices
void cl::probeRuntime(OpenClInfo& info) {
	try {
		std::vector<Platform> platforms;
		Platform::get(&platforms);

		for (Platform& platform : platforms) {
			info.version = platform.getInfo<CL_PLATFORM_VERSION>();
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
				DeviceInfoOpenCl devInfo(DeviceType::OPEN_CL, maxPixel);
				devInfo.device = std::make_shared<Device>(dev);
				devInfo.versionDevice = versionDevice;
				devInfo.versionC = versionC;
				devInfo.pitch = pitch;
				devInfo.extensions = extensions;
				info.devices.push_back(devInfo);
			}
		}

	} catch (const Error& err) {
		info.warning = std::format("OpenCL init error: {}", err.what());

	} catch (...) {
		info.warning = "unknown error loading Open CL";
	}
}

//set up device to use
void OpenClExecutor::init() {
	assert(mDeviceInfo.type == DeviceType::OPEN_CL && "device type must be OpenCL here");
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
			mData.COMP_MAX_TOL, mData.deps, mData.dmin, mData.dmax, mData.dnan,
			mData.w, mData.h, mData.ir, mData.iw, mData.zMin, mData.zMax, 
			mData.COMP_MAX_ITER, mData.pyramidRowCount 
		};
		clData.core = Buffer(clData.context, CL_MEM_READ_ONLY, sizeof(KernelData));
		clData.queue.enqueueWriteBuffer(clData.core, CL_FALSE, 0, sizeof(KernelData), &data);

		//allocate yuv data
		ImageFormat fmt8(CL_R, CL_UNSIGNED_INT8);
		for (int idx = 0; idx < mData.bufferCount; idx++) {
			Image2D yuv(clData.context, CL_MEM_READ_ONLY, fmt8, mData.w, mData.h * 3ull);
			clData.yuv.push_back(yuv);
		}

		//output images
		ImageFormat outFmt(CL_RGBA, CL_FLOAT);
		for (Image2D& im : clData.out) {
			im = Image2D(clData.context, CL_MEM_READ_WRITE, outFmt, mData.w, mData.h);
			cl_float4 bg = { mData.bgcolorYuv[0], mData.bgcolorYuv[1], mData.bgcolorYuv[2], 0.0f };
			clData.queue.enqueueFillImage(im, bg, Size2(), Size2(mData.w, mData.h));
		}
		clData.yuvOut = Buffer(clData.context, CL_MEM_WRITE_ONLY, 3ull * mData.cpupitch * mData.h);
		clData.rgbaOut = Buffer(clData.context, CL_MEM_WRITE_ONLY, 4ull * mData.w * mData.h);

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
//-------- INPUT YUV DATA ----------
//----------------------------------

void OpenClExecutor::inputData(int64_t frameIndex, const ImageYuv& inputFrame) {
	int64_t fr = frameIndex % mData.bufferCount;
	try {
		clData.queue.enqueueWriteImage(clData.yuv[fr], CL_TRUE, Size2(), Size2(mData.w, mData.h * 3), mData.cpupitch, 0, inputFrame.data());
	
	} catch (const Error& err) {
		errorLogger().logError("OpenCL input error: ", err.what());
	}
}

//----------------------------------
//-------- CREATE PYRAMID ----------
//----------------------------------

void OpenClExecutor::createPyramidTransformed(int64_t frameIndex, const Affine2D& trf) {

}

void OpenClExecutor::createPyramid(int64_t frameIndex) {
	//util::ConsoleTimer ic("ocl pyramid");
	int w = mData.w;
	int h = mData.h;
	int64_t frIdx = frameIndex % mData.bufferCount;
	int64_t pyrIdx = frameIndex % mData.pyramidCount;

	try {
		//convert yuv image to first level of Y pyramid
		Image& im = clData.buffer[0].result;
		Image& buf1 = clData.buffer[0].filterH;
		scale_8u32f_1(clData.yuv[frIdx], im, clData);

		//filter first level
		filter_32f_h1(im, buf1, 0, clData);
		filter_32f_v1(buf1, im, 0, clData);
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
}

//----------------------------------
//-------- COMPUTE -----------------
//----------------------------------

//internal function to start compute kernel
void OpenClExecutor::compute(int64_t frameIndex, const CoreData& core, int rowStart, int rowEnd) {
	//util::ConsoleTimer timer("ocl compute start");
	assert(frameIndex > 0 && "invalid pyramid index");

	try {
		//local memory size in bytes
		int memsiz = (7LL * core.iw * core.iw + 108) * sizeof(cl_double) + 6 * sizeof(cl_double*);
		//set up compute kernel
		Kernel& kernel = clData.kernels.compute;
		kernel.setArg(0, (cl_long) frameIndex);
		kernel.setArg(1, clData.pyramid);
		kernel.setArg(2, clData.results);
		kernel.setArg(3, clData.core);
		kernel.setArg(4, memsiz, nullptr);
		kernel.setArg(5, rowStart);

		//threads
		NDRange ndlocal = NDRange(core.iw, 32 / core.iw); //based on cuda warp
		NDRange ndglobal = NDRange(ndlocal[0] * core.ixCount, ndlocal[1] * (rowEnd - rowStart));
		clData.queue.enqueueNDRangeKernel(kernel, NullRange, ndglobal, ndlocal);

	} catch (const Error& err) {
		errorLogger().logError("OpenCL compute error: ", err.what());
	}
}

void OpenClExecutor::computeStart(int64_t frameIndex, std::vector<PointResult>& results) {
	try {
		//reset computed flag
		clData.queue.enqueueFillBuffer<cl_char>(clData.results, 0, 0, sizeof(cl_PointResult) * mData.resultCount);

	} catch (const Error& err) {
		errorLogger().logError("OpenCL compute error: ", err.what());
	}
	//compute first part of points
	compute(frameIndex, mData, 0, mData.iyCount / 4);
}

void OpenClExecutor::computeTerminate(int64_t frameIndex, std::vector<PointResult>& results) {
	//util::ConsoleTimer timer("ocl compute end");

	//compute rest of points
	compute(frameIndex, mData, mData.iyCount / 4, mData.iyCount);
	
	try {
		//copy results from device to host buffer
		clData.queue.enqueueReadBuffer(clData.results, CL_TRUE, 0, sizeof(cl_PointResult) * mData.resultCount, clData.cl_results.data());

		//convert from cl_PointResult to PointResult
		for (size_t i = 0; i < results.size(); i++) {
			cl_PointResult& pr = clData.cl_results[i];
			double x0 = pr.xm - mData.w / 2.0 + pr.u * pr.direction;
			double y0 = pr.ym - mData.h / 2.0 + pr.v * pr.direction;
			double fdir = 1.0 - 2.0 * pr.direction;
			results[i] = { pr.idx, pr.ix0, pr.iy0, x0, y0, pr.u * fdir, pr.v * fdir, PointResultType(pr.result), pr.zp, pr.direction };
		}

	} catch (const Error& err) {
		errorLogger().logError("OpenCL compute error: ", err.what());
	}
}

//----------------------------------
//-------- OUTPUT STABILIZED -------
//----------------------------------

void OpenClExecutor::outputData(int64_t frameIndex, const Affine2D& trf) {
	//util::ConsoleTimer timer("ocl output");
	int64_t frIdx = frameIndex % mData.bufferCount;
	auto& [outStart, outWarped, outFilterH, outFilterV, outFinal] = clData.out;

	try {
		//convert input yuv to float image
		scale_8u32f_3(clData.yuv[frIdx], outStart, clData);
		//fill static background when requested
		if (mData.bgmode == BackgroundMode::COLOR) {
			cl_float4 bg = { mData.bgcolorYuv[0], mData.bgcolorYuv[1], mData.bgcolorYuv[2], 0.0f };
			clData.queue.enqueueFillImage(outWarped, bg, Size2(), Size2(mData.w, mData.h));
		}
		//warp input on top of background
		warp_back(outStart, outWarped, clData, trf.toArray());

		//filtering
		filter_32f_h3(outWarped, outFilterH, clData);
		filter_32f_v3(outFilterH, outFilterV, clData);

		//unsharp mask
		cl_float4 factor = { mData.unsharp.y, mData.unsharp.u, mData.unsharp.v };
		unsharp(outWarped, outFinal, outFilterV, clData, factor);

	} catch (const Error& err) {
		errorLogger().logError("OpenCL output error: ", err.what());
	}
}

void OpenClExecutor::getOutputYuv(int64_t frameIndex, ImageYuvData& image) {
	try {
		//convert to YUV444 for output
		scale_32f8u_3(clData.out[4], clData.yuvOut, mData.cpupitch, clData);
		
		//copy to cpu memory
		Size2 region(image.strideInBytes(), mData.h);
		for (int i = 0; i < 3; i++) {
			Size2 offset(0, mData.h * i);
			clData.queue.enqueueReadBufferRect(clData.yuvOut, CL_TRUE, offset, Size2(), region, mData.cpupitch, 0, 0, 0, image.addr(i, 0, 0));
		}

		//forward image index
		image.setIndex(frameIndex);

	} catch (const Error& err) {
		errorLogger().logError("OpenCL output error: ", err.what());
	}
}

void OpenClExecutor::getOutputRgba(int64_t frameIndex, ImageRGBA& image) {
	try {
		yuv_to_rgba(clData.kernels.yuv32f_to_rgba, clData.out[4], image.plane(0), clData, image.width(), image.height());
		image.setIndex(frameIndex);

	} catch (const Error& err) {
		errorLogger().logError("OpenCL output error: ", err.what());
	}
}

void OpenClExecutor::getOutput(int64_t frameIndex, unsigned char* cudaNv12ptr, int cudaPitch) {
	throw std::runtime_error("not supported");
}

//utility function to read from image
void OpenClExecutor::readImage(Image src, size_t destPitch, void* dest, CommandQueue queue) const {
	size_t w = src.getImageInfo<CL_IMAGE_WIDTH>();
	size_t h = src.getImageInfo<CL_IMAGE_HEIGHT>();
	queue.enqueueReadImage(src, CL_TRUE, Size2(), Size2(w, h), destPitch, 0, dest);
}

//utility function to read from image array
void OpenClExecutor::readImage(Image2DArray src, int idx, size_t destPitch, void* dest, cl::CommandQueue queue) const {
	size_t w = src.getImageInfo<CL_IMAGE_WIDTH>();
	size_t h = src.getImageInfo<CL_IMAGE_HEIGHT>();
	queue.enqueueReadImage(src, CL_TRUE, Size3(0, 0, idx), Size3(w, h, 1ull), destPitch, 0, dest);
}

Matf OpenClExecutor::getPyramid(int64_t frameIndex) const {
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

Matf OpenClExecutor::getTransformedOutput() const {
	std::vector<cl_float4> imageData(1ull * mData.w * mData.h);
	readImage(clData.out[1], mData.w * sizeof(cl_float4), imageData.data(), clData.queue);
	
	Matf warped = Mat<float>::allocate(mData.h * 3ull, mData.w);
	float* ptr = warped.data();
	for (int k = 0; k < 3; k++) {
		for (int i = 0; i < mData.w * mData.h; i++) {
			*ptr = imageData[i].s[k];
			ptr++;
		}
	}
	return warped;
}

void OpenClExecutor::getInput(int64_t frameIndex, ImageYuv& image) const {
	int64_t fr = frameIndex % mData.bufferCount;
	Image im = clData.yuv[fr];
	clData.queue.enqueueReadImage(im, CL_TRUE, Size2(), Size2(image.w, image.h * 3), image.stride, 0, image.data());
}

void OpenClExecutor::getInput(int64_t frameIndex, ImageRGBA& image) const {
	size_t fridx = frameIndex % clData.yuv.size();
	Kernel k = clData.kernels.yuv8u_to_rgba;
	yuv_to_rgba(k, clData.yuv[fridx], image.data(), clData, image.w, image.h);
}

void OpenClExecutor::getWarped(int64_t frameIndex, ImageRGBA& image) {
	Kernel k = clData.kernels.yuv32f_to_rgba;
	yuv_to_rgba(k, clData.out[1], image.data(), clData, image.w, image.h);
}