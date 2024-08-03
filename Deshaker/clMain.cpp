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

#include <format>
#include <algorithm>
#include <regex>


//data structure
cl::Data clData;

bool operator == (const cl::ImageFormat& a, const cl::ImageFormat& b) {
	return a.image_channel_data_type == b.image_channel_data_type && a.image_channel_order == b.image_channel_order;
}

//check available devices
void cl::probeRuntime(OpenClInfo& info) {
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
			DeviceInfo<OpenClFrame> devInfo(DeviceType::OPEN_CL, maxPixel);
			devInfo.device = dev;
			devInfo.versionDevice = versionDevice;
			devInfo.versionC = versionC;
			devInfo.pitch = pitch;
			devInfo.extensions = extensions;
			info.devices.push_back(devInfo);
		}
	}
}

//set up device to use
void cl::init(CoreData& core, ImageYuv& inputFrame, const DeviceInfoBase* device) {
	assert(device->type == DeviceType::OPEN_CL && "device type must be OpenCL here");
	const DeviceInfo<OpenClFrame>* devInfo = static_cast<const DeviceInfo<OpenClFrame>*>(device);

	try {
		clData.context = Context(devInfo->device);
		clData.queue = CommandQueue(clData.context, devInfo->device);
		clData.secondQueue = CommandQueue(clData.context, devInfo->device);

		//supported image formats
		std::vector<ImageFormat> fmts;
		clData.context.getSupportedImageFormats(CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, &fmts);

		//constant core data structure
		KernelData data = { core.COMP_MAX_TOL, core.deps, core.dmin, core.dmax, core.dnan, 
			core.w, core.h, core.ir, core.iw, core.zMin, core.zMax, core.COMP_MAX_ITER, core.pyramidRowCount };
		clData.core = Buffer(clData.context, CL_MEM_READ_ONLY, sizeof(KernelData));
		clData.queue.enqueueWriteBuffer(clData.core, CL_FALSE, 0, sizeof(KernelData), &data);

		//allocate yuv data
		ImageFormat fmt8(CL_R, CL_UNSIGNED_INT8);
		for (int idx = 0; idx < core.bufferCount; idx++) {
			Image2D yuv(clData.context, CL_MEM_READ_ONLY, fmt8, core.w, core.h * 3ull);
			clData.yuv.push_back(yuv);
		}

		//output images
		ImageFormat outFmt(CL_RGBA, CL_FLOAT);
		for (Image2D& im : clData.out) {
			im = Image2D(clData.context, CL_MEM_READ_WRITE, outFmt, core.w, core.h);
			cl_float4 bg = { core.bgcol_yuv.colors[0], core.bgcol_yuv.colors[1], core.bgcol_yuv.colors[2], 0.0f };
			clData.queue.enqueueFillImage(im, bg, Size2(), Size2(core.w, core.h));
		}
		clData.yuvOut = Buffer(clData.context, CL_MEM_WRITE_ONLY, 3ull * core.cpupitch * core.h);
		clData.rgbaOut = Buffer(clData.context, CL_MEM_WRITE_ONLY, 4ull * core.w * core.h);

		//image format gray single channel float
		ImageFormat fmt32(CL_DEPTH, CL_FLOAT);

		//buffer pyramid for filtering
		for (int z = 0; z < core.pyramidLevels; z++) {
			int hh = core.h >> z;
			int ww = core.w >> z;
			BufferImages buf = {
				Image2D(clData.context, CL_MEM_READ_WRITE, fmt32, ww, hh),
				Image2D(clData.context, CL_MEM_READ_WRITE, fmt32, ww, hh),
				Image2D(clData.context, CL_MEM_READ_WRITE, fmt32, ww, hh)
			};
			clData.buffer.push_back(buf);
		}

		//allocate pyramid, one image for all levels
		for (size_t idx = 0; idx < core.pyramidCount; idx++) {
			Image2D im = Image2D(clData.context, CL_MEM_READ_WRITE, fmt32, core.w, core.pyramidRowCount);
			clData.pyramid.push_back(im);
		}

		//point results
		clData.results = Buffer(clData.context, CL_MEM_WRITE_ONLY, sizeof(cl_PointResult) * core.resultCount);
		clData.cl_results.resize(core.resultCount);

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
			errorLogger.logError("OpenCL build error:\n", msg);
		}

	} catch (const Error& err) {
		errorLogger.logError("OpenCL init error: ", err.what());
	}
}

void cl::shutdown(const CoreData& core) {
	clData = {};
}

//----------------------------------
//-------- INPUT YUV DATA ----------
//----------------------------------

void cl::inputData(int64_t frameIdx, const CoreData& core, const ImageYuv& inputFrame) {
	int64_t fr = frameIdx % core.bufferCount;
	try {
		clData.queue.enqueueWriteImage(clData.yuv[fr], CL_TRUE, Size2(), Size2(core.w, core.h * 3), core.cpupitch, 0, inputFrame.data());
	
	} catch (const Error& err) {
		errorLogger.logError("OpenCL input error: ", err.what());
	}
}

//----------------------------------
//-------- CREATE PYRAMID ----------
//----------------------------------

void cl::createPyramid(int64_t frameIdx, const CoreData& core) {
	//util::ConsoleTimer ic("ocl pyramid");
	int w = core.w;
	int h = core.h;
	int64_t frIdx = frameIdx % core.bufferCount;
	int64_t pyrIdx = frameIdx % core.pyramidCount;

	try {
		//convert yuv image to first level of Y pyramid
		Image& im = clData.buffer[0].result;
		scale_8u32f_1(clData.yuv[frIdx], im, clData);
		clData.queue.enqueueCopyImage(im, clData.pyramid[pyrIdx], Size2(), Size2(), Size2(w, h));

		//lower levels of pyramid
		size_t row = h;
		for (size_t z = 1; z < core.pyramidLevels; z++) {
			Image& src = clData.buffer[z - 1].result;
			Image& dest = clData.buffer[z].result;
			Image& buf1 = clData.buffer[z - 1].filterH;
			Image& buf2 = clData.buffer[z - 1].filterV;
			filter_32f_h1(src, buf1, 0, clData);
			filter_32f_v1(buf1, buf2, 0, clData);
			remap_downsize_32f(buf2, dest, clData);
			int ww = w >> z;
			int hh = h >> z;
			clData.queue.enqueueCopyImage(dest, clData.pyramid[pyrIdx], Size2(), Size2(0ull, row), Size2(ww, hh));
			row += hh;
		}

	} catch (const Error& err) {
		errorLogger.logError("OpenCL pyramid error: ", err.what());
	}
}

//----------------------------------
//-------- COMPUTE -----------------
//----------------------------------

void cl::compute(int64_t frameIdx, const CoreData& core, int rowStart, int rowEnd) {
	//util::ConsoleTimer timer("ocl compute start");
	assert(frameIdx > 0 && "invalid pyramid index");
	int64_t pyrIdx = frameIdx % core.pyramidCount;
	int64_t pyrIdxPrev = (frameIdx - 1) % core.pyramidCount;

	try {
		//local memory size in bytes
		int memsiz = (7LL * core.iw * core.iw + 108) * sizeof(cl_double) + 6 * sizeof(cl_double*);
		//set up compute kernel
		Kernel& kernel = clData.kernels.compute;
		kernel.setArg(0, (cl_long) frameIdx);
		kernel.setArg(1, clData.pyramid[pyrIdxPrev]);
		kernel.setArg(2, clData.pyramid[pyrIdx]);
		kernel.setArg(3, clData.results);
		kernel.setArg(4, clData.core);
		kernel.setArg(5, memsiz, nullptr);
		kernel.setArg(6, rowStart);

		//threads
		NDRange ndlocal = NDRange(core.iw, 32 / core.iw); //based on cuda warp
		NDRange ndglobal = NDRange(ndlocal[0] * core.ixCount, ndlocal[1] * (rowEnd - rowStart));
		clData.queue.enqueueNDRangeKernel(kernel, NullRange, ndglobal, ndlocal);

	} catch (const Error& err) {
		errorLogger.logError("OpenCL compute error: ", err.what());
	}
}

void cl::computeStart(int64_t frameIdx, const CoreData& core) {
	try {
		//reset computed flag
		clData.queue.enqueueFillBuffer<cl_char>(clData.results, 0, 0, sizeof(cl_PointResult) * core.resultCount);

	} catch (const Error& err) {
		errorLogger.logError("OpenCL compute error: ", err.what());
	}
	//compute first part of points
	compute(frameIdx, core, 0, core.iyCount / 4);
}

void cl::computeTerminate(int64_t frameIdx, const CoreData& core, std::vector<PointResult>& results) {
	//util::ConsoleTimer timer("ocl compute end");

	//compute rest of points
	compute(frameIdx, core, core.iyCount / 4, core.iyCount);
	
	try {
		//copy results from device to host buffer
		clData.queue.enqueueReadBuffer(clData.results, CL_TRUE, 0, sizeof(cl_PointResult) * core.resultCount, clData.cl_results.data());

		//convert from cl_PointResult to PointResult
		for (size_t i = 0; i < results.size(); i++) {
			cl_PointResult& pr = clData.cl_results[i];
			results[i] = { pr.idx, pr.ix0, pr.iy0, pr.xm, pr.ym, pr.xm - core.w / 2, pr.ym - core.h / 2, pr.u, pr.v, PointResultType(pr.result), pr.z };
		}

	} catch (const Error& err) {
		errorLogger.logError("OpenCL compute error: ", err.what());
	}
}

//----------------------------------
//-------- OUTPUT STABILIZED -------
//----------------------------------

void cl::outputData(int64_t frameIdx, const CoreData& core, std::array<double, 6> trf) {
	//util::ConsoleTimer timer("ocl output");
	int64_t frIdx = frameIdx % core.bufferCount;
	auto& [outStart, outWarped, outFilterH, outFilterV, outFinal] = clData.out;

	try {
		//convert input yuv to float image
		scale_8u32f_3(clData.yuv[frIdx], outStart, clData);
		//fill static background when requested
		if (core.bgmode == BackgroundMode::COLOR) {
			cl_float4 bg = { core.bgcol_yuv.colors[0], core.bgcol_yuv.colors[1], core.bgcol_yuv.colors[2], 0.0f };
			clData.queue.enqueueFillImage(outWarped, bg, Size2(), Size2(core.w, core.h));
		}
		//warp input on top of background
		warp_back(outStart, outWarped, clData, trf);

		//filtering
		filter_32f_h3(outWarped, outFilterH, clData);
		filter_32f_v3(outFilterH, outFilterV, clData);

		//unsharp mask
		cl_float4 factor = { core.unsharp.y, core.unsharp.u, core.unsharp.v };
		unsharp(outWarped, outFinal, outFilterV, clData, factor);

	} catch (const Error& err) {
		errorLogger.logError("OpenCL output error: ", err.what());
	}
}

void cl::outputDataCpu(int64_t frameIndex, const CoreData& core, ImageYuv& image) {
	try {
		//convert to YUV444 for output
		scale_32f8u_3(clData.out[4], clData.yuvOut, core.cpupitch, clData);
		//copy to cpu memory
		clData.queue.enqueueReadBuffer(clData.yuvOut, CL_TRUE, 0, 3ull * core.cpupitch * core.h, image.data());
		image.index = frameIndex;

	} catch (const Error& err) {
		errorLogger.logError("OpenCL output error: ", err.what());
	}
}

void cl::outputDataCpu(int64_t frameIndex, const CoreData& core, ImageRGBA& image) {
	try {
		yuv_to_rgba(clData.kernels.yuv32f_to_rgba, clData.out[4], image.data(), clData, image.w, image.h);
		image.index = frameIndex;

	} catch (const Error& err) {
		errorLogger.logError("OpenCL output error: ", err.what());
	}
}

//utility function to read from image
void cl::readImage(Image src, size_t destPitch, void* dest, CommandQueue queue) {
	size_t w = src.getImageInfo<CL_IMAGE_WIDTH>();
	size_t h = src.getImageInfo<CL_IMAGE_HEIGHT>();
	queue.enqueueReadImage(src, CL_TRUE, Size2(), Size2(w, h), destPitch, 0, dest);
}

void cl::getPyramid(float* pyramid, int64_t index, const CoreData& core) {
	size_t pyrIdx = index % core.pyramidCount;
	size_t wbytes = core.w * sizeof(float);

	try {
		readImage(clData.pyramid[pyrIdx], wbytes, pyramid, clData.queue);

	} catch (const Error& err) {
		errorLogger.logError("OpenCL get pyramid: ", err.what());
	}
}

Matf cl::getTransformedOutput(const CoreData& core) {
	std::vector<cl_float4> imageData(1ull * core.w * core.h);
	readImage(clData.out[1], core.w * sizeof(cl_float4), imageData.data(), clData.queue);
	
	Matf warped = Mat<float>::allocate(core.h * 3ull, core.w);
	float* ptr = warped.data();
	for (int k = 0; k < 3; k++) {
		for (int i = 0; i < core.w * core.h; i++) {
			*ptr = imageData[i].s[k];
			ptr++;
		}
	}
	return warped;
}

void cl::getInput(int64_t idx, ImageYuv& image, const CoreData& core) {
	int64_t fr = idx % core.bufferCount;
	Image im = clData.yuv[fr];
	clData.queue.enqueueReadImage(im, CL_TRUE, Size2(), Size2(image.w, image.h * 3), image.stride, 0, image.data());
}

void cl::getCurrentInputFrame(ImageRGBA& image, int64_t idx) {
	size_t fridx = idx % clData.yuv.size();
	yuv_to_rgba(clData.kernels.yuv8u_to_rgba, clData.yuv[fridx], image.data(), clData, image.w, image.h);
}

void cl::getTransformedOutput(ImageRGBA& image) {
	yuv_to_rgba(clData.kernels.yuv32f_to_rgba, clData.out[1], image.data(), clData, image.w, image.h);
}