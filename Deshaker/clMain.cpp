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
#include "AVException.hpp"

#include <format>
#include <algorithm>
#include <regex>


//data structure
cl::Data clData;

bool operator == (const cl::ImageFormat& a, const cl::ImageFormat& b) {
	return a.image_channel_data_type == b.image_channel_data_type && a.image_channel_order == b.image_channel_order;
}

//check available devices
OpenClInfo cl::probeRuntime() {
	OpenClInfo info;
	std::vector<Platform> platforms;
	Platform::get(&platforms);

	for (Platform& platform : platforms) {
		info.version = platform.getInfo<CL_PLATFORM_VERSION>();
		std::vector<Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

		for (Device& dev : devices) {
			cl_bool avail = dev.getInfo<CL_DEVICE_AVAILABLE>();
			if (avail == false) continue;

			cl_int prefWidth = dev.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>();
			if (prefWidth == 0) continue;

			cl_int nativeWidth = dev.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE>();
			if (nativeWidth == 0) continue;

			cl_bool hasImage = dev.getInfo<CL_DEVICE_IMAGE_SUPPORT>();
			if (hasImage == false) continue;

			//cl_uint pitch = dev.getInfo<CL_DEVICE_IMAGE_PITCH_ALIGNMENT>();

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

			//we have a valid device
			DeviceInfoCl devInfo(DeviceType::OPEN_CL, maxPixel);
			devInfo.device = dev;
			devInfo.versionDevice = versionDevice;
			devInfo.versionC = versionC;
			info.devices.push_back(devInfo);
		}
	}
	return info;
}

//set up device to use
void cl::init(CoreData& core, ImageYuv& inputFrame, OpenClInfo clinfo, const DeviceInfo* device) {
	assert(device->type == DeviceType::OPEN_CL && "device type must be OpenCL here");
	const DeviceInfoCl* devInfo = static_cast<const DeviceInfoCl*>(device);
	try {
		clData.context = Context(devInfo->device);
		clData.queue = CommandQueue(clData.context, devInfo->device);

		//allocate yuv data
		ImageFormat fmt8(CL_R, CL_UNSIGNED_INT8);
		for (int idx = 0; idx < core.bufferCount; idx++) {
			Image2D yuv(clData.context, CL_MEM_READ_ONLY, fmt8, core.w, core.h * 3ull);
			clData.yuv.push_back(yuv);
		}

		//check supported image formats
		std::vector<ImageFormat> fmts;
		clData.context.getSupportedImageFormats(CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, &fmts);
		ImageFormat outFmt(CL_RGBA, CL_FLOAT);
		bool hasFormatSupport = std::count(fmts.cbegin(), fmts.cend(), outFmt) > 0;
		if (!hasFormatSupport) throw Error(-1, "image format not supported");

		//image format gray single channel float
		ImageFormat fmt32(CL_R, CL_FLOAT);

		//buffer pyramid for filtering, need 3 buffer images on every pyramid level
		int siz = 3;
		clData.pyrBuffer.resize(siz);
		for (size_t idx = 0; idx < siz; idx++) {
			clData.pyrBuffer[idx].resize(core.pyramidLevels);
			for (int z = 0; z < core.pyramidLevels; z++) {
				int hh = core.h >> z;
				int ww = core.w >> z;
				clData.pyrBuffer[idx][z] = Image2D(clData.context, CL_MEM_READ_WRITE, fmt32, ww, hh);
			}
		}

		//allocate pyramid, one image for all levels
		clData.pyr.resize(core.pyramidCount);
		for (size_t idx = 0; idx < core.pyramidCount; idx++) {
			clData.pyr[idx] = { 
				Image2D(clData.context, CL_MEM_READ_WRITE, fmt32, core.w, core.pyramidRowCount),
				Image2D(clData.context, CL_MEM_READ_WRITE, fmt32, core.w, core.pyramidRowCount),
				Image2D(clData.context, CL_MEM_READ_WRITE, fmt32, core.w, core.pyramidRowCount)
			};
		}

		//output images
		for (Image2D& im : clData.out) {
			im = Image2D(clData.context, CL_MEM_READ_WRITE, outFmt, core.w, core.h);
			cl_float4 bg = { core.bgcol_yuv.colors[0], core.bgcol_yuv.colors[1], core.bgcol_yuv.colors[2], 0.0f };
			clData.queue.enqueueFillImage(im, bg, Size2(), Size2(core.w, core.h));
		}
		clData.yuvOut = Buffer(clData.context, CL_MEM_WRITE_ONLY, 3ull * core.cpupitch * core.h);
		clData.rgbOut = Buffer(clData.context, CL_MEM_WRITE_ONLY, 3ull * core.w * core.h);

		//point results
		clData.results = Buffer(clData.context, CL_MEM_WRITE_ONLY, sizeof(cl_PointResult) * core.resultCount);
		clData.cl_results.resize(core.resultCount);

		//compile device code
		Program::Sources sources;
		std::string kernelNames[] = { kernelsInputOutput, kernelCompute };
		for (const std::string& str : kernelNames) {
			sources.emplace_back(str.c_str(), str.size());
		}

		//build program to latest C version
		std::string compilerFlag = std::format("-cl-std=CL{}.{}", devInfo->versionC / 1000, devInfo->versionC % 1000);
		Program program(clData.context, sources);
		program.build();

		//assign kernels to map
		for (auto& entry : clData.kernelMap) {
			entry.second = Kernel(program, entry.first.c_str());
		}

	} catch (const BuildError& err) {
		for (auto& data : err.getBuildLog()) {
			Device dev = data.first;
			std::string msg = data.second;
			errorLogger.logError("OpenCL init error: ", msg);
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
	int w = core.w;
	int h = core.h;
	int64_t frIdx = frameIdx % core.bufferCount;
	int64_t pyrIdx = frameIdx % core.pyramidCount;

	try {
		//convert yuv image to first level of Y pyramid
		Image& im = clData.pyrBuffer[2][0];
		scale_8u32f_1(clData.yuv[frIdx], im, clData);
		clData.queue.enqueueCopyImage(im, clData.pyr[pyrIdx].Y, Size2(), Size2(), Size2(w, h));
		//first level of DX
		filter_32f_h1(im, clData.pyr[pyrIdx].DX, 3, 0, w, h, clData);
		//first level of DY
		filter_32f_v1(im, clData.pyr[pyrIdx].DY, 3, 0, w, h, clData);

		//lower levels of pyramid
		size_t row = h;
		size_t ww = w;
		size_t hh = h;
		for (size_t z = 1; z < core.pyramidLevels; z++) {
			Image& src = clData.pyrBuffer[2][z - 1];
			Image& filterH = clData.pyrBuffer[0][z - 1];
			Image& filterV = clData.pyrBuffer[1][z - 1];
			Image& dest = clData.pyrBuffer[2][z];

			filter_32f_h1(src, filterH, 0, 0, ww, hh, clData);
			filter_32f_v1(filterH, filterV, 0, 0, ww, hh, clData);
			remap_downsize_32f(filterV, dest, clData);

			hh = dest.getImageInfo<CL_IMAGE_HEIGHT>();
			ww = dest.getImageInfo<CL_IMAGE_WIDTH>();
			clData.queue.enqueueCopyImage(dest, clData.pyr[pyrIdx].Y, Size2(), Size2(0ull, row), Size2(ww, hh));
			filter_32f_h1(dest, clData.pyr[pyrIdx].DX, 3, row, ww, hh, clData);
			filter_32f_v1(dest, clData.pyr[pyrIdx].DY, 3, row, ww, hh, clData);

			row += hh;
		}

	} catch (const Error& err) {
		errorLogger.logError("OpenCL pyramid error: ", err.what());
	}
}

//----------------------------------
//-------- COMPUTE -----------------
//----------------------------------

void cl::computePartOne() {}
void cl::computePartTwo() {}

void cl::computeTerminate(int64_t frameIdx, const CoreData& core, std::vector<PointResult>& results) {
	assert(frameIdx > 0 && "invalid pyramid index");
	int64_t pyrIdx = frameIdx % core.pyramidCount;
	int64_t pyrIdxPrev = (frameIdx - 1) % core.pyramidCount;

	try {
		Kernel kernel = clData.kernel("compute");
		kernel.setArg(0, clData.pyr[pyrIdxPrev].Y);
		kernel.setArg(1, clData.pyr[pyrIdxPrev].DX);
		kernel.setArg(2, clData.pyr[pyrIdxPrev].DY);
		kernel.setArg(3, clData.pyr[pyrIdx].Y);
		kernel.setArg(4, clData.results);
		NDRange ndglobal = NDRange(1ull * core.iw * core.ixCount, core.iyCount);
		NDRange ndlocal = NDRange(core.iw, 1);
		clData.queue.enqueueNDRangeKernel(kernel, NullRange, ndglobal, ndlocal);

		//copy from device to host buffer in cl_PointResult
		clData.queue.enqueueReadBuffer(clData.results, CL_TRUE, 0, sizeof(cl_PointResult) * core.resultCount, clData.cl_results.data());

		//convert from cl_PointResult to PointResult
		for (size_t i = 0; i < results.size(); i++) {
			cl_PointResult& pr = clData.cl_results[i];
			results[i] = { pr.idx, pr.ix0, pr.iy0, pr.xm, pr.ym, pr.xm - core.w / 2, pr.ym - core.h / 2, pr.u, pr.v, PointResultType(pr.result) };
		}

	} catch (const Error& err) {
		errorLogger.logError("OpenCL compute error: ", err.what());
	}
}

//----------------------------------
//-------- OUTPUT STABILIZED -------
//----------------------------------

//utility function to read from image
void cl::readImage(Image src, size_t destPitch, void* dest, CommandQueue queue) {
	size_t w = src.getImageInfo<CL_IMAGE_WIDTH>();
	size_t h = src.getImageInfo<CL_IMAGE_HEIGHT>();
	queue.enqueueReadImage(src, CL_TRUE, Size2(), Size2(w, h), destPitch, 0, dest);
}

void cl::outputData(int64_t frameIdx, const CoreData& core, OutputContext outCtx, std::array<double, 6> trf) {
	//ConsoleTimer timer;
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

		//convert to YUV444 for output
		scale_32f8u_3(outFinal, clData.yuvOut, core.cpupitch, clData);

		//copy output to host
		if (outCtx.encodeCpu) {
			clData.queue.enqueueReadBuffer(clData.yuvOut, CL_FALSE, 0, 3ull * core.cpupitch * core.h, outCtx.outputFrame->data());
		}

	} catch (const Error& err) {
		errorLogger.logError("OpenCL output error: ", err.what());
	}
}

ImageYuv cl::getInput(int64_t idx, const CoreData& core) {
	ImageYuv out(core.h, core.w, core.w);
	int64_t fr = idx % core.bufferCount;
	Image im = clData.yuv[fr];
	clData.queue.enqueueReadImage(im, CL_TRUE, Size2(), Size2(core.w, core.h * 3), core.w, 0, out.data());
	return out;
}

void cl::getPyramid(float* pyramid, size_t idx, const CoreData& core) {
	size_t pyrIdx = idx % core.pyramidCount;
	size_t wbytes = core.w * sizeof(float);

	try {
		float* ptr = pyramid;
		for (const Image& im : { clData.pyr[pyrIdx].Y, clData.pyr[pyrIdx].DX, clData.pyr[pyrIdx].DY }) {
			readImage(im, wbytes, ptr, clData.queue);
			ptr += core.pyramidRowCount * core.w;
		}

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

void cl::getCurrentInputFrame(ImagePPM& image, int64_t idx) {
	size_t fridx = idx % clData.yuv.size();
	yuv_to_rgb("yuv8u_to_rgb", clData.yuv[fridx], image.data(), clData, image.w, image.h);
}

void cl::getCurrentOutputFrame(ImagePPM& image) {
	yuv_to_rgb("yuv32f_to_rgb", clData.out[1], image.data(), clData, image.w, image.h);
}