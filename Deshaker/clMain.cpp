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
#include <format>
#include "AVException.hpp"

//data
ClData clData;

using size2 = cl::array<cl::size_type, 2>;

//check available devices
OpenClInfo cl::probeRuntime() {
	clData.devices.clear();
	using size_t = std::size_t;
	OpenClInfo info;
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	if (platforms.size() > 0) {
		cl::Platform pf = platforms[0];
		info.version = pf.getInfo<CL_PLATFORM_VERSION>();
		pf.getDevices(CL_DEVICE_TYPE_GPU, &clData.devices);

		for (int i = 0; i < clData.devices.size(); i++) {
			cl::Device& dev = clData.devices[i];

			cl_bool avail = dev.getInfo<CL_DEVICE_AVAILABLE>();
			if (avail == false) continue;

			cl_int prefWidth = dev.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>();
			if (prefWidth == 0) continue;

			cl_int nativeWidth = dev.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE>();
			if (nativeWidth == 0) continue;

			cl_bool hasImage = dev.getInfo<CL_DEVICE_IMAGE_SUPPORT>();
			if (hasImage == false) continue;

			int64_t maxPixelWidth = dev.getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>();
			int64_t maxPixelHeight = dev.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>();

			cl_device_fp_config doubleConfig = dev.getInfo< CL_DEVICE_DOUBLE_FP_CONFIG>();
			cl_int minDoubleConfig = CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN | CL_FP_DENORM;
			if ((doubleConfig & minDoubleConfig) == 0) continue;

			cl_ulong localMemSize = dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();

			int64_t maxPixel = localMemSize / sizeof(float);
			if (maxPixelWidth < maxPixel) maxPixel = maxPixelWidth;
			if (maxPixelHeight < maxPixel) maxPixel = maxPixelHeight;

			DeviceInfoCl devInfo(DeviceType::OPEN_CL, i, maxPixel, &dev);
			info.devices.push_back(devInfo);
		}
	}
	return info;
}

//set up device to use
void cl::init(CoreData& core, ImageYuv& inputFrame, std::size_t devIdx) {
	try {
		cl::Device dev = clData.devices[devIdx];
		clData.context = cl::Context(dev);
		clData.queue = cl::CommandQueue(clData.context, dev);

		//allocate yuv data
		for (int idx = 0; idx < core.bufferCount; idx++) {
			cl::ImageFormat fmt(CL_R, CL_UNSIGNED_INT8);
			cl::Image2D y(clData.context, 0, fmt, core.w, core.h, core.cpupitch);
			cl::Image2D u(clData.context, 0, fmt, core.w, core.h, core.cpupitch);
			cl::Image2D v(clData.context, 0, fmt, core.w, core.h, core.cpupitch);
			clData.yuv.push_back({ y,u,v });
		}

		auto pyramidCreator = [&] (int levels, int h, int w, std::vector<cl::Image2D>& dest) {
			int hh = h;
			int ww = w;
			for (int z = 0; z < levels; z++) {
				cl::ImageFormat fmt(CL_R, CL_FLOAT);
				dest[z] = cl::Image2D(clData.context, 0, fmt, ww, hh);
				hh /= 2;
				ww /= 2;
			}
		};

		//allocate pyramid as individual images
		clData.pyr.resize(core.pyramidCount);
		for (size_t idx = 0; idx < core.pyramidCount; idx++) {
			clData.pyr[idx].resize(3); // Y - DX - DY
			for (int k = 0; k < 3; k++) {
				clData.pyr[idx][k].resize(core.pyramidLevels); //levels 0..zMax
				pyramidCreator(core.pyramidLevels, core.h, core.w, clData.pyr[idx][k]);
			}
		}

		//buffer pyramid for filtering
		clData.pyrBuffer.resize(2);
		for (size_t idx = 0; idx < 2; idx++) {
			clData.pyrBuffer[idx].resize(core.pyramidLevels);
			pyramidCreator(core.pyramidLevels, core.h, core.w, clData.pyrBuffer[idx]);
		}

		//buffer to hold one filter kernel
		clData.filterKernel = cl::Buffer(clData.context, CL_MEM_READ_ONLY, sizeof(float) * FilterKernel::maxSize);
		
		//compile kernels
		cl::Program::Sources sources;
		std::string kernelNames[] = { kernelsInputOutput };
		for (const std::string& str : kernelNames) {
			sources.emplace_back(str.c_str(), str.size());
		}

		cl::Program program(clData.context, sources);
		program.build(dev, "-cl-opt-disable");

		clData.scale_8u32f = cl::Kernel(program, "scale_8u32f");
		clData.filter_32f_h = cl::Kernel(program, "filter_32f_h");
		clData.filter_32f_v = cl::Kernel(program, "filter_32f_v");
		clData.remap_downsize_32f = cl::Kernel(program, "remap_downsize_32f");

	} catch (const cl::BuildError& err) {
		for (auto& data : err.getBuildLog()) {
			cl::Device dev = data.first;
			std::string msg = data.second;
			errorLogger.logError("init: ", msg);
		}

	} catch (const cl::Error& err) {
		errorLogger.logError("init: ", err.what());
	}
}

void cl::inputData(int64_t frameIdx, const CoreData& core, const ImageYuv& inputFrame) {
	int64_t fr = frameIdx % core.bufferCount;
	int frameSize = core.cpupitch * core.h;
	size2 origin = {};
	size2 region = { size_t(core.w), size_t(core.h) };
	const unsigned char* ptr = inputFrame.data();
	for (int i = 0; i < 3; i++) {
		clData.queue.enqueueWriteImage(clData.yuv[fr][i], CL_TRUE, origin, region, core.cpupitch, 0, ptr);
		ptr += frameSize;
	}
}

void cl::createPyramid(int64_t frameIdx, const CoreData& core) {
	int w = core.w;
	int h = core.h;
	int64_t frIdx = frameIdx % core.bufferCount;
	int64_t pyrIdx = frameIdx % core.pyramidCount;

	try {
		scale_8u32f(clData.yuv[frIdx][0], clData.pyr[pyrIdx][0][0], clData);
		for (size_t z = 1; z <= core.zMax; z++) {
			const FilterKernel& fk = core.filterKernels[0]; //gauss size 5 
			filter_32f_h(clData.pyr[pyrIdx][0][z - 1], clData.pyrBuffer[0][z - 1], fk.k, fk.siz, clData);
			filter_32f_v(clData.pyrBuffer[0][z - 1], clData.pyrBuffer[1][z - 1], fk.k, fk.siz, clData);
			remap_downsize_32f(clData.pyrBuffer[1][z - 1], clData.pyr[pyrIdx][0][z], clData);
		}
		for (size_t z = 0; z <= core.zMax; z++) {
			const FilterKernel& fk = core.filterKernels[3]; //delta
			filter_32f_h(clData.pyr[pyrIdx][0][z], clData.pyr[pyrIdx][1][z], fk.k, fk.siz, clData);
			filter_32f_v(clData.pyr[pyrIdx][0][z], clData.pyr[pyrIdx][2][z], fk.k, fk.siz, clData);
		}

	} catch (const cl::Error& err) {
		errorLogger.logError("pyramid: ", err.what());
	}
}

void cl::computePartOne() {}
void cl::computePartTwo() {}
void cl::computeTerminate() {}

void cl::outputData(int64_t frameIdx, const CoreData& core, OutputContext outCtx, cu::Affine trf) {
	int64_t fr = frameIdx % core.bufferCount;
	int frameSize = core.cpupitch * core.h;
	size2 origin = {};
	size2 region = { size_t(core.w), size_t(core.h) };

	if (outCtx.encodeCpu) {
		unsigned char* ptr = outCtx.outputFrame->data();
		for (int i = 0; i < 3; i++) {
			clData.queue.enqueueReadImage(clData.yuv[fr][i], CL_TRUE, origin, region, core.cpupitch, 0, ptr);
			ptr += frameSize;
		}
	}
}

void cl::shutdown() {
	clData.yuv.clear();
	clData.pyr.clear();
}

ImageYuv cl::getInput(int64_t idx) {
	return {};
}

Matf cl::getTransformedOutput() {
	return Matf();
}

void cl::getPyramid(float* pyramid, size_t idx, const CoreData& core) {
	size_t pyrIdx = idx % core.pyramidCount;
	size_t wbytes = core.w * sizeof(float);

	try {
		float* ptr = pyramid;
		for (int k = 0; k < 3; k++) {
			for (int z = 0; z < core.pyramidLevels; z++) {
				cl::Image im = clData.pyr[pyrIdx][k][z];
				size2 origin = {};
				size_t w = im.getImageInfo<CL_IMAGE_WIDTH>();
				size_t h = im.getImageInfo<CL_IMAGE_HEIGHT>();
				size2 region = { w, h };
				clData.queue.enqueueReadImage(im, CL_TRUE, origin, region, core.w * sizeof(float), 0, ptr);
				ptr += h * core.w;
			}
		}

	} catch (const cl::Error& err) {
		errorLogger.logError("OpenCL get pyramid: ", err.what());
	}
}

bool cl::getCurrentInputFrame(ImagePPM& image) {
	return false;
}

bool cl::getCurrentOutputFrame(ImagePPM& image) {
	return false;
}