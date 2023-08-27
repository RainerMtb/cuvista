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

//set up device to use
void cl::init(CoreData& core, ImageYuv& inputFrame, std::size_t devIdx, std::vector<int> pyramidRows) {
	clData.w = core.w;
	clData.h = core.h;

	try {
		cl::Device dev = clData.devices[devIdx];
		clData.context = cl::Context(dev);
		clData.queue = cl::CommandQueue(clData.context, dev);

		//allocate yuv data
		for (int idx = 0; idx < core.bufferCount; idx++) {
			cl::ImageFormat fmt(CL_INTENSITY, CL_UNSIGNED_INT8);
			cl::Image2D y(clData.context, 0, fmt, clData.w, clData.h, core.cpupitch);
			cl::Image2D u(clData.context, 0, fmt, clData.w, clData.h, core.cpupitch);
			cl::Image2D v(clData.context, 0, fmt, clData.w, clData.h, core.cpupitch);
			clData.yuv.push_back({ y,u,v });
		}

		//allocate pyramid as individual images
		for (int idx = 0; idx < core.pyramidCount; idx++) {
			std::vector<std::vector<cl::Image2D>> vv;
			for (int k = 0; k < 3; k++) {
				std::vector<cl::Image2D> v;
				for (int z = 0; z <= core.pyramidLevels; z++) {
					int hh = pyramidRows[z];
					cl::ImageFormat fmt(CL_DEPTH, CL_FLOAT);
					cl::Image2D im(clData.context, 0, fmt, clData.w, hh);
					v.push_back(im);
				}
				vv.push_back(v);
			}
			clData.pyr.push_back(vv);
		}

		cl::Program::Sources sources;
		std::string kernelNames[] = { scale_8u32f_kernel };
		for (const std::string& str : kernelNames) {
			sources.emplace_back(str.c_str(), str.size());
		}

		cl::Program program(clData.context, sources);
		program.build();

		clData.scale_8u32f = cl::Kernel(program, "scale_8u32f");

	} catch (const cl::BuildError& err) {
		for (auto& data : err.getBuildLog()) {
			cl::Device dev = data.first;
			std::string msg = data.second;
			errorLogger.logError(msg);
		}

	} catch (const cl::Error& err) {
		errorLogger.logError(err.what());
	}
}

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

void cl::inputData(int64_t frameIdx, const CoreData& core, const ImageYuv& inputFrame) {
	int64_t fr = frameIdx % core.bufferCount;
	size_t frameSize = core.cpupitch * core.h;
	cl::array<cl::size_type, 2> origin = {};
	cl::array<cl::size_type, 2> region = { size_t(clData.w), size_t(clData.h) };
	clData.queue.enqueueWriteImage(clData.yuv[fr][0], CL_TRUE, origin, region, core.cpupitch, 0, inputFrame.data());
	clData.queue.enqueueWriteImage(clData.yuv[fr][1], CL_TRUE, origin, region, core.cpupitch, 0, inputFrame.data() + frameSize);
	clData.queue.enqueueWriteImage(clData.yuv[fr][2], CL_TRUE, origin, region, core.cpupitch, 0, inputFrame.data() + frameSize * 2);
}

void cl::createPyramid(int64_t frameIdx, const CoreData& core) {
	int w = core.w;
	int h = core.h;
	int64_t frIdx = frameIdx % core.bufferCount;
	int64_t pyrIdx = frameIdx % core.pyramidCount;

	scale_8u32f(clData.yuv[frIdx][0], clData.pyr[pyrIdx][0][0], clData);
}

void cl::computePartOne() {}
void cl::computePartTwo() {}
void cl::computeTerminate() {}

void cl::outputData(int64_t frameIdx, const CoreData& core, OutputContext outCtx, cu::Affine trf) {
	int64_t fr = frameIdx % core.bufferCount;
	size_t frameSize = core.cpupitch * core.h;
	cl::array<cl::size_type, 2> origin = {};
	cl::array<cl::size_type, 2> region = { size_t(clData.w), size_t(clData.h) };

	if (outCtx.encodeCpu) {
		clData.queue.enqueueReadImage(clData.yuv[fr][0], CL_TRUE, origin, region, core.cpupitch, 0, outCtx.outputFrame->data());
		clData.queue.enqueueReadImage(clData.yuv[fr][1], CL_TRUE, origin, region, core.cpupitch, 0, outCtx.outputFrame->data() + frameSize);
		clData.queue.enqueueReadImage(clData.yuv[fr][2], CL_TRUE, origin, region, core.cpupitch, 0, outCtx.outputFrame->data() + frameSize * 2);
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
				cl::array<cl::size_type, 2> origin = {};
				cl::array<cl::size_type, 2> region = { im.getImageInfo<CL_IMAGE_WIDTH>(), im.getImageInfo<CL_IMAGE_HEIGHT>() };
				clData.queue.enqueueReadImage(im, CL_TRUE, origin, region, 0, 0, ptr);
				ptr += region[1] * core.w;
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