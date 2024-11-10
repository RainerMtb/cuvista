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

#pragma once

#include "AvxMat.hpp"
#include "AvxWrapper.hpp"
#include "Image2.hpp"
#include "Affine2D.hpp"
#include "FrameExecutor.hpp"


 //---------------------------------------------------------------------
 //---------- AVX FRAME ------------------------------------------------
 //---------------------------------------------------------------------


class AvxFrame : public FrameExecutor {

public:
	AvxFrame(CudaData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool);

	void inputData(int64_t frameIndex, const ImageYuv& inputFrame) override;
	void createPyramid(int64_t frameIndex) override ;
	void computeStart(int64_t frameIndex, std::vector<PointResult>& results) override;
	void computeTerminate(int64_t frameIndex, std::vector<PointResult>& results) override;
	void outputData(int64_t frameIndex, const Affine2D& trf) override;
	void getOutput(int64_t frameIndex, ImageYuvData& image) override;
	void getOutput(int64_t frameIndex, ImageRGBA& image) override;
	void getOutput(int64_t frameIndex, unsigned char* cudaNv12ptr, int cudaPitch) override;
	Matf getTransformedOutput() const override;
	Matf getPyramid(int64_t frameIndex) const override;
	void getInput(int64_t frameIndex, ImageYuv& image) const override;
	void getInput(int64_t frameIndex, ImageRGBA& image) const override;
	void getWarped(int64_t frameIndex, ImageRGBA& image) override;

private:
	int walign;  //align widths of matrices to this
	int pitch;   //aligned width of frame
	std::vector<ImageYuv> mYUV;
	std::vector<AvxMatf> mPyr;
	std::vector<AvxMatf> mWarped;
	std::vector<AvxMatf> mOutput;
	AvxMatf mFilterBuffer, mFilterResult, mYuvPlane;

	std::vector<std::vector<V16f>> filterKernels = {
		{ 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f },
		{ 0.0f,    0.25f, 0.5f,   0.25f, 0.0f    },
		{ 0.0f,    0.25f, 0.5f,   0.25f, 0.0f    }
	};

	void unsharp(const AvxMatf& warped, AvxMatf& gauss, float unsharp, AvxMatf& out);
	void write(ImageYuvData& dest);
	void write(std::span<unsigned char> nv12, int cudaPitch);
	void downsample(const float* srcptr, int h, int w, int stride, float* destptr, int destStride);
	void filter(const AvxMatf& src, int r0, int h, int w, AvxMatf& dest, std::span<V16f> ks);

	std::pair<V8d, V8d> transform(V8d x, V8d y, V8d m00, V8d m01, V8d m02, V8d m10, V8d m11, V8d m12);
	void warpBack(const Affine2D& trf, const AvxMatf& input, AvxMatf& dest);

	V16f interpolate(V16f f00, V16f f10, V16f f01, V16f f11, V16f dx, V16f dy);
	V16f interpolate(V16f f00, V16f f10, V16f f01, V16f f11, V16f dx, V16f dy, V16f dx1, V16f dy1);

	void yuvToFloat(const ImageYuv& yuv, size_t plane, AvxMatf& dest);
	void yuvToRgba(const unsigned char* y, const unsigned char* u, const unsigned char* v, int h, int w, int stride, ImageRGBA& dest) const;
	void yuvToRgba(const float* y, const float* u, const float* v, int h, int w, int stride, ImageRGBA& dest) const;

	int align(int base, int alignment);
	V8d sd(int c1, int c2, int y0, int x0, const AvxMatf& Y);
};
