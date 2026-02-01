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
#include "Affine2D.hpp"
#include "FrameExecutor.hpp"


 //---------------------------------------------------------------------
 //---------- AVX FRAME ------------------------------------------------
 //---------------------------------------------------------------------


class AvxFrame : public FrameExecutor {

public:
	AvxFrame(CoreData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool);

	void inputData(int64_t frameIndex, const ImageYuv& inputFrame) override;
	void createPyramid(int64_t frameIndex, AffineDataFloat trf = {}, bool warp = false) override;
	void computeStart(int64_t frameIndex, std::span<PointResult> results) override;
	void computeTerminate(int64_t frameIndex, std::span<PointResult> results) override;
	void outputData(int64_t frameIndex, AffineDataFloat trf) override;
	void getOutputYuv(int64_t frameIndex, ImageYuv& image) const override;
	void getOutputImage(int64_t frameIndex, ImageBaseRgb& image) const override;
	bool getOutputNvenc(int64_t frameIndex, ImageNV12& image, unsigned char* cudaNv12ptr) const override;
	Matf getTransformedOutput() const override;
	Matf getPyramid(int64_t frameIndex) const override;
	void getInput(int64_t frameIndex, ImageYuv& image) const override;
	void getInput(int64_t frameIndex, ImageBaseRgb& image) const override;
	void getWarped(int64_t frameIndex, ImageBaseRgb& image) override;

private:
	int walign;  //align widths of matrices to this
	int pitch;   //aligned width of frame
	std::vector<ImageYuv> mYUV;
	std::vector<AvxMatf> mPyr;
	std::vector<AvxMatf> mWarped;
	std::vector<AvxMatf> mOutput;
	AvxMatf mFilterBuffer, mFilterResult, mYuvPlane;

	using uchar = unsigned char;

	std::vector<std::vector<V16f>> filterKernels = {
		{ 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f },
		{ 0.0f,    0.25f, 0.5f,   0.25f, 0.0f    },
		{ 0.0f,    0.25f, 0.5f,   0.25f, 0.0f    }
	};

	//factors for conversion yuv to rgb
	std::vector<float> fu = { 0.0f, -0.337633f, 1.732446f, 0.0f };
	std::vector<float> fv = { 1.370705f, -0.698001f, 0.0f, 0.0f };

	void unsharp(const AvxMatf& warped, AvxMatf& gauss, float unsharp, AvxMatf& out);
	void write(ImageYuv& dest) const;
	void write(ImageNV12& image) const;
	void downsample(const float* srcptr, int h, int w, int stride, float* destptr, int destStride);
	void filter(const AvxMatf& src, int r0, int h, int w, AvxMatf& dest, std::span<V16f> ks);

	void warpBack(const AffineDataFloat& trf, const AvxMatf& input, AvxMatf& dest);

	V16f interpolate(V16f f00, V16f f10, V16f f01, V16f f11, V16f dx, V16f dy);
	V16f interpolate(V16f f00, V16f f10, V16f f01, V16f f11, V16f dx, V16f dy, V16f dx1, V16f dy1);

	void yuvToFloat(const ImageYuv& yuv, size_t plane, AvxMatf& dest);
	void yuvToRgb(const uchar* y, const uchar* u, const uchar* v, int h, int w, int stride, ImageBaseRgb& dest) const;
	void yuvToRgb(const float* y, const float* u, const float* v, int h, int w, int stride, ImageBaseRgb& dest) const;

	static V16f yuvToRgbLoadUchar(const uchar* src);
	static V16f yuvToRgbLoadFloat(const float* src);

	V8d sd(int c1, int c2, int y0, int x0, const AvxMatf& Y);
};
