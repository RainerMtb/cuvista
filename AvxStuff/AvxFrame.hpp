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

	void init() override;
	Image8& inputDestination(int64_t frameIndex) override;
	void inputData(int64_t frameIndex) override;
	int64_t createPyramid(int64_t frameIndex, AffineDataFloat trf = {}, bool warp = false) override;
	void adjustPyramid(int64_t frameIndex, double gamma) override;
	void computeStart(int64_t frameIndex, std::span<PointResult> results) override;
	void computeTerminate(int64_t frameIndex, std::span<PointResult> results) override;
	void outputData(int64_t frameIndex, AffineDataFloat trf) override;
	void getOutput(int64_t frameIndex, Image8& image) const override;
	bool getOutput(int64_t frameIndex, Image8& image, int cudaNv12stride, unsigned char* cudaNv12ptr) const override;
	Matf getTransformedOutput() const override;
	Matf getPyramid(int64_t frameIndex) const override;
	void getInput(int64_t frameIndex, Image8& image) const override;

private:
	ImageYuv mReadBuffer;
	std::vector<ImageYuv> mInput;
	std::vector<AvxMatf> mPyr;
	AvxMatf mFilterBuffer, mFilterResult;
	AvxMatf mBackground, mFilterBuffer4, mFilterResult4, mWarped, mOutput;
	std::vector<int> luma;

	using uchar = unsigned char;

	std::vector<std::vector<V16f>> mFilterKernels = {
		{ 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f },
		{ 0.0f,    0.25f, 0.5f,   0.25f, 0.0f    },
		{ 0.0f,    0.25f, 0.5f,   0.25f, 0.0f    }
	};

	std::vector<V16f> mFilterKernels4 = {
		{ 0.0f,  0.0f , 0.0625f, 0.0f },
		{ 0.25f, 0.25f, 0.25f,   0.0f },
		{ 0.5f,  0.5f , 0.375f,  0.0f },
		{ 0.25f, 0.25f, 0.25f,   0.0f },
		{ 0.0f,  0.0f , 0.0625f, 0.0f }
	};

	//factors for conversion yuv to rgb
	std::vector<float> mFactorU = { 0.0f, -0.392f, 2.017f, 0.0f };
	std::vector<float> mFactorV = { 1.596f, -0.813f, 0.0f, 0.0f };

	void yuvToFloat4(const ImageYuv& yuv, AvxMatf& dest);
	void downsample(const float* srcptr, int h, int w, int stride, float* destptr, int destStride);
	void filter1(const AvxMatf& src, int h, int w, AvxMatf& dest, std::span<V16f> ks);
	void filter4(const AvxMatf& src, int h, int w, AvxMatf& dest);

	void warpBack1(const AffineDataFloat& trf, const AvxMatf& input, AvxMatf& dest);
	void warpBack4(const AffineDataFloat& trf, const AvxMatf& input, AvxMatf& dest);
	void unsharp4(const AvxMatf& warped, AvxMatf& gauss, AvxMatf& out);

	V16f interpolate(V16f f00, V16f f10, V16f f01, V16f f11, V16f dx, V16f dy);
	V16f interpolate(V16f f00, V16f f10, V16f f01, V16f f11, V16f dx, V16f dy, V16f dx1, V16f dy1);

	void yuvToRgba(const ImageYuv& yuv, Image8& dest) const;
	void vuyxToRgba(const AvxMatf& vuyx, Image8& dest) const;
	void writeVuyx(Image8& dest) const;
	void writeYuv(Image8& dest) const;
	void writeNV12(Image8& dest) const;

	V8d sd(int c1, int c2, int y0, int x0, const AvxMatf& Y);
};
