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

#include "MovieFrame.hpp"
#include "AvxMat.hpp"
#include "AvxWrapper.hpp"


 //---------------------------------------------------------------------
 //---------- AVX FRAME ------------------------------------------------
 //---------------------------------------------------------------------

class AvxFrame : public MovieFrame {

public:
	AvxFrame(MainData& data, MovieReader& reader, MovieWriter& writer);

	void inputData() override;
	void createPyramid(int64_t frameIndex) override;
	void computeStart(int64_t frameIndex) override;
	void computeTerminate(int64_t frameIndex) override;
	void outputData(const AffineTransform& trf) override;
	void outputCpu(int64_t frameIndex, ImageYuv& image) override;
	void outputCuda(int64_t frameIndex, unsigned char* cudaNv12ptr, int cudaPitch) override;
	Mat<float> getTransformedOutput() const override;
	Mat<float> getPyramid(size_t idx) const override;
	void getInput(int64_t index, ImageYuv& image) const override;
	void getInput(int64_t frameIndex, ImagePPM& image) override;
	void outputRgbWarped(int64_t frameIndex, ImagePPM& image) override;
	std::string getClassName() const override;
	std::string getClassId() const override;

private:
	int walign = 32;
	int pitch;
	std::vector<ImageYuv> mYUV;
	std::vector<AvxMatFloat> mPyr;
	std::vector<AvxMatFloat> mWarped;
	std::vector<AvxMatFloat> mOutput;
	AvxMatFloat mFilterBuffer, mFilterResult, mYuvPlane;

	float filterKernels[3][5] = {
		{ 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f },
		{ 0,       0.25f, 0.5f,   0.25f, 0 },
		{ 0,       0.25f, 0.5f,   0.25f, 0 }
	};

	void unsharp(const AvxMatFloat& warped, AvxMatFloat& gauss, float unsharp, size_t z);
	void write(ImageYuv& dest);
	void write(std::span<unsigned char> nv12, int cudaPitch);
	void downsample(const float* srcptr, int h, int w, int stride, float* destptr, int destStride);
	void filter(const AvxMatFloat& src, int r0, int h, int w, AvxMatFloat& dest, std::span<float> k);
	void filter(std::span<VF16> v, std::span<float> k, AvxMatFloat& dest, int r0, int c0);

	std::pair<VD8, VD8> transform(VD8 x, VD8 y, VD8 m00, VD8 m01, VD8 m02, VD8 m10, VD8 m11, VD8 m12);
	void warpBack(const AffineTransform& trf, const AvxMatFloat& input, AvxMatFloat& dest);

	VF16 interpolate(VF16 f00, VF16 f10, VF16 f01, VF16 f11, VF16 dx, VF16 dy);
	VF16 interpolate(VF16 f00, VF16 f10, VF16 f01, VF16 f11, VF16 dx, VF16 dy, VF16 dx1, VF16 dy1);

	void yuvToFloat(const ImageYuv& yuv, size_t plane, AvxMatFloat& dest);
	void yuvToRgb(const unsigned char* y, const unsigned char* u, const unsigned char* v, int h, int w, int stride, ImagePPM& dest);
	void yuvToRgb(const float* y, const float* u, const float* v, int h, int w, int stride, ImagePPM& dest);
};
