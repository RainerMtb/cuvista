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

 //---------------------------------------------------------------------
 //---------- AVX FRAME ------------------------------------------------
 //---------------------------------------------------------------------

class AvxMatFloat : public CoreMat<float> {
public:
	AvxMatFloat() : CoreMat<float>() {}
	AvxMatFloat(int h, int w) : CoreMat<float>(h, w) {}
	AvxMatFloat(int h, int w, float value) : CoreMat<float>(h, w, value) {}

	int w() const { return int(CoreMat::w); }
	int h() const { return int(CoreMat::h); }
	float* row(int r) { return addr(r, 0); }
	void saveAsBinary(const std::string& filename) { Matf::fromArray(h(), w(), array, false).saveAsBinary(filename); }
};

class AvxFrame : public MovieFrame {

public:
	AvxFrame(MainData& data, MovieReader& reader, MovieWriter& writer);

	void inputData() override;
	void createPyramid(int64_t frameIndex) override;
	void computeStart(int64_t frameIndex) override;
	void computeTerminate(int64_t frameIndex) override;
	void outputData(const AffineTransform& trf, OutputContext outCtx) override;
	Mat<float> getTransformedOutput() const override;
	Mat<float> getPyramid(size_t idx) const override;
	ImageYuv getInput(int64_t index) const override;
	void getInputFrame(int64_t frameIndex, ImagePPM& image) override;
	void getTransformedOutput(int64_t frameIndex, ImagePPM& image) override;
	std::string getClassName() const override;
	std::string getClassId() const override;

private:
	std::vector<ImageYuv> mYUV;
	std::vector<AvxMatFloat> mPyr;
	AvxMatFloat mFilterBuffer, mFilterResult;

	__m512 filterKernels[3] = {
		_mm512_setr_ps(0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f, 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f, 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f, 0),
		_mm512_setr_ps(0, 0.25f, 0.5f, 0.25f, 0, 0, 0, 0, 0, 0.25f, 0.5f, 0.25f, 0, 0, 0, 0),
		_mm512_setr_ps(0, 0.25f, 0.5f, 0.25f, 0, 0, 0, 0, 0, 0.25f, 0.5f, 0.25f, 0, 0, 0, 0)
	};

	void downsample(const float* srcptr, int h, int w, int stride, float* destptr, int destStride);
	void filter(const float* srcptr, int h, int w, int stride, float* destptr, int destStride, __m512 k);
	float* filterTriple(__m512i index, __m512 input, __m512 k, float* dest, int destStride);
	float* filterVector(__m512i index, __m512 input, __m512 k, float* dest, int destStride);

	float sum(__m512 a, int from, int to);
	float sum(__m256 a, int from, int to);
};
