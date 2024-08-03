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

template <class T> class DeviceInfo;

class CudaFrame : public MovieFrame {

private:
	DeviceInfo<CudaFrame>* device;

public:
	CudaFrame(MainData& data, MovieReader& reader, MovieWriter& writer);

	~CudaFrame();

	void inputData() override;

	void createPyramid(int64_t frameIndex) override;

	void computeStart(int64_t frameIndex) override;

	void computeTerminate(int64_t frameIndex) override;

	void outputData(const AffineTransform& trf) override;

	void getOutput(int64_t frameIndex, ImageYuv& image) override;

	void getOutput(int64_t frameIndex, ImageRGBA& image) override;

	void getOutput(int64_t frameIndex, unsigned char* cudaNv12ptr, int cudaPitch) override;

	Mat<float> getTransformedOutput() const override;

	Mat<float> getPyramid(int64_t index) const override;

	void getInput(int64_t index, ImageYuv& image) const override;

	void getInput(int64_t frameIndex, ImageRGBA& image) override;

	void getWarped(int64_t frameIndex, ImageRGBA& image) override;
	
	MovieFrameId getId() const override;
};