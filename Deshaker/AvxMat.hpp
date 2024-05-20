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

#include "Mat.h"


//specialization of mat for use with avx
class AvxMatFloat : protected CoreMat<float> {
public:
	int64_t frameIndex = -1;

	AvxMatFloat() : CoreMat<float>() {}
	AvxMatFloat(int h, int w) : CoreMat<float>(h, w) {}
	AvxMatFloat(int h, int w, float value) : CoreMat<float>(h, w, value) {}

	int w() const { return int(CoreMat::w); }
	int h() const { return int(CoreMat::h); }

	float& at(int row, int col) { return CoreMat::at(row, col); }
	const float& at(int row, int col) const { return CoreMat::at(row, col); }
	float* addr(int row, int col) { return CoreMat::addr(row, col); }
	const float* addr(int row, int col) const { return CoreMat::addr(row, col); }
	float* data() { return CoreMat::data(); }
	const float* data() const { return CoreMat::data(); }

	float* row(int r) { return addr(r, 0); }
	const float* row(int r) const { return addr(r, 0); }

	void fill(float value) { 
		std::fill(array, array + numel(), value); 
	}

	void saveAsBinary(const std::string& filename) const { 
		Matf::fromArray(h(), w(), array, false).saveAsBinary(filename); 
	}

	void saveAsBMP(const std::string& filename) const {
		im::ImageMatShared<float>(h(), w(), w(), array).saveAsBMP(filename, 255.0f);
	}

	Matf copyToMatf() const { 
		return Matf::fromRowData(h(), w(), w(), array); 
	}

	Matf toMatf() const { 
		return Matf::fromArray(h(), w(), array, false); 
	}
};
