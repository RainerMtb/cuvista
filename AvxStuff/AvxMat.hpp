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

#include "Mat.hpp"
#include "Image.hpp"


//specialization of mat for use with avx
template <class T> class AvxMat : protected CoreMat<T> {
public:
	int64_t frameIndex = -1;

	AvxMat() : CoreMat<T>() {}
	AvxMat(int h, int w) : CoreMat<T>(h, w) {}
	AvxMat(int h, int w, T value) : CoreMat<T>(h, w, value) {}

	int w() const { return int(CoreMat<T>::w); }
	int h() const { return int(CoreMat<T>::h); }

	T& at(int row, int col) { return CoreMat<T>::at(row, col); }
	const T& at(int row, int col) const { return CoreMat<T>::at(row, col); }
	T* addr(int row, int col) { return CoreMat<T>::addr(row, col); }
	const T* addr(int row, int col) const { return CoreMat<T>::addr(row, col); }
	T* data() { return CoreMat<T>::data(); }
	const T* data() const { return CoreMat<T>::data(); }

	T* row(int r) { return addr(r, 0); }
	const T* row(int r) const { return addr(r, 0); }

	void fill(T value) { 
		std::fill(this->array, this->array + this->numel(), value); 
	}

	void saveAsBinary(const std::string& filename) const { 
		Mat<T>::fromArray(h(), w(), this->array, false).saveAsBinary(filename); 
	}

	void saveAsBMP(const std::string& filename) const {
		im::ImageMatShared<T>(h(), w(), w(), this->array).saveAsBMP(filename, 255);
	}

	Mat<T> copyToMat() const { 
		return Mat<T>::fromRowData(h(), w(), w(), this->array); 
	}

	Mat<T> toMat() const { 
		return Mat<T>::fromArray(h(), w(), this->array, false); 
	}

	void toConsole(int digits = 5) const {
		toMat().toConsole("avx", digits);
	}
};

using AvxMatf = AvxMat<float>;
using AvxMatd = AvxMat<double>;