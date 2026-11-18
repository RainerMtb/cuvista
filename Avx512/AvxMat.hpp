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

//specialization of mat for use with avx
template <class T> class AvxMat : public Mat<T> {

public:
	int64_t frameIndex = -1;

	AvxMat() : 
		Mat<T>() 
	{}

	AvxMat(int h, int w) :
		Mat<T>(h, w)
	{}

	AvxMat(int h, int w, T value) : 
		Mat<T>(h, w) 
	{
		this->setValues(value);
	}

	void toConsole(int digits = 5) const {
		Mat<T>::toConsole("avx", digits);
	}
};

using AvxMatf = AvxMat<float>;
using AvxMatf4 = AvxMat<Float4>;
using AvxMatd = AvxMat<double>;