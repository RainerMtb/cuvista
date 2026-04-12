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

#include <span>
#include "AvxWrapper.hpp"
#include "CoreData.hpp"
#include "Affine2D.hpp"

namespace avx {

	void yuvToRgbaPacked(V16f y, V16f u, V16f v, unsigned char* dest, V16f fu, V16f fv);

	void inv(std::span<V8d> v);
	void inv(std::span<V8d> v, std::span<size_t> piv);

	double norm1(std::span<V8d> v);

	void toConsole(std::span<V8d> v, int digits = 5);

	void toConsole(V8d v, int digits = 5);

	void computeSimilar(std::span<PointBase> points, Matd& M, Affine2D& affine);

	static constexpr unsigned char mask8(int a, int b, int c, int d) {
		return a & 3 | (b & 3) << 2 | (c & 3) << 4 | (d & 3) << 6;
	}

	static constexpr unsigned char mask8(int m0, int m1, int m2, int m3, int m4, int m5, int m6, int m7) {
		return m0 & 1 | (m1 & 1) << 1 | (m2 & 1) << 2 | (m3 & 1) << 3 | (m4 & 1) << 4 | (m5 & 1) << 5 | (m6 & 1) << 6 | (m7 & 1) << 7;
	}
}
