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
#include "immintrin.h"
#include "AvxWrapper.hpp"
#include "CoreData.hpp"
#include "Affine2D.hpp"

namespace avx {

	void transpose16x8(std::span<VF16> data);

	void transpose16x4(std::span<VF16> data);

	__m128i yuvToRgbaPacked(VF4 y, VF4 u, VF4 v);

	void inv(std::span<VD8> v);

	double norm1(std::span<VD8> v);

	void toConsole(std::span<VD8> v, int digits = 5);

	void toConsole(VD8 v, int digits = 5);

	void computeSimilar(std::span<PointBase> points, Matd& M, Affine2D& affine);
}