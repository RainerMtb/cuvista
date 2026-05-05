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

#include <cmath>
#include <cassert>
#include <limits>
#include "BaseData.hpp"


float Triplet::operator [] (size_t idx) const {
	assert(idx < 3 && "invalid Triplet index");
	switch (idx) {
	case 0: return y;
	case 1: return u;
	case 2: return v;
	default: return std::numeric_limits<float>::quiet_NaN();
	}
}

float Quartet::operator [] (size_t idx) const {
	assert(idx < 4 && "invalid index");
	switch (idx) {
	case 0: return v;
	case 1: return u;
	case 2: return y;
	case 3: return x;
	default: return std::numeric_limits<float>::quiet_NaN();
	}
}

FloatVuyx::FloatVuyx(const Quartet& other) {
	v = other.v;
	u = other.u;
	y = other.y;
	x = other.x;
}

float FloatVuyx::operator [] (size_t idx) const {
	assert(idx < 4 && "invalid index");
	switch (idx) {
	case 0: return v;
	case 1: return u;
	case 2: return y;
	case 3: return x;
	default: return std::numeric_limits<float>::quiet_NaN();
	}
}

FloatVuyx operator + (const FloatVuyx& a, const FloatVuyx& b) {
	return { a.v + b.v, a.u + b.u, a.y + b.y, a.x + b.x };
}

FloatVuyx operator - (const FloatVuyx& a, const FloatVuyx& b) {
	return { a.v - b.v, a.u - b.u, a.y - b.y, a.x - b.x };
}

FloatVuyx operator * (const FloatVuyx& a, const FloatVuyx& b) {
	return { a.v * b.v, a.u * b.u, a.y * b.y, a.x * b.x };
}

FloatVuyx operator * (float f, const FloatVuyx& a) {
	return { a.v * f, a.u * f, a.y * f, a.x * f };
}

FloatVuyx operator / (const FloatVuyx& a, const FloatVuyx& b) {
	return { a.v / b.v, a.u / b.u, a.y / b.y, a.x / b.x };
}

FloatVuyx operator / (float f, const FloatVuyx& a) {
	return { a.v / f, a.u / f, a.y / f, a.x / f };
}

FloatVuyx std::fma(const FloatVuyx& x, const FloatVuyx& y, const FloatVuyx& z) {
	return { std::fma(x.v, y.v, z.v), std::fma(x.u, y.u, z.u), std::fma(x.y, y.y, z.y), 1.0f };
}