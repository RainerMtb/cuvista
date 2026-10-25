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
#include <algorithm>
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

//------------------- Float4 ------------------

Float4::Float4(const Quartet& other) {
	f0 = other.v;
	f1 = other.u;
	f2 = other.y;
	f3 = other.x;
}

float Float4::operator [] (size_t idx) const {
	assert(idx < 4 && "invalid index");
	switch (idx) {
	case 0: return f0;
	case 1: return f1;
	case 2: return f2;
	case 3: return f3;
	default: return std::numeric_limits<float>::quiet_NaN();
	}
}

Float4 operator + (const Float4& a, const Float4& b) {
	return { a.f0 + b.f0, a.f1 + b.f1, a.f2 + b.f2, a.f3 + b.f3 };
}

Float4 operator - (const Float4& a, const Float4& b) {
	return { a.f0 - b.f0, a.f1 - b.f1, a.f2 - b.f2, a.f3 - b.f3 };
}

Float4 operator * (const Float4& a, const Float4& b) {
	return { a.f0 * b.f0, a.f1 * b.f1, a.f2 * b.f2, a.f3 * b.f3 };
}

Float4 operator * (float f, const Float4& a) {
	return { a.f0 * f, a.f1 * f, a.f2 * f, a.f3 * f };
}

Float4 operator / (const Float4& a, const Float4& b) {
	return { a.f0 / b.f0, a.f1 / b.f1, a.f2 / b.f2, a.f3 / b.f3 };
}

Float4 operator / (float f, const Float4& a) {
	return { a.f0 / f, a.f1 / f, a.f2 / f, a.f3 / f };
}

Float4 std::fma(const Float4& x, const Float4& y, const Float4& z) {
	return { std::fma(x.f0, y.f0, z.f0), std::fma(x.f1, y.f1, z.f1), std::fma(x.f2, y.f2, z.f2), 1.0f };
}

Float4 Float4::clamp(float lo, float hi) const {
	return { std::clamp(f0, lo, hi), std::clamp(f1, lo, hi), std::clamp(f2, lo, hi), std::clamp(f3, lo, hi) };
}

//------------------- Double4 ------------------

double Double4::operator [] (size_t idx) const {
	assert(idx < 4 && "invalid index");
	switch (idx) {
	case 0: return d0;
	case 1: return d1;
	case 2: return d2;
	case 3: return d3;
	default: return std::numeric_limits<float>::quiet_NaN();
	}
}

Double4 operator + (const Double4& a, const Double4& b) {
	return { a.d0 + b.d0, a.d1 + b.d1, a.d2 + b.d2, a.d3 + b.d3 };
}

Double4 operator - (const Double4& a, const Double4& b) {
	return { a.d0 - b.d0, a.d1 - b.d1, a.d2 - b.d2, a.d3 - b.d3 };
}

Double4 operator * (const Double4& a, const Double4& b) {
	return { a.d0 * b.d0, a.d1 * b.d1, a.d2 * b.d2, a.d3 * b.d3 };
}

Double4 operator * (float f, const Double4& a) {
	return { a.d0 * f, a.d1 * f, a.d2 * f, a.d3 * f };
}

Double4 operator / (const Double4& a, const Double4& b) {
	return { a.d0 / b.d0, a.d1 / b.d1, a.d2 / b.d2, a.d3 / b.d3 };
}

Double4 operator / (float f, const Double4& a) {
	return { a.d0 / f, a.d1 / f, a.d2 / f, a.d3 / f };
}

Double4 std::fma(const Double4& x, const Double4& y, const Double4& z) {
	return { std::fma(x.d0, y.d0, z.d0), std::fma(x.d1, y.d1, z.d1), std::fma(x.d2, y.d2, z.d2), 1.0f };
}

Double4 Double4::clamp(double lo, double hi) const {
	return { std::clamp(d0, lo, hi), std::clamp(d1, lo, hi), std::clamp(d2, lo, hi), std::clamp(d3, lo, hi) };
}