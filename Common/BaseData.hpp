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

//the six parameters for an affine transformation
struct AffineDataFloat {
	float m00, m01, m02, m10, m11, m12;
};

//the six parameters for an affine transformation
struct AffineDataDouble {
	double m00, m01, m02, m10, m11, m12;
};

//used in CoreData because in cuda _constant_ allocated symbols must be initialized, no constructors allowed
struct Triplet {
	float y, u, v;

	float operator [] (size_t idx) const;
};

//used in CoreData because in cuda _constant_ allocated symbols must be initialized, no constructors allowed
struct Quartet {
	float a, y, u, v;

	float operator [] (size_t idx) const;
};

struct FloatAyuv {
	float a, y, u, v;

	FloatAyuv(float a, float y, float u, float v) : a { a }, y { y }, u { u }, v { v } {}
	FloatAyuv(float f) : FloatAyuv(f, f, f, f) {}
	FloatAyuv() : FloatAyuv(0, 0, 0, 0) {}
	FloatAyuv(const Quartet& other);

	float operator [] (size_t idx) const;
	friend FloatAyuv operator + (const FloatAyuv& a, const FloatAyuv& b);
	friend FloatAyuv operator - (const FloatAyuv& a, const FloatAyuv& b);
	friend FloatAyuv operator * (const FloatAyuv& a, const FloatAyuv& b);
	friend FloatAyuv operator * (float f, const FloatAyuv& a);
	friend FloatAyuv operator / (const FloatAyuv& a, const FloatAyuv& b);
	friend FloatAyuv operator / (float f, const FloatAyuv& a);
};

namespace std {

	FloatAyuv fma(const FloatAyuv& x, const FloatAyuv& y, const FloatAyuv& z);
}