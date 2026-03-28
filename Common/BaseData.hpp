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
	float v, u, y, x;

	float operator [] (size_t idx) const;
};

struct FloatVuyx {
	float v, u, y, x;

	FloatVuyx(float v, float u, float y, float x) : v { v }, u { u }, y { y }, x { x } {}
	FloatVuyx(float f) : FloatVuyx(f, f, f, f) {}
	FloatVuyx() : FloatVuyx(0, 0, 0, 0) {}
	FloatVuyx(const Quartet& other);

	float operator [] (size_t idx) const;
	friend FloatVuyx operator + (const FloatVuyx& a, const FloatVuyx& b);
	friend FloatVuyx operator - (const FloatVuyx& a, const FloatVuyx& b);
	friend FloatVuyx operator * (const FloatVuyx& a, const FloatVuyx& b);
	friend FloatVuyx operator * (float f, const FloatVuyx& a);
	friend FloatVuyx operator / (const FloatVuyx& a, const FloatVuyx& b);
	friend FloatVuyx operator / (float f, const FloatVuyx& a);
};

namespace std {

	FloatVuyx fma(const FloatVuyx& x, const FloatVuyx& y, const FloatVuyx& z);
}