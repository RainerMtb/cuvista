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

struct Float4 {
	float f0, f1, f2, f3;

	Float4(float f0, float f1, float f2, float f3) : f0 { f0 }, f1 { f1 }, f2 { f2 }, f3 { f3 } {}
	Float4(float f) : Float4(f, f, f, f) {}
	Float4() : Float4(0, 0, 0, 0) {}
	Float4(const Quartet& other);

	float operator [] (size_t idx) const;
	friend Float4 operator + (const Float4& a, const Float4& b);
	friend Float4 operator - (const Float4& a, const Float4& b);
	friend Float4 operator * (const Float4& a, const Float4& b);
	friend Float4 operator * (float f, const Float4& a);
	friend Float4 operator / (const Float4& a, const Float4& b);
	friend Float4 operator / (float f, const Float4& a);

	Float4 clamp(float lo, float hi) const;
};

struct Double4 {
	double d0, d1, d2, d3;

	Double4(double d0, double d1, double d2, double d3) : d0 { d0 }, d1 { d1 }, d2 { d2 }, d3 { d3 } {}
	Double4(double d) : Double4(d, d, d, d) {}
	Double4() : Double4(0, 0, 0, 0) {}

	double operator [] (size_t idx) const;
	friend Double4 operator + (const Double4& a, const Double4& b);
	friend Double4 operator - (const Double4& a, const Double4& b);
	friend Double4 operator * (const Double4& a, const Double4& b);
	friend Double4 operator * (float f, const Double4& a);
	friend Double4 operator / (const Double4& a, const Double4& b);
	friend Double4 operator / (float f, const Double4& a);

	Double4 clamp(double lo, double hi) const;
};

namespace std {

	Float4 fma(const Float4& x, const Float4& y, const Float4& z);

	Double4 fma(const Double4& x, const Double4& y, const Double4& z);
}
