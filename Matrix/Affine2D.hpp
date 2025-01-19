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
#include <utility>
#include <array>

struct AffineCore;

struct Point {
	double x, y;
};

class Affine2D : protected Mat<double> {

protected:
	void setParam(double m00, double m01, double m02, double m10, double m11, double m12);

	Affine2D(double m00, double m01, double m02, double m10, double m11, double m12);
	Affine2D(Matd mat);

public:
	Affine2D();

	//set rigid transform parameters directly
	static Affine2D fromParam(double scale, double rot, double dx, double dy);

	//add values to identity transform
	static Affine2D fromValues(double scale, double rot, double dx, double dy);

	//identity transform
	static Affine2D identity();

	//reset to null transform = identity
	Affine2D& reset();

	bool operator == (const Affine2D& other) const;

	bool equals(const Affine2D& other, double tolerance = 0.0) const;

	Affine2D& setParam(double scale, double rot, double dx, double dy);

	Affine2D& addTranslation(double dx, double dy);

	Affine2D& addRotation(double angleRad);

	Affine2D& addRotationDegrees(double angleRad);

	Affine2D& addZoom(double zoom);

	//transform point x0, y0
	Point transform(size_t x0, size_t y0) const;

	//transform point x0, y0
	Point transform(int x0, int y0) const;

	//transform point x0, y0
	Point transform(const Point& p) const;

	//transform point x0, y0
	Point transform(double x0, double y0) const;

	double scale() const;

	double rot() const;

	double rotMinutes() const;

	double dX() const;

	double dY() const;

	double arrayValue(int idx) const;

	std::array<double, 6> toArray() const;

	AffineCore toAffineCore() const;

	std::string toString(const std::string& title = "", int digits = -1) const override;

	std::wstring toWString(const std::string& title = "", int digits = -1) const override;

	Affine2D& toConsole(const std::string& title = "", int digits = -1) override;
};
