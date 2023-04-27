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

#include "Mat.h"
#include <utility>

class Affine2D : public Mat<double> {

protected:
	virtual void setParam(double m00, double m01, double m02, double m10, double m11, double m12) {
		array[0] = m00;   array[1] = m01;   array[2] = m02;
		array[3] = m10;   array[4] = m11;   array[5] = m12;
		array[6] = 0;     array[7] = 0;     array[8] = 1;
	}

	Affine2D(double m00, double m01, double m02, double m10, double m11, double m12) : Mat<double>(3, 3) {
		setParam(m00, m01, m02, m10, m11, m12);
	}

	Affine2D(double scale, double rot, double dx, double dy) : Affine2D(scale, rot, dx, -rot, scale, dy) {}

	Affine2D(Mat<double> mat) : Mat<double>(mat.array, mat.rows(), mat.cols(), true) {}

public:
	Affine2D() : Affine2D(1, 0, 0, 0) {}

	//rigid transform parameters
	static Affine2D fromValues(double scale, double rot, double dx, double dy) { 
		return Affine2D(scale, rot, dx, dy); 
	}

	//identity transform
	static Affine2D identity() { 
		return Affine2D(); 
	}

	//reset to null transform = identity
	Affine2D& reset() {
		setValuesByRow({ 1, 0, 0, 0, 1, 0, 0, 0, 1 });
		return *this;
	}

	virtual void setParam(double scale, double rot, double dx, double dy) {
		setParam(scale, rot, dx, -rot, scale, dy);
	}

	Affine2D& addTranslation(double dx, double dy) {
		setValues(times(Mat<double>::fromRows(3, 3, { 1, 0, dx, 0, 1, dy, 0, 0, 1 })));
		return *this;
	}

	Affine2D& addRotation(double angleRad) {
		setValues(times(Mat<double>::fromRows(3, 3, { std::cos(angleRad), std::sin(angleRad), 0, -std::sin(angleRad), std::cos(angleRad), 0, 0, 0, 1 })));
		return *this;
	}

	Affine2D& addZoom(double zoom) {
		setValues(times(Mat<double>::fromRows(3, 3, { 1 / zoom, 0, 0, 0, 1 / zoom, 0, 0, 0, 1 })));
		return *this;
	}

	Affine2D invert() {
		auto out = inv();
		return out.has_value() ? Affine2D(out.value()) : Affine2D();
	}

	//transform point x0, y0
	std::pair<double, double> transform(size_t x0, size_t y0) const {
		return transform((double) x0, (double) y0);
	}

	//transform point x0, y0
	std::pair<double, double> transform(int x0, int y0) const {
		return transform((double) x0, (double) y0);
	}

	//transform point x0, y0
	std::pair<double, double> transform(double x0, double y0) const {
		return std::make_pair(x0 * at(0, 0) + y0 * at(0, 1) + at(0, 2), x0 * at(1, 0) + y0 * at(1, 1) + at(1, 2));
	}

	double scale() const {
		assert(std::abs(at(0, 0) - at(1, 1)) < Mat::EQUAL_TOL && "scale is not affine");
		return at(0, 0); 
	}

	double rot() const { 
		assert(std::abs(at(0, 1) + at(1, 0)) < Mat::EQUAL_TOL && "rotation is not affine");
		return at(0, 1); 
	}

	double rotMilliDegrees() const {
		return rot() * 180.0 / std::numbers::pi * 3600.0;
	}

	double dX() const { return at(0, 2); }

	double dY() const { return at(1, 2); }

	virtual Affine2D& toConsole(const std::string& title = "Affine", int digits = -1) override {
		std::cout << std::setprecision(10) << title << " [["
			<< array[0] << ", " << array[1] << ", " << array[2] << " // " 
			<< array[3] << ", " << array[4] << ", " << array[5] << "]] " << std::endl;
		return *this;
	}
};
