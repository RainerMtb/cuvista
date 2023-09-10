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

#include "Affine2D.h"
#include "CoreData.cuh"
#include "Instrumentor.h"
#include "ThreadPool.h"

class AffineTransform : public Affine2D {

private:
	Mat A = Mat<double>::zeros(6, 6);
	Mat b = Mat<double>::zeros(6, 1);

protected:
	AffineTransform(double m00, double m01, double m02, double m10, double m11, double m12) : Affine2D(m00, m01, m02, m10, m11, m12) {}

public:
	AffineTransform(double scale, double rot, double dx, double dy) : AffineTransform(scale, rot, dx, -rot, scale, dy) {}

	AffineTransform() : AffineTransform(1, 0, 0, 0) {}

	//compute parameters from gives points
	void computeAffine(std::vector<PointResult>& points);

	//compute affine transform from given points
	AffineTransform static computeAffine(std::vector<PointResult>::iterator begin, size_t count);

	//compute similar transform, no shear
	void computeSimilarLoop(std::vector<PointResult>::iterator it, size_t count);

	//compute similar transform, no shear
	void computeSimilarDirect(std::vector<PointResult>::iterator it, size_t count, ThreadPool& threadPool);

	//convert to cuda struct
	std::array<double, 6> toArray() const;

private:
	static void savePointResults(std::vector<PointResult>::iterator begin, size_t count);
};