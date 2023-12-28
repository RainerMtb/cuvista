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
#include "CoreData.hpp"
#include "ThreadPool.hpp"

class AffineTransform : public Affine2D {

private:
	Mat A = Mat<double>::zeros(6, 6);
	Mat b = Mat<double>::zeros(6, 1);

protected:
	AffineTransform(int64_t frameIndex, double m00, double m01, double m02, double m10, double m11, double m12) :
		Affine2D(m00, m01, m02, m10, m11, m12),
		frameIndex { frameIndex } {}

	AffineTransform(double m00, double m01, double m02, double m10, double m11, double m12) :
		AffineTransform(-1, m00, m01, m02, m10, m11, m12) {}

public:
	int64_t frameIndex;

	AffineTransform(int64_t frameIndex, double scale, double rot, double dx, double dy) :
		AffineTransform(frameIndex, scale, rot, dx, -rot, scale, dy) {}

	AffineTransform() :
		AffineTransform(-1, 1, 0, 0, 0) {}

	//compute parameters from gives points
	void computeAffine(std::vector<PointResult>& points);

	//compute affine transform from given points
	bool computeAffine(std::vector<PointResult>::iterator begin, size_t count);

	//convert to cuda struct
	std::array<double, 6> toArray() const;
};


class AffineSolver : public AffineTransform {

public:
	//compute similar transform, no shear
	virtual void computeSimilar(std::vector<PointResult>::iterator it, size_t count) = 0;
};

class AffineSolverSimple : public AffineSolver {

public:
	void computeSimilar(std::vector<PointResult>::iterator it, size_t count) override;
};

class AffineSolverFast : public AffineSolver {

private:
	ThreadPool& threadPool;

public:
	AffineSolverFast(ThreadPool& threadPool) :
		threadPool { threadPool } {}

	void computeSimilar(std::vector<PointResult>::iterator it, size_t count) override;
};
