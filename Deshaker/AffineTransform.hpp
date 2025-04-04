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
#include "Affine2D.hpp"
#include "CoreData.hpp"
#include "ThreadPoolBase.h"


class AffineTransform : public Affine2D {

protected:
	Mat A = Mat<double>::zeros(6, 6);
	Mat b = Mat<double>::zeros(6, 1);

	AffineTransform(int64_t frameIndex, double m00, double m01, double m02, double m10, double m11, double m12) :
		Affine2D(m00, m01, m02, m10, m11, m12),
		frameIndex { frameIndex } {}

	AffineTransform(double m00, double m01, double m02, double m10, double m11, double m12) :
		AffineTransform(-1, m00, m01, m02, m10, m11, m12) {}

public:
	int64_t frameIndex;

	AffineTransform() :
		AffineTransform(0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0) {}

	//compute parameters from gives points
	const AffineTransform& computeAffineDirect(const PointResult& p1, const PointResult& p2, const PointResult& p3);

	//compute similar transform directly from two points
	const AffineTransform& computeSimilarDirect(const PointResult& p1, const PointResult& p2);

	//compute affine transform from given points
	bool computeAffine(std::vector<PointResult>& points);

	friend std::ostream& operator << (std::ostream& os, const AffineTransform& trf);
};


class AffineSolver : public AffineTransform {

protected:
	//compute similar transform, no shear
	virtual const AffineTransform& computeSimilar(std::span<PointBase> points) = 0;

public:
	virtual const AffineTransform& computeSimilar(std::span<PointContext> points);
	virtual const AffineTransform& computeSimilar(std::span<PointResult> points);
	virtual ~AffineSolver() {}
};

class AffineSolverSimple : public AffineSolver {

private:
	Matd Adata, bdata;

	const AffineTransform& computeSimilar(std::span<PointBase> points) override;

public:
	AffineSolverSimple(size_t maxPoints) :
		Adata { Matd::allocate(maxPoints * 2, 4) },
		bdata { Matd::allocate(maxPoints * 2, 1) } {}
};

class AffineSolverFast : public AffineSolver {

private:
	ThreadPoolBase& threadPool;
	Matd Adata;

	const AffineTransform& computeSimilar(std::span<PointBase> points) override;

public:
	AffineSolverFast(ThreadPoolBase& threadPool, size_t maxPoints) :
		threadPool { threadPool },
		Adata { Matd::allocate(6, maxPoints * 2) } {}
};

class AffineSolverAvx : public AffineSolver {

private:
	Matd M;

	const AffineTransform& computeSimilar(std::span<PointBase> points) override;

public:
	AffineSolverAvx(size_t maxPoints) :
		M { Matd::allocate(6, maxPoints * 2 + 8) } {}
};
