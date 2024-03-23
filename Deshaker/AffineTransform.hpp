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
#include "Affine2D.h"
#include "CoreData.hpp"
#include "ThreadPool.hpp"

struct PointContext {
	PointResult* ptr;
	double delta = 0.0;
	double deltaAngleCos = 0.0;
	double deltaAngle = 0.0;
	double distanceEllipse = 0.0;
	double tx = 0.0;
	double ty = 0.0;
	double confidence = 0.0;

	double distance = 0.0;
	double distanceRelative = 0.0;

	PointContext(PointResult& pr) : ptr { &pr } {}
	PointContext() : ptr { nullptr } {}
};

struct PointBase {
	int x, y;
	double u, v;

	PointBase(const PointResult& pr) : x { pr.x }, y { pr.y }, u { pr.u }, v { pr.v } {}
	PointBase(const PointContext& pc) : x { pc.ptr->x }, y { pc.ptr->y }, u { pc.ptr->u }, v { pc.ptr->v } {}
};

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

	AffineTransform(int64_t frameIndex, double scale, double rot, double dx, double dy) :
		AffineTransform(frameIndex, scale, rot, dx, -rot, scale, dy) {}

	AffineTransform() :
		AffineTransform(-1, 1, 0, 0, 0) {}

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
	ThreadPool& threadPool;
	Matd Adata;

	const AffineTransform& computeSimilar(std::span<PointBase> points) override;

public:
	AffineSolverFast(ThreadPool& threadPool, size_t maxPoints) :
		threadPool { threadPool },
		Adata { Matd::allocate(6, maxPoints * 2) } {}
};
