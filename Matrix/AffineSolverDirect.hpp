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
#include "ThreadPoolBase.hpp"


class AffineSolverBase {

public:
	virtual void computeSimilar(std::span<PointBase> points, Affine2D& dest) = 0;
};


class AffineSolverDirect : public AffineSolverBase {

protected:
	Matd M;
	size_t m = 0;
	double dx = 0, dy = 0;
	double b = 0, s0 = 0, s1 = 0, e = 0, f = 0, g = 0, h = 0, j = 0, k = 0, n = 0;
	double rd[4] = {};

	ThreadPoolBase& threadPool;

	void initParam(std::span<PointBase> points);

public:
	AffineSolverDirect(ThreadPoolBase& threadPool, size_t maxPoints) :
		threadPool { threadPool },
		M { Matd::allocate(6, maxPoints * 2 + 16) }
	{}
};
