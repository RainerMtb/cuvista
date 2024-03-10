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

#include "AffineTransform.hpp"
#include "MainData.hpp"

class ThreadPool;

class FrameResult {

public:
	FrameResult(MainData& data, ThreadPool& threadPool) :
		mData { data },
		mFiniteResults(data.resultCount),
		mAffineSolver { std::make_unique<AffineSolverFast>(threadPool, data.resultCount) } {}

	//compute resulting transformation for this frame
	void computeTransform(std::vector<PointResult>& results, ThreadPool& threadPool, int64_t frameIndex, RNGbase* rng);

	//get the last computed treansform
	const AffineTransform& getTransform() const;

	//reset internal state of this class
	void reset();

private:
	const MainData& mData;
	std::unique_ptr<AffineSolver> mAffineSolver;
	AffineTransform mBestTransform;

	ptrdiff_t mFiniteCount = 0;
	std::vector<PointResult> mFiniteResults;
	std::vector<PointResult>::iterator mFiniteEnd;
	std::vector<PointContext> mConsList;

	using IT = std::vector<PointContext>::iterator;
	void computePointContext(IT begin, IT end, const AffineTransform& trf, double radius);

	void computeExperimental(std::vector<PointResult>& results, ThreadPool& threadPool, int64_t frameIndex, RNGbase* rng);
	void compute(std::vector<PointResult>& results, ThreadPool& threadPool, int64_t frameIndex, RNGbase* rng);
};
