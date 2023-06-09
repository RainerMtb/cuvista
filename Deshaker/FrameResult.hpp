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

#include <algorithm>
#include "AffineTransform.hpp"
#include "ThreadPool.h"
#include "MainData.hpp"

class FrameResult {

public:
	std::vector<PointResult> mFiniteResults;
	ptrdiff_t mCountFinite = 0;
	ptrdiff_t mCountConsens = 0;

	std::vector<AffineTransform> mTransformsList;
	AffineTransform mTransform;

	FrameResult(const MainData& data) : mFiniteResults(data.resultCount) {}

	//compute resulting transformation for this frame
	const AffineTransform& computeTransform(const std::vector<PointResult>& results, const MainData& data, ThreadPool& threadPool, RNGbase* rng);
};
