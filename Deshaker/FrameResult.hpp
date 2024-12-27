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
#include "AffineTransform.hpp"
#include "MainData.hpp"
#include "ThreadPoolBase.h"

class FrameResult {

	using SamplerPtr = std::shared_ptr<SamplerBase<PointResult>>;

public:
	FrameResult(MainData& data, ThreadPoolBase& threadPool);

	//compute resulting transformation for this frame
	void computeTransform(std::vector<PointResult>& results, ThreadPoolBase& threadPool, int64_t frameIndex, SamplerPtr rng);

	//get the last computed treansform
	const AffineTransform& getTransform() const;

	//reset internal state of this class
	void reset();

private:
	const MainData& mData;
	std::unique_ptr<AffineSolver> mAffineSolver;
	AffineTransform mBestTransform;
	std::vector<PointContext> mConsList;

	double sqr(double value);
};
