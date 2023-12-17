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

#include "FrameResult.hpp"

struct TransformValues {
	double s, dx, dy, da;
};

struct TrajectoryItem {
	//accumulated values up to most recent frame
	inline static Mat currentSum = Mat<double>::zeros(1, 3);
	//values calculated from previous frame to this frame
	Mat<double> values = Mat<double>::zeros(1, 3);
	//accumulated values up to this frame
	Mat<double> sum = Mat<double>::zeros(1, 3);
	//frame is identical to previous frame
	bool isDuplicateFrame;
	//frame index for debugging
	int64_t frameIndex;

	TrajectoryItem(double u, double v, double a, int64_t frameIndex);
};

class Trajectory {

private:
	//holds all trajectory items, grows with each frame
	std::vector<TrajectoryItem> trajectory;
	//temporary mats
	Matd tempDelta = Matd::zeros(1, 3);
	Matd tempAvg = Matd::zeros(1, 3);
	Matd tempSum = Matd::zeros(1, 3);
	//current output
	AffineTransform out;

public:
	const TrajectoryItem& addTrajectoryTransform(double dx, double dy, double da, int64_t frameIndex);
	const TrajectoryItem& addTrajectoryTransform(const AffineTransform& transform);

	//create affine transformation
	const AffineTransform& computeSmoothTransform(const MainData& data, int64_t frameWriteIndex);

	//read transforms
	void readTransforms(std::map<int64_t, TransformValues> transformsMap);

	//trajectory count
	int64_t getTrajectorySize();
};
