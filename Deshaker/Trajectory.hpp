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
	Mat<double> values;
	//accumulated values up to this frame
	Mat<double> sum;
	//frame is identical to previous frame
	bool isDuplicateFrame;

	TrajectoryItem(double u, double v, double a);
};

class Trajectory {

private:
	//holds all trajectory items, grows with each frame
	std::vector<TrajectoryItem> trajectory;
	//final averaged values for this frame, but still without additional zoom
	Mat<double> delta = Mat<double>::zeros(1, 3);

public:
	TrajectoryItem addTrajectoryTransform(double dx, double dy, double da);
	TrajectoryItem addTrajectoryTransform(const Affine2D& transform, int64_t frameIdx);

	//create affine transformation
	AffineTransform computeTransformForFrame(const MainData& data, int64_t frameWriteIndex);

	//read transforms
	void readTransforms(std::map<int64_t, TransformValues> transformsMap);

	//trajectory count
	int64_t getTrajectorySize();
};
