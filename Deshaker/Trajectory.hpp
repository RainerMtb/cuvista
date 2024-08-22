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

class TrajectoryMat : public Matd {
public:
	TrajectoryMat();
	TrajectoryMat(double u, double v, double a);

	double u();
	double v();
	double a();
};

struct TrajectoryItem {
	//accumulated values up to most recent frame
	inline static TrajectoryMat currentSum;
	//values calculated from previous frame to this frame
	TrajectoryMat values;
	//accumulated values up to this frame
	TrajectoryMat sum;
	//frame is identical to previous frame
	bool isDuplicateFrame;
	//frame index for debugging
	int64_t frameIndex;
	//zoom value to fill frame
	double zoom;

	TrajectoryItem(double u, double v, double a, int64_t frameIndex);
};

class Trajectory {

private:
	//holds all trajectory items, grows with each frame
	std::vector<TrajectoryItem> trajectory;
	//temporary mats
	TrajectoryMat delta;
	TrajectoryMat tempAvg;
	TrajectoryMat tempSum;
	//current output
	AffineTransform currentTransform;
	//current zoom value
	double currentZoom = 1.0;

	int64_t clamp(int64_t val, int64_t lo, int64_t hi);

public:
	double calcRequiredZoom(double dx, double dy, double rot, double w, double h, const MainData& data, int64_t frameWriteIndex);

	const TrajectoryItem& addTrajectoryTransform(double dx, double dy, double da, int64_t frameIndex);
	const TrajectoryItem& addTrajectoryTransform(const AffineTransform& transform);

	//create affine transformation
	const AffineTransform& computeSmoothTransform(const MainData& data, int64_t frameWriteIndex);

	//read transforms
	void readTransforms(std::map<int64_t, TransformValues> transformsMap);

	//trajectory count
	int64_t getTrajectorySize();
};
