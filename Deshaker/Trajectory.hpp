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

class TrajectoryValues {

public:
	double u, v, a;

	TrajectoryValues();
	TrajectoryValues(double x);
	TrajectoryValues(double u, double v, double a);

	TrajectoryValues& set(double u, double v, double a);
	TrajectoryValues& set(const TrajectoryValues& other);

	bool operator == (const TrajectoryValues& other) const;

	TrajectoryValues operator + (const TrajectoryValues& other);
	TrajectoryValues& operator += (const TrajectoryValues& other);

	TrajectoryValues operator - (const TrajectoryValues& other);
	TrajectoryValues& operator -= (const TrajectoryValues& other);

	TrajectoryValues operator * (const TrajectoryValues& other);
	TrajectoryValues& operator *= (const TrajectoryValues& other);

	TrajectoryValues operator / (const TrajectoryValues& other);
	TrajectoryValues& operator /= (const TrajectoryValues& other);
};

class TrajectoryItem {

public:
	//values calculated from previous frame to this frame
	TrajectoryValues values;
	//accumulated values up to this frame
	TrajectoryValues sum;
	//smoothed values
	TrajectoryValues smoothed;
	//frame is identical to previous frame
	bool isDuplicateFrame;
	//frame index for debugging
	int64_t frameIndex;
	//zoom value to fill stabilized frame
	double zoomRequired;
	//zoom value smoothed
	double zoom;
	//check if has been computed
	bool isComputed;

	bool operator == (const TrajectoryItem& other) const;
};

class Trajectory {

private:
	//holds all trajectory items, grows with each frame
	std::vector<TrajectoryItem> mTrajectory;
	//temporary mats
	TrajectoryValues tempAvg;
	TrajectoryValues tempSum;
	TrajectoryValues delta;

	//current output transform
	AffineTransform mCurrentTransform;
	//accumulated values up to most recent frame
	TrajectoryValues mCurrentSum;
	//accumulated zoom value for the previous frame
	double mCurrentZoom = 1.0;

	int64_t clamp(int64_t val, int64_t lo, int64_t hi);

public:
	//compute zoom value to fill the frame
	double calcRequiredZoom(double dx, double dy, double rot, double w, double h);

	//add transform to the trajectory list
	void addTrajectoryTransform(double dx, double dy, double da, int64_t frameIndex);
	//add transform to the trajectory list
	void addTrajectoryTransform(const AffineTransform& transform);

	//replace transform in the trajectory list
	void setTrajectoryTransform(const AffineTransform& transform);

	//compute smoothed transform for frame
	void computeSmoothTransform(const MainData& data, int64_t frameIndex);

	//compute smoothed zoom for frame
	void computeSmoothZoom(const MainData& data, int64_t frameIndex);

	//get computed transform
	const AffineTransform& getTransform(const MainData& data, int64_t frameIndex);

	//load transforms from list into trajectory
	void readTransforms(std::map<int64_t, TransformValues> transformsMap);

	//trajectory count
	int64_t size();

	//reset for iteration
	void reset();

	//reserve trajectory size
	void reserve(int64_t siz);

	//get a copy of the complete trajectory
	std::vector<TrajectoryItem> getTrajectory();

	//check if this item is already computed
	bool isComputed(int64_t frameindex);

	//output trajectory items
	friend std::ostream& operator << (std::ostream& os, const Trajectory& trajectory);
};
