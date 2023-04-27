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

//residual transformation of frames

#include "Trajectory.hpp"

//create new item and append to list, transformation u, v, a
TrajectoryItem::TrajectoryItem(double u, double v, double a) {
	values = Mat<double>::fromRow({u, v, a});
	currentSum += values;	//sum to this point
	sum = currentSum;		//store sum for this point
	isDuplicateFrame = std::abs(u) < 0.5 && std::abs(v) < 0.5 && std::abs(a) < 0.001;
}

TrajectoryItem Trajectory::addTrajectoryTransform(double dx, double dy, double da) {
	return trajectory.emplace_back(dx, dy, da);
}

//append new result to list
TrajectoryItem Trajectory::addTrajectoryTransform(const Affine2D& transform, int64_t frameIdx) {
	return addTrajectoryTransform(transform.dX(), transform.dY(), transform.rot());
}

int64_t Trajectory::getTrajectorySize() {
	return trajectory.size();
}

int64_t clamp(int64_t val, int64_t lo, int64_t hi) {
	if (val < lo) return lo;
	if (val > hi) return hi;
	return val;
}

AffineTransform Trajectory::computeTransformForFrame(const MainData& data, int64_t frameWriteIndex) {
	//compute average movements over current window
	double sig = data.radius * data.cSigmaParam;
	double sumW = 0.0;
	Mat avg = Mat<double>::zeros(1, 3);
	for (int64_t i = -data.radius; i <= data.radius; i++) {
		double w = std::exp(-0.5 * i * i / (sig * sig));
		int64_t idx = clamp(frameWriteIndex + i, 0, trajectory.size() - 1);
		avg += trajectory[idx].sum * w;
		sumW += w;
	}
	avg /= sumW;
	TrajectoryItem& ti = trajectory[frameWriteIndex];

	if (ti.isDuplicateFrame) {
		//start with usual image transformation
		Mat deltaMax = ti.sum - avg;
		//then take midpoint between transformation of previous frame and calculated transform
		delta = (delta + deltaMax) / 2;

	} else {
		//normal procedure
		delta = ti.sum - avg;
	}

	//get transform to apply to image
	AffineTransform out; //default null transform
	out.addTranslation(data.w / 2.0, data.h / 2.0)			//translate to origin
		.addRotation(delta[0][2])							//rotation
		.addTranslation(delta[0][0], delta[0][1])			//computed translation to stabilize
		.addZoom(data.imZoom)								//zoom as set
		.addTranslation(data.w / -2.0, data.h / -2.0)		//translate back to center
		;
	return out;
}

void Trajectory::readTransforms(std::map<int64_t, TransformValues> transformsMap) {
	TransformValues trf = { 1.0, 0.0, 0.0, 0.0 };
	int64_t maxFrame = transformsMap.rbegin()->first;

	for (int i = 0; i <= maxFrame; i++) {
		auto item = transformsMap.find(i);
		if (item != transformsMap.end()) trf = item->second;
		addTrajectoryTransform(trf.dx, trf.dy, trf.da);
	}
}
