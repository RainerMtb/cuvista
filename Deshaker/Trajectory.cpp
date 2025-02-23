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

TrajectoryMat::TrajectoryMat(double u, double v, double a) : 
	Matd(1, 3) 
{
	array[0] = u; array[1] = v; array[2] = a;
}

TrajectoryMat::TrajectoryMat() : TrajectoryMat(0, 0, 0) {}

double TrajectoryMat::u() { return at(0, 0); }
double TrajectoryMat::v() { return at(0, 1); }
double TrajectoryMat::a() { return at(0, 2); }

//create new item and append to list, transformation u, v, a
TrajectoryItem::TrajectoryItem(double u, double v, double a, int64_t frameIndex) :
	values { u, v, a },
	frameIndex { frameIndex }
{
	currentSum += values;	   //sum to this point
	sum.setData(currentSum);   //store sum for this point
	isDuplicateFrame = frameIndex > 0 && std::abs(u) < 0.5 && std::abs(v) < 0.5 && std::abs(a) < 0.001;
}

const TrajectoryItem& Trajectory::addTrajectoryTransform(double dx, double dy, double da, int64_t frameIndex) {
	return trajectory.emplace_back(dx, dy, da, frameIndex);
}

//append new result to list
const TrajectoryItem& Trajectory::addTrajectoryTransform(const AffineTransform& transform) {
	return addTrajectoryTransform(transform.dX(), transform.dY(), transform.rot(), transform.frameIndex);
}

int64_t Trajectory::getTrajectorySize() {
	return trajectory.size();
}

int64_t Trajectory::clamp(int64_t val, int64_t lo, int64_t hi) {
	return std::clamp(val, lo, hi);
}

const AffineTransform& Trajectory::computeSmoothTransform(const MainData& data, int64_t frameWriteIndex) {
	//compute average movements over current window
	double sig = data.radius * data.cSigmaParam;
	double sumWeight = 0.0;
	tempAvg.setValues(0.0);
	for (int64_t i = -data.radius; i <= data.radius; i++) {
		double w = std::exp(-0.5 * i * i / (sig * sig));
		int64_t idx = clamp(frameWriteIndex + i, 0, trajectory.size() - 1);
		tempSum.setValues(trajectory[idx].sum);
		tempSum *= w;
		tempAvg += tempSum;
		sumWeight += w;
	}
	tempAvg /= sumWeight;
	TrajectoryItem& ti = trajectory[frameWriteIndex];

	if (ti.isDuplicateFrame) {
		//take midpoint between transformation of previous frame and calculated transform
		delta += ti.sum;
		delta -= tempAvg;
		delta /= 2.0;

	} else {
		//normal procedure
		delta.setValues(ti.sum);
		delta -= tempAvg;
	}

	//calculate zoom
	double fitZoom = calcRequiredZoom(delta.u(), delta.v(), delta.a(), data.w, data.h, data, frameWriteIndex);
	double smoothFitZoom = std::max(currentZoom * data.zoomFallback, fitZoom);
	trajectory[frameWriteIndex].zoom = currentZoom = smoothFitZoom;
	double zoom = std::clamp(smoothFitZoom, data.zoomMin, data.zoomMax);
	//std::printf("%04d zoom required %5.2f smoothed %5.2f final %5.2f\n", frameWriteIndex, fitZoom, smoothFitZoom, zoom);

	//get transform to apply to image
	currentTransform.reset()
		.addTranslation(data.w / 2.0, data.h / 2.0)		//translate to origin
		.addRotation(delta.a())					    	//rotation
		.addZoom(zoom)							        //zoom as set
		.addTranslation(delta.u(), delta.v())	        //computed translation to stabilize
		.addTranslation(data.w / -2.0, data.h / -2.0)	//translate back to center
		;
	currentTransform.frameIndex = frameWriteIndex;
	return currentTransform;
}

void Trajectory::readTransforms(std::map<int64_t, TransformValues> transformsMap) {
	TransformValues trf = { 1.0, 0.0, 0.0, 0.0 };
	int64_t maxFrame = transformsMap.rbegin()->first;

	for (int i = 0; i <= maxFrame; i++) {
		auto item = transformsMap.find(i);
		if (item != transformsMap.end()) trf = item->second;
		addTrajectoryTransform(trf.dx, trf.dy, trf.da, i);
	}
}

double Trajectory::calcRequiredZoom(double dx, double dy, double rot, double w, double h, const MainData& data, int64_t frameWriteIndex) {
	double w2 = w / 2.0;
	double h2 = h / 2.0;
	Affine2D trf = Affine2D::fromValues(1.0, -rot, dx, dy);
	std::vector<Point> corners = { {w2, h2}, {-w2, h2}, {-w2, -h2}, {w2, -h2}, {w2, h2} };
	std::vector<Point> transformedCorners(5);
	for (int i = 0; i < 5; i++) transformedCorners[i] = trf.transform(corners[i]);
	Point m = trf.transform(0.0, 0.0);

	double zoom = 1.0;
	for (size_t i = 0; i < 4; i++) {
		Point& c = corners[i];
		for (size_t k = 0; k < 4; k++) {
			Point& a = transformedCorners[k];
			Point& b = transformedCorners[k + 1];
			double z = ((c.x - m.x) * (b.y - a.y) - (c.y - m.y) * (b.x - a.x)) / ((m.y - a.y) * (b.x - a.x) + (a.x - m.x) * (b.y - a.y));
			zoom = std::max(zoom, z);
		}
	}
	//std::cout << frameWriteIndex << " " << dx << " " << dy << " " << (rot * 180 / std::numbers::pi) << " " << zoom << std::endl;
	return zoom;
}