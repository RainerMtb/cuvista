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

#include "Trajectory.hpp"

//----------- TrajectoryValues
//----------- Sort of Vector holding trajectory values

TrajectoryValues::TrajectoryValues(double u, double v, double a) :
	u { u },
	v { v },
	a { a } {}

TrajectoryValues::TrajectoryValues(double x) :
	TrajectoryValues(x, x, x) {}

TrajectoryValues::TrajectoryValues() :
	TrajectoryValues(0.0, 0.0, 0.0) {}

TrajectoryValues& TrajectoryValues::set(double u, double v, double a) {
	this->u = u;
	this->v = v;
	this->a = a;
	return *this;
}

TrajectoryValues& TrajectoryValues::set(const TrajectoryValues& other) {
	return set(other.u, other.v, other.a);
}

bool TrajectoryValues::operator == (const TrajectoryValues& other) const {
	return u == other.u && v == other.v && a == other.a;
}

TrajectoryValues TrajectoryValues::operator + (const TrajectoryValues& other) {
	return TrajectoryValues(u + other.u, v + other.v, a + other.a);
}

TrajectoryValues& TrajectoryValues::operator += (const TrajectoryValues& other) {
	return set(u + other.u, v + other.v, a + other.a);
}

TrajectoryValues TrajectoryValues::operator - (const TrajectoryValues& other) {
	return TrajectoryValues(u - other.u, v - other.v, a - other.a);
}

TrajectoryValues& TrajectoryValues::operator -= (const TrajectoryValues& other) {
	return set(u - other.u, v - other.v, a - other.a);
}

TrajectoryValues TrajectoryValues::operator * (const TrajectoryValues& other) {
	return TrajectoryValues(u * other.u, v * other.v, a * other.a);
}

TrajectoryValues& TrajectoryValues::operator *= (const TrajectoryValues& other) {
	return set(u * other.u, v * other.v, a * other.a);
}

TrajectoryValues TrajectoryValues::operator / (const TrajectoryValues& other) {
	return TrajectoryValues(u / other.u, v / other.v, a / other.a);
}

TrajectoryValues& TrajectoryValues::operator /= (const TrajectoryValues& other) {
	return set(u / other.u, v / other.v, a / other.a);
}


//----------- Trajectory Item
//----------- Current state of the trajectory

bool TrajectoryItem::operator == (const TrajectoryItem& other) const {
	return values == other.values && smoothed == other.smoothed && sum == other.sum
		&& isDuplicateFrame == other.isDuplicateFrame && frameIndex == other.frameIndex 
		&& zoom == other.zoom && zoomRequired == other.zoomRequired;
}

//----------- Trajectory
//----------- List of Items

//append new result to trajectory
void Trajectory::addTrajectoryTransform(double u, double v, double a, int64_t frameIndex) {
	TrajectoryItem item;
	item.values.u = u;
	item.values.v = v;
	item.values.a = a;
	item.frameIndex = frameIndex;
	item.zoomRequired = 1.0;
	item.zoom = 1.0;
	mCurrentSum += item.values;
	item.sum = mCurrentSum;
	item.isDuplicateFrame = frameIndex > 0 && std::abs(u) < 0.5 && std::abs(v) < 0.5 && std::abs(a) < 0.001;
	item.isComputed = false;
	mTrajectory.push_back(item);
}

//append new result to trajectory
void Trajectory::addTrajectoryTransform(const AffineTransform& transform) {
	addTrajectoryTransform(transform.dX(), transform.dY(), transform.rot(), transform.frameIndex);
}

//set trajectory item with updated values
void Trajectory::setTrajectoryTransform(const AffineTransform& transform) {
	TrajectoryItem& item = mTrajectory[transform.frameIndex];
	TrajectoryValues& val = item.values;
	val.u = transform.dX();
	val.v = transform.dY();
	val.a = transform.rot();
	mCurrentSum += item.values;
	item.sum = mCurrentSum;
	item.isDuplicateFrame = transform.frameIndex > 0 && std::abs(val.u) < 0.5 && std::abs(val.v) < 0.5 && std::abs(val.a) < 0.001;
	item.isComputed = false;
}

int64_t Trajectory::size() {
	return mTrajectory.size();
}

void Trajectory::reset() {
	mCurrentSum.u = 0.0;
	mCurrentSum.v = 0.0;
	mCurrentSum.a = 0.0;
}

void Trajectory::reserve(int64_t siz) {
	mTrajectory.reserve(siz);
}

std::vector<TrajectoryItem> Trajectory::getTrajectory() {
	return mTrajectory;
}

bool Trajectory::isComputed(int64_t frameIndex) {
	return mTrajectory[frameIndex].isComputed;
}

int64_t Trajectory::clamp(int64_t val, int64_t lo, int64_t hi) {
	return std::clamp(val, lo, hi);
}

//compute average movements over current window but do not set zoom
void Trajectory::computeSmoothTransform(const MainData& data, int64_t frameIndex) {
	double sig = data.radius * data.cSigmaParam;
	double sumWeight = 0.0;
	tempAvg.set(0.0, 0.0, 0.0);
	for (int64_t i = -data.radius; i <= data.radius; i++) {
		double w = std::exp(-0.5 * i * i / (sig * sig));
		int64_t idx = clamp(frameIndex + i, 0, mTrajectory.size() - 1);
		tempSum.set(mTrajectory[idx].sum);
		tempSum *= w;
		tempAvg += tempSum;
		sumWeight += w;
	}
	tempAvg /= sumWeight;
	TrajectoryItem& ti = mTrajectory[frameIndex];
	assert(ti.isComputed == false && "trajectory item has already been computed");

	if (ti.isDuplicateFrame) {
		//take midpoint between transformation of previous frame and calculated transform
		delta += ti.sum;
		delta -= tempAvg;
		delta /= 2.0;

	} else {
		//normal procedure
		delta.set(ti.sum);
		delta -= tempAvg;
	}
	//on repeated passes just add delta to old smoothing
	ti.smoothed += delta;

	//calculate zoom to fill frame
	ti.zoomRequired = calcRequiredZoom(ti.smoothed.u, ti.smoothed.v, ti.smoothed.a, data.w, data.h);

	ti.isComputed = true;
}

//smooth dynamic zooming, but can be limited to effective static zoom
void Trajectory::computeSmoothZoom(const MainData& data, int64_t frameIndex) {
	TrajectoryItem& ti = mTrajectory[frameIndex];
	double gradient = data.zoomFallback - 1.0;

	for (int i = 0; i <= data.radius && frameIndex + i < mTrajectory.size(); i++) {
		int64_t idx = frameIndex + i;
		gradient = std::max(gradient, (mTrajectory[idx].zoomRequired - mCurrentZoom) / (1.0 + i));
	}

	mCurrentZoom = ti.zoom = std::clamp(mCurrentZoom + gradient, data.zoomMin, data.zoomMax);
}

//build AffineTransform from trajectory
const AffineTransform& Trajectory::getTransform(const MainData& data, int64_t frameIndex) {
	TrajectoryItem& ti = mTrajectory[frameIndex];

	//get transform to apply to image
	mCurrentTransform.reset()
		.addTranslation(data.w / 2.0, data.h / 2.0)          //translate to origin
		.addRotation(ti.smoothed.a)                        //rotation
		.addZoom(ti.zoom)                                    //zoom as set in trajectory
		.addTranslation(ti.smoothed.u, ti.smoothed.v)    //computed translation to stabilize
		.addTranslation(data.w / -2.0, data.h / -2.0)        //translate back to center
		;
	mCurrentTransform.frameIndex = frameIndex;
	return mCurrentTransform;
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

double Trajectory::calcRequiredZoom(double dx, double dy, double rot, double w, double h) {
	double w2 = w / 2.0;
	double h2 = h / 2.0;
	Affine2D trf = Affine2D::setTransforms(1.0, -rot, dx, dy);
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
	return zoom;
}

std::ostream& operator << (std::ostream& os, const Trajectory& trajectory) {
	for (int i = 0; i < trajectory.mTrajectory.size(); i++) {
		const TrajectoryItem& item = trajectory.mTrajectory[i];
		os << std::format("{} [{}] u={:10.7f}, v={:10.7f}, aSec={:10.7f} / smu={:10.7f}, smv={:10.7f}, smaSec={:10.7f} / zoom={:7.5f}",
			i, item.frameIndex, item.values.u, item.values.v, item.values.a * 1000, item.smoothed.u, item.smoothed.v, item.smoothed.a * 1000, item.zoom) 
			<< std::endl;
	}
	return os;
}
