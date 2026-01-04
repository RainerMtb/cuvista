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

#include "MovieWriter.hpp"
#include "Util.hpp"


FrameResult::FrameResult(MainData& data, ThreadPoolBase& threadPool) :
	mData { data },
	mPool { threadPool }
{
	size_t siz = data.resultCount;

	//create solver class
	if (data.hasAvx512()) {
		mAffineSolver = std::make_unique<AffineSolverAvx>(siz);
	} else {
		mAffineSolver = std::make_unique<AffineSolverFast>(threadPool, siz);
	}

	//prepare lists
	mConsList.reserve(siz);
	mPointList.resize(siz);
	mWork.resize(siz);
	mCluster.resize(siz);
}

const AffineTransform& FrameResult::getTransform() const {
	return mBestTransform;
}

void FrameResult::reset() {
	mAffineSolver->reset();
	mAffineSolver->frameIndex = 0;
	mBestTransform.reset();
}

using namespace util;

const AffineTransform& FrameResult::computeTransform(std::span<PointResult> results, int64_t frameIndex) {

	mAffineSolver->reset();
	mAffineSolver->frameIndex = frameIndex;

	mConsList.clear();
	//consider region of interest and build list of PointContext
	for (PointResult& pr : results) {
		pr.isConsens = false;
		pr.isConsidered = false;
		double x = pr.x + mData.w / 2.0;
		double y = pr.y + mData.h / 2.0;
		const RoiCrop& roi = mData.roiCrop;
		if (pr.isValid() && x > roi.horizontal && x < mData.w - roi.horizontal && y > roi.vertical && y < mData.h - roi.vertical) {
			pr.isConsidered = true;
			mConsList.emplace_back(pr);
		}
	}
	size_t numValid = mConsList.size();
	
	//second copy of initial points
	mPointList = mConsList;

	const size_t cMinConsensPoints = 8; //min numbers of points for consensus set
	if (numValid > cMinConsensPoints) {
		//util::ConsoleTimer ic("trf " + std::to_string(frameIndex));

		// STEP 1 traditional method
		mBestTransform = computeClassic(numValid, frameIndex);
		//std::cout << "frame " << frameIndex << " mConsList size " << mConsList.size() << std::endl;
		//ic.interval("trf classic");

		if (mConsList.size() < numValid / 10) {
			// STEP 2 dbscan
			mBestTransform = computeDbScan(frameIndex);
		}
		//ic.interval("trf dbscan");

		for (PointContext& pc : mConsList) pc.ptr->isConsens = true;
	}

	//std::cout << "frame " << frameIndex << ", " << mBestTransform << std::endl;
	return mBestTransform;
}

AffineTransform FrameResult::computeClassic(size_t numValid, int64_t frameIndex) {
	auto sortDist = [] (const PointContext& pc1, const PointContext& pc2) { return pc1.distance < pc2.distance; };
	auto sortDelta = [] (const PointContext& pc1, const PointContext& pc2) { return pc1.delta < pc2.delta; };
	auto sortRel = [] (const PointContext& pc1, const PointContext& pc2) { return pc1.distanceRelative < pc2.distanceRelative; };

	const int cConsLoopCount = 8;			          //max number of loops when searching for consensus set
	const int cConsLoopPercent = 95;		          //percentage of points for next loop 0..100
	const double cConsensDistanceSqr = sqr(1.25);     //max offset for a point to be in the consensus set
	const double cConsensDistRelative = 0.2;          //max offset normalized by length

	//calculate average displacement distance and cut off outsiders
	double averageLength = 0.0;
	for (PointContext& pc : mConsList) {
		pc.ptr->length = std::sqrt(sqr(pc.ptr->u) + sqr(pc.ptr->v));
		averageLength += pc.ptr->length;
	}
	averageLength /= numValid;

	//set delta value to deviation from average length
	for (PointContext& pc : mConsList) {
		pc.delta = std::abs(pc.ptr->length - averageLength);
	}

	std::sort(mConsList.begin(), mConsList.end(), sortDelta);
	mConsList.resize(numValid * cConsLoopPercent / 100);
	AffineTransform trf;

	size_t numCons = 0;
	for (int i = 0; i < cConsLoopCount; i++) {
		//average length of current points
		averageLength = 0.0;
		for (const PointContext& pc : mConsList) {
			averageLength += pc.ptr->length;
		}
		averageLength /= mConsList.size();

		//transform for selected points
		trf = mAffineSolver->computeSimilar(mConsList);

		size_t numConsAbsolute = 0;
		size_t numConsRelative = 0;
		//calculate error distance based on transform
		for (PointContext& pc : mConsList) {
			PointResult& pr = *pc.ptr;
			auto [tx, ty] = mAffineSolver->transform(pr.x, pr.y);
			pc.distance = sqr(pr.x + pr.u - tx) + sqr(pr.y + pr.v - ty);
			pc.distanceRelative = pc.distance / pr.length;

			if (pc.distance < cConsensDistanceSqr)
				numConsAbsolute++;
			else if (pc.distanceRelative < cConsensDistRelative)
				numConsRelative++;
		}

		//only then rely on relatives when there are many more than absolute ones
		if (numConsAbsolute > 40 || numConsAbsolute * 10 > numConsRelative) {
			numCons = numConsAbsolute;
			std::sort(mConsList.begin(), mConsList.end(), sortDist);

		} else {
			//std::cout << data.status.frameInputIndex << " ABS " << numConsAbsolute << " REL " << numConsRelative << std::endl;
			numCons = numConsRelative;
			std::sort(mConsList.begin(), mConsList.end(), sortRel);
		}

		//stop if enough points are consens
		size_t cutoff = mConsList.size() * cConsLoopPercent / 100;
		if (numCons > cutoff) {
			break;

		} else {
			mConsList.resize(cutoff);
		}
	}

	mConsList.resize(numCons);
	return trf;
}

//adaptation of dbscan to find clusters of movements
AffineTransform FrameResult::computeDbScan(int64_t frameIndex) {

	return mBestTransform;
}

bool FrameResult::checkSizes() {
	int sum = std::count_if(mPointList.begin(), mPointList.end(), [&] (const PointContext& pc) { return pc.clusterIndex < 0; });
	for (const ClusterSize& cs : mClusterSizes) sum += cs.siz;
	return sum == mPointList.size();
}

std::ostream& operator << (std::ostream& out, const ClusterSize& cs) {
	return out << cs.siz << "-" << cs.index;
}