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
	mBestCluster.reserve(siz);
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
	const size_t cMinConsensPoints = 8; //min numbers of points for consensus set
	const double cDbScan = 0.08;        //when to try dbscan

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

	if (numValid > cMinConsensPoints) {
		//util::ConsoleTimer ic("trf " + std::to_string(frameIndex));

		// STEP 1 traditional method
		mBestTransform = computeClassic(numValid, frameIndex);
		//std::cout << "frame " << frameIndex << " mConsList size " << mConsList.size() << std::endl;

		if (mConsList.size() < numValid * cDbScan) {
			// STEP 2 dbscan
			computeDbScan(frameIndex);

			if (mClusterSizes.size() == 0) {
				//likely a hard scene change, return default transform
				mBestTransform = AffineTransform();

			} else {
				int idx = mClusterSizes[0].index;
				auto fcnSelect = [&] (const PointContext& pc) { return pc.clusterIndex == idx; };
				//writeVideo({}, mBestCluster, frameIndex, "dbscan");
				//TODO
				mBestCluster.clear();
				std::copy_if(mPointList.begin(), mPointList.end(), std::back_inserter(mBestCluster), fcnSelect);
			}

		}

		for (PointContext& pc : mConsList) pc.ptr->isConsens = true;
	}

	//std::cout << "frame " << frameIndex << ", " << mBestTransform << std::endl;
	return mBestTransform;
}

AffineTransform FrameResult::computeClassic(size_t numValid, int64_t frameIndex) {
	auto sortDist = [] (const PointContext& pc1, const PointContext& pc2) { return pc1.distance < pc2.distance; };
	auto sortDelta = [] (const PointContext& pc1, const PointContext& pc2) { return pc1.delta < pc2.delta; };
	auto sortRel = [] (const PointContext& pc1, const PointContext& pc2) { return pc1.distanceRelative < pc2.distanceRelative; };

	constexpr int cConsLoopCount = 8;                     //max number of loops when searching for consensus set
	constexpr int cConsLoopPercent = 95;                  //percentage of points for next loop 0..100
	constexpr double cConsensDistRelative = 0.2;          //max offset normalized by length
	constexpr double cConsensDistanceSqr = sqr(1.25);     //max offset for a point to be in the consensus set

	//calculate average displacement distance and cut off outsiders
	double averageLength = 0.0;
	for (PointContext& pc : mConsList) {
		averageLength += pc.length;
	}
	averageLength /= numValid;

	//set delta value to deviation from average length
	for (PointContext& pc : mConsList) {
		pc.delta = std::abs(pc.length - averageLength);
	}

	std::sort(mConsList.begin(), mConsList.end(), sortDelta);
	mConsList.resize(numValid * cConsLoopPercent / 100);
	AffineTransform trf;

	size_t numCons = 0;
	for (int i = 0; i < cConsLoopCount; i++) {
		//average length of current points
		averageLength = 0.0;
		for (const PointContext& pc : mConsList) {
			averageLength += pc.length;
		}
		averageLength /= mConsList.size();

		//transform for selected points
		trf = mAffineSolver->computeSimilar(mConsList);

		size_t numConsAbsolute = 0;
		size_t numConsRelative = 0;
		//calculate error distance based on transform
		for (PointContext& pc : mConsList) {
			auto [tx, ty] = mAffineSolver->transform(pc.x, pc.y);
			pc.distance = sqr(pc.x + pc.u - tx) + sqr(pc.y + pc.v - ty);
			pc.distanceRelative = pc.distance / pc.length;

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

bool FrameResult::clusterDistance(const PointContext& pc1, const PointContext& pc2) {
	constexpr double eps = 7.5;
	const double fa = 10000.0 / std::max(mData.w, mData.h);
	const double fl = 25.0;

	double s = pc1.length + pc2.length;
	double da = 1.0 - (pc1.u * pc2.u + pc1.v * pc2.v) / (pc1.length * pc2.length); //dot product represents cos(angle)
	double f1 = da * fa * s;
	double dl = std::abs(pc1.length - pc2.length) / (s + 1.0); //length difference normalized
	double f2 = dl * fl;

	return f1 + f2 < eps;
}

//adaptation of dbscan to find clusters of movements
void FrameResult::computeDbScan(int64_t frameIndex) {
	//writeImage(mBestTransform, mConsList, frameIndex, "classic");
	//util::ConsoleTimer ct("dbscan " + std::to_string(frameIndex));
	constexpr int minPts = 25;

	//build clusters
	mClusterSizes.clear();
	for (PointContext& pc : mPointList) {
		int idxRead = 0;
		int idxWrite = 0;
		int idxMarker = 0;
	
		if (pc.clusterIndex == -2) {
			//build new region including center point
			for (PointContext& check : mPointList) {
				if (check.clusterIndex < 0 && clusterDistance(pc, check)) {
					mWork[idxWrite] = &check;
					idxWrite++;
				}
			}

			//check neighborhood size including center point
			if (idxWrite > minPts) {
				//build new cluster
				int clusterIdx = mClusterSizes.size();
				mWork[0]->clusterIndex = clusterIdx;
				mWork[0]->clusterGeneration = 0;
				for (int idx = 1; idx < idxWrite; idx++) {
					mWork[idx]->clusterIndex = clusterIdx;
					mWork[idx]->clusterGeneration = 0;
				}

				//do not expand cluster recursively, seems to work better this way
				int numel = idxWrite;
				for (idxRead = 1; idxRead < numel; idxRead++) {
					const PointContext* pr = mWork[idxRead];

					//neighborhood to this center point
					idxMarker = idxWrite;
					for (PointContext& check : mPointList) {
						if (check.clusterIndex < 0 && clusterDistance(*pr, check)) {
							mWork[idxWrite] = &check;
							idxWrite++;
						}
					}

					//check neighborhood size excluding center point
					if (idxWrite - idxMarker >= minPts) {
						//mark as belonging to this cluster
						for (int idx = idxMarker; idx < idxWrite; idx++) {
							mWork[idx]->clusterIndex = clusterIdx;
							mWork[idx]->clusterGeneration = pr->clusterGeneration + 1;
						}

					} else {
						//discard too small neighborhood
						idxWrite = idxMarker;
					}
				}

				//store established cluster size
				mClusterSizes.emplace_back(clusterIdx, idxWrite);
			}
		}
	}

	auto fcnSort = [] (const ClusterSize& cs1, const ClusterSize& cs2) { return cs1.siz > cs2.siz; };
	std::sort(mClusterSizes.begin(), mClusterSizes.end(), fcnSort);
	assert(checkSizes() && "invalid clusters");
	//std::cout << "frame " << frameIndex << ": [" << mClusterSizes.size() << "] " << util::collectionToString(mClusterSizes, 15) << std::endl;
}

//for debugging
bool FrameResult::checkSizes() {
	int sum = std::count_if(mPointList.begin(), mPointList.end(), [&] (const PointContext& pc) { return pc.clusterIndex < 0; });
	for (const ClusterSize& cs : mClusterSizes) sum += cs.siz;
	return sum == mPointList.size();
}

//for debugging
std::ostream& operator << (std::ostream& out, const FrameResult::ClusterSize& cs) {
	return out << cs.siz << "-" << cs.index;
}

//for debugging
void FrameResult::writeVideo(const AffineTransform& trf, std::span<PointContext> res, int64_t frameIndex, const std::string& title) {
	static std::ofstream file("f:/pic/results.yuv", std::ios::binary);
	ImageYuv yuv(mData.h, mData.w);
	yuv.setColor(Color::DARK_GRAY);
	std::vector<PointResult> resvec;
	for (const PointContext& pc : res) resvec.push_back(*pc.ptr);
	ResultImageWriter::writeImage(trf, resvec, frameIndex, yuv, mPool);
	yuv.writeText(title, mData.w / 2, 0, 2, 2, im::TextAlign::TOP_CENTER);
	yuv.save(file);
}