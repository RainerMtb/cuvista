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

#include "MovieWriterImpl.hpp"
#include "FrameResult.hpp"
#include "Util.hpp"

using namespace util;


//--------------------------
//---- setup ---------------
//--------------------------

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
	mResultData.consList.resize(siz);
	mResultData.bestCluster.resize(siz);
	for (int i = 0; i < 4; i++) {
		mResultData.clusters.emplace_back(i);
		mResultData.clusters[i].points.resize(siz);
		mResultData.clusters[i].work.resize(siz);
	}

	//collect dbscan flags
	mResultData.flags.resize(6);

	params.f1 = 25000.0 / std::max(mData.w, mData.h);
	params.f2 = 25.0;
}


//--------------------------
//---- functions -----------
//--------------------------

const AffineTransform& FrameResult::getTransform() const {
	return mResultData.transform;
}

void FrameResult::reset() {
	mAffineSolver->reset();
	mAffineSolver->frameIndex = 0;
	mResultData.transform.reset();
}

bool FrameResult::clusterDistance(const PointContext& pc1, const PointContext& pc2) const {
	double s = pc1.length + pc2.length;
	double da = 1.0 - (pc1.u * pc2.u + pc1.v * pc2.v) / (pc1.length * pc2.length); //dot product represents cos(angle)
	double f1 = da * params.f1 * s;
	double dl = std::abs(pc1.length - pc2.length) / (s + 1.0); //length difference normalized
	double f2 = dl * params.f2;

	return f1 + f2 < params.eps;
}

struct {
	bool operator () (const PointContext& pc1, const PointContext& pc2) const {
		return pc1.distanceRelative < pc2.distanceRelative;
	}
} SortRel;

struct {
	bool operator () (const PointContext& pc1, const PointContext& pc2) const {
		return pc1.distance < pc2.distance;
	}
} SortDist;

struct {
	bool operator () (const PointContext& pc1, const PointContext& pc2) const {
		return pc1.delta < pc2.delta;
	}
} SortDelta;


//--------------------------------
//--------- main entry point -----
//--------------------------------

const AffineTransform& FrameResult::computeTransform(std::span<PointResult> results, int64_t frameIndex, double gamma) {
	//util::ConsoleTimer timer("FrameResult");
	mAffineSolver->reset();
	mAffineSolver->frameIndex = frameIndex;
	mResultData.frameIndex = frameIndex;
	mResultData.gamma = gamma;
	mResultData.bestCluster.clear();
	mResultData.bestClusterVector = -1;
	mResultData.bestClusterIndex = -1;

	std::ranges::fill(mResultData.flags, 0);
	mResultData.consList.clear();
	size_t numValid = 0;
	//consider region of interest and build list of PointContext
	for (PointResult& pr : results) {
		pr.isConsens = false;
		pr.isConsidered = false;
		double x = pr.x + mData.w / 2.0;
		double y = pr.y + mData.h / 2.0;
		const RoiCrop& roi = mData.roiCrop;
		if (pr.isValid() && x > roi.horizontal && x < mData.w - roi.horizontal && y > roi.vertical && y < mData.h - roi.vertical) {
			pr.isConsidered = true;
			numValid++;
		}
	}

	if (numValid > params.minConsPoints) {
		mResultData.transform.reset();
		//util::ConsoleTimer ic("trf " + std::to_string(frameIndex));

		// STEP 1 traditional method
		if (mData.runTransformClassic) {
			mResultData.consList.resize(numValid);
			std::ranges::copy_if(results, mResultData.consList.begin(), [] (const PointResult& pr) { return pr.isConsidered; });
			mResultData.transform = computeClassic(numValid, frameIndex);
			//debugLogger().format("frame {} classic {}", frameIndex, mResultData.consList.size());
		}

		// STEP 2 dbscan
		mResultData.flags[0] = mData.runTransformDbScan;
		mResultData.flags[1] = mResultData.consList.size() < numValid * params.minDbScanRel;
		mResultData.flags[2] = mResultData.consList.size() < params.minDbScanAbs;
		if (std::accumulate(mResultData.flags.begin(), mResultData.flags.end(), 0) == 3) {
			//run dbscan on points in different order
			std::vector<std::future<void>> futs(4);
			for (int i = 0; i < 4; i++) {
				futs[i] = mPool.add([&, numValid, frameIndex, i] { computeDbScan(results, numValid, frameIndex, i); });
			}
			for (auto& f : futs) f.wait();

			//find largest cluster
			ClusterData nullCluster = {};
			const ClusterData* bestCluster = &nullCluster;
			int sizMax = 0;
			for (const ClusterData& cd : mResultData.clusters) {
				if (cd.sizes.size() > 0) {
					const ClusterSize& cs = cd.sizes.front();
					if (cs.siz > sizMax) {
						sizMax = cs.siz;
						mResultData.bestClusterVector = cd.index;
						mResultData.bestClusterIndex = cs.index;
						bestCluster = &cd;
					}
				}
			}

			size_t sizeBestCluster = 0;
			size_t sizeAllClusters = 0;
			if (mResultData.bestClusterVector > -1) {
				auto fcnSelect = [&] (const PointContext& pc) { return pc.clusterIndex == mResultData.bestClusterIndex; };
				std::ranges::copy_if(bestCluster->points, std::back_inserter(mResultData.bestCluster), fcnSelect);
				sizeBestCluster = mResultData.bestCluster.size();
				sizeAllClusters = std::accumulate(bestCluster->sizes.begin(), bestCluster->sizes.end(), 0ull, [] (size_t s, const ClusterSize& cs) { return s + cs.siz; });
			}

			//compare clusters and classic result
			size_t numCons = mResultData.consList.size();
			mResultData.flags[3] = sizeBestCluster > numCons * 10;
			mResultData.flags[4] = sizeBestCluster > sizeAllClusters * 10 / 100;
			mResultData.flags[5] = bestCluster->sizes.size() < 20;
			if (sizeBestCluster == 0) {
				//likely a hard scene change, return default transform
				mResultData.transform = AffineTransform();
				mResultData.consList.clear();
				mResultData.dbscanIndex++;
				//debugLogger().format("frame {} dbscan no cluster", frameIndex);
				//writeVideo(mPointList, frameIndex, std::to_string(frameIndex) + " points");

			} else if (std::accumulate(mResultData.flags.begin(), mResultData.flags.end(), 0) == 6) {
				//compute transform from largest cluster
				computeDbScanTransform(frameIndex);
				mResultData.consList = mResultData.bestCluster;
				mResultData.dbscanIndex++;
				//debugLogger().format("#{} frame {} [{}] dbscan {}", mResultData.dbscanIndex, frameIndex, bestCluster->sizes.size(), collectionToString(bestCluster->sizes, 8));
			}
		}

		//mark final set of consens points
		for (PointContext& pc : mResultData.consList) pc.ptr->isConsens = true;
	}

	//std::cout << "frame " << frameIndex << ", " << mBestTransform << std::endl;
	return mResultData.transform;
}


//-----------------------------
//--------- classic -----------
//-----------------------------

AffineTransform FrameResult::computeClassic(size_t numValid, int64_t frameIndex) {
	//calculate average displacement distance and cut off outsiders
	double averageLength = 0.0;
	for (PointContext& pc : mResultData.consList) {
		averageLength += pc.length;
	}
	averageLength /= numValid;

	//set delta value to deviation from average length
	for (PointContext& pc : mResultData.consList) {
		pc.delta = std::abs(pc.length - averageLength);
	}

	std::sort(mResultData.consList.begin(), mResultData.consList.end(), SortDelta);
	mResultData.consList.resize(numValid * params.consLoopPercent / 100);
	AffineTransform trf;

	size_t numCons = 0;
	for (int i = 0; i < params.consLoopCount && mResultData.consList.size() > params.minConsPoints; i++) {
		//average length of current points
		averageLength = 0.0;
		for (const PointContext& pc : mResultData.consList) {
			averageLength += pc.length;
		}
		averageLength /= mResultData.consList.size();

		//transform for selected points
		trf = mAffineSolver->computeSimilar(mResultData.consList);

		numCons = 0;
		size_t numConsAbsolute = 0;
		size_t numConsRelative = 0;
		//calculate error distance based on transform
		for (PointContext& pc : mResultData.consList) {
			auto [tx, ty] = mAffineSolver->transform(pc.x, pc.y);
			pc.distance = sqr(pc.x + pc.u - tx) + sqr(pc.y + pc.v - ty);
			pc.distanceRelative = pc.distance / pc.length;

			if (pc.distance < params.consDistanceSqr) {
				numConsAbsolute++;
				numCons++;

			} else if (pc.distanceRelative < params.consDistRelative) {
				numConsRelative++;
				numCons++;
			}
		}

		//only then rely on relatives when there are many more than absolute ones
		if (numConsAbsolute > 40 || numConsAbsolute * 10 > numConsRelative) {
			std::ranges::sort(mResultData.consList, SortDist);

		} else {
			//std::cout << data.status.frameInputIndex << " ABS " << numConsAbsolute << " REL " << numConsRelative << std::endl;
			std::ranges::sort(mResultData.consList, SortRel);
		}

		//stop if enough points are consens
		size_t cutoff = mResultData.consList.size() * params.consLoopPercent / 100;
		if (numCons > cutoff) {
			break;

		} else {
			mResultData.consList.resize(cutoff);
		}
	}

	mResultData.consList.resize(numCons);
	return trf;
}


//----------------------------
//--------- dbscan -----------
//----------------------------

//adaptation of dbscan to find clusters of movements
void FrameResult::computeDbScan(std::span<PointResult> results, size_t numValid, int64_t frameIndex, size_t index) {
	//writeImage(mBestTransform, mConsList, frameIndex, "classic");
	//util::ConsoleTimer ct("dbscan " + std::to_string(frameIndex));
	ClusterData& cd = mResultData.clusters[index];
	std::vector<PointContext>& points = cd.points;
	std::vector<PointContext*>& work = cd.work;
	cd.sizes.clear();

	//copy input points
	switch (index) {
	case 0:
		//normal order
		points.resize(numValid);
		std::copy_if(results.begin(), results.end(), points.begin(), [] (const PointResult& pr) { return pr.isConsidered; });
		break;

	case 1:
		//reversed order
		points.resize(numValid);
		std::copy_if(results.rbegin(), results.rend(), points.begin(), [] (const PointResult& pr) { return pr.isConsidered; });
		break;

	case 2:
		//column major
		points.clear();
		for (int c = 0; c < mData.ixCount; c++) {
			for (int r = 0; r < mData.iyCount; r++) {
				int idx = c + r * mData.ixCount;
				if (PointResult& pr = results[idx]; pr.isValid()) points.emplace_back(pr);
			}
		}
		break;

	case 3:
		//reversed column major
		points.clear();
		for (int c = mData.ixCount - 1; c >= 0; c--) {
			for (int r = mData.iyCount - 1; r >= 0; r--) {
				int idx = c + r * mData.ixCount;
				if (PointResult& pr = results[idx]; pr.isValid()) points.emplace_back(pr);
			}
		}
		break;
	}

	//build clusters
	for (PointContext& pc : points) {
		int idxRead = 0;
		int idxWrite = 0;
		int idxMarker = 0;

		if (pc.clusterIndex == -2) {
			//build new region including center point
			for (PointContext& check : points) {
				if (check.clusterIndex < 0 && clusterDistance(pc, check)) {
					work[idxWrite] = &check;
					idxWrite++;
				}
			}

			//check neighborhood size including center point
			if (idxWrite > params.minPts) {
				//build new cluster
				int clusterIdx = cd.sizes.size();
				work[0]->clusterIndex = clusterIdx;
				work[0]->clusterGeneration = 0;
				for (int idx = 1; idx < idxWrite; idx++) {
					work[idx]->clusterIndex = clusterIdx;
					work[idx]->clusterGeneration = 0;
				}

				//do not expand cluster recursively, seems to work better this way
				int numel = idxWrite;
				for (idxRead = 1; idxRead < numel; idxRead++) {
					const PointContext* pr = work[idxRead];

					//neighborhood to this center point
					idxMarker = idxWrite;
					for (PointContext& check : points) {
						if (check.clusterIndex < 0 && clusterDistance(*pr, check)) {
							work[idxWrite] = &check;
							idxWrite++;
						}
					}

					//check neighborhood size excluding center point
					if (idxWrite - idxMarker >= params.minPts) {
						//mark as belonging to this cluster
						for (int idx = idxMarker; idx < idxWrite; idx++) {
							work[idx]->clusterIndex = clusterIdx;
							work[idx]->clusterGeneration = pr->clusterGeneration + 1;
						}

					} else {
						//discard too small neighborhood
						idxWrite = idxMarker;
					}
				}

				//store established cluster size
				cd.sizes.emplace_back(clusterIdx, idxWrite);
			}
		}
	}

	std::ranges::sort(cd.sizes, [] (const ClusterSize& cs1, const ClusterSize& cs2) { return cs1.siz > cs2.siz; });
	//std::cout << "frame " << frameIndex << ": [" << mClusterSizes.size() << "] " << util::collectionToString(mClusterSizes, 15) << std::endl;
}

void FrameResult::computeDbScanTransform(int64_t frameIndex) {
	AffineTransform trf;

	trf = mAffineSolver->computeSimilar(mResultData.bestCluster);
	for (PointContext& pc : mResultData.bestCluster) {
		auto [tx, ty] = mAffineSolver->transform(pc.x, pc.y);
		pc.distance = sqr(pc.x + pc.u - tx) + sqr(pc.y + pc.v - ty);
		pc.distanceRelative = pc.distance / pc.length;
	}
	std::ranges::sort(mResultData.bestCluster, SortRel);

	//writeVideo(mConsList, frameIndex, std::to_string(frameIndex) + " classic"); // <<<<<<<<<<<<
	//writeVideo(mBestCluster, frameIndex, std::to_string(frameIndex) + " dbscan"); // <<<<<<<<<<<<
	size_t siz = mResultData.bestCluster.size() * params.finalSizePercent / 100;
	mResultData.bestCluster.resize(siz);
	mResultData.transform = mAffineSolver->computeSimilar(mResultData.bestCluster);
}

std::vector<int> FrameResultData::getClusterSizes() const {
	std::vector<int> out;
	if (bestClusterVector > -1) {
		for (const ClusterSize& cs : clusters[bestClusterVector].sizes) {
			out.push_back(cs.siz);
		}
	}
	return out;
}

const FrameResultData& FrameResult::getResultData() const {
	return mResultData;
}

//--- for debugging
std::ostream& operator << (std::ostream& out, const ClusterSize& cs) {
	return out << cs.siz;
}

//--- for debugging
void FrameResult::writeVideo(std::span<PointContext> res, int64_t frameIndex, const std::string& title) const {
	static std::ofstream file("f:/resultVid.nv12", std::ios::binary);
	static ImageNV12 im(mData.h, mData.w);
	ImageBgr bgr(mData.h, mData.w);
	bgr.setColor(Color::DARK_GRAY);
	std::vector<PointResult> resvec(res.begin(), res.end());
	ResultImageWriter::writeImage({}, resvec, frameIndex, bgr, mPool, false);
	bgr.writeText(title, mData.w / 2, 0, im::TextAlign::TOP_CENTER);
	bgr.convertTo(im);
	im.write(file);
}
