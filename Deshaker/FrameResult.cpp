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


struct ClusterSizes {
	int index;
	int siz;

	bool operator < (const ClusterSizes& other) {
		return siz > other.siz; //sort in descending order
	}
};


FrameResult::FrameResult(MainData& data, ThreadPoolBase& threadPool) :
	mData { data },
	mPool { threadPool }
{

	//create solver class
	if (data.hasAvx512()) {
		mAffineSolver = std::make_unique<AffineSolverAvx>(data.resultCount);
	} else {
		mAffineSolver = std::make_unique<AffineSolverFast>(threadPool, data.resultCount);
	}

	//prepare result lists
	mConsList.reserve(data.resultCount);
	mList1.resize(data.resultCount);
	mList2.resize(data.resultCount);

	//init debug storage and put in first item
	resultStore.clear();
	resultStore.emplace_back(0);
}

const AffineTransform& FrameResult::getTransform() const {
	return mBestTransform;
}

void FrameResult::reset() {
	mAffineSolver->reset();
	mAffineSolver->frameIndex = 0;
	mBestTransform.reset();
}

const AffineTransform& FrameResult::computeTransform(std::span<PointResult> results, int64_t frameIndex) {
	using namespace util;

	const size_t cMinConsensPoints = 8;	              //min numbers of points for consensus set
	const int cConsLoopCount = 8;			          //max number of loops when searching for consensus set
	const int cConsLoopPercent = 95;		          //percentage of points for next loop 0..100
	const double cConsensDistanceSqr = sqr(1.25);     //max offset for a point to be in the consensus set
	const double cConsensDistRelative = 0.2;          //max offset normalized by length

	auto sortDist = [] (const PointContext& pc1, const PointContext& pc2) { return pc1.distance < pc2.distance; };
	auto sortDelta = [] (const PointContext& pc1, const PointContext& pc2) { return pc1.delta < pc2.delta; };
	auto sortRel = [] (const PointContext& pc1, const PointContext& pc2) { return pc1.distanceRelative < pc2.distanceRelative; };

	//util::ConsoleTimer ic("computeTransform");
	mAffineSolver->reset();
	mAffineSolver->frameIndex = frameIndex;

	//filter considerable points by region of interest
	mConsList.clear();
	//consider region of interest
	for (PointResult& pr : results) {
		pr.isConsens = false;
		pr.isConsidered = false;
		double x = pr.x + mData.w / 2.0;
		double y = pr.y + mData.h / 2.0;
		const RoiCrop& roi = mData.roiCrop;
		if (pr.isValid() && x > roi.left && x < mData.w - roi.right && y > roi.top && y < mData.h - roi.bottom) {
			pr.isConsidered = true;
			mConsList.emplace_back(pr);
		}
	}
	size_t numValid = mConsList.size();
	
	//second copy of initial points
	mList2 = mConsList;

	if (numValid > cMinConsensPoints) {
		//calculate average displacement distance and cut off outsiders
		double averageLength = 0.0;
		for (PointContext& pc : mConsList) {
			pc.ptr->length = std::sqrt(sqr(pc.ptr->u) + sqr(pc.ptr->v));
			averageLength += pc.ptr->length;
		}
		averageLength /= numValid;
		for (PointContext& pc : mConsList) {
			//set delta value to deviation from average length
			pc.delta = std::abs(pc.ptr->length - averageLength);
		}
		std::sort(mConsList.begin(), mConsList.end(), sortDelta);
		mConsList.resize(numValid * cConsLoopPercent / 100);
		AffineTransform trf1, trf2;

		// STEP 1
		// traditional method
		{
			size_t numCons = 0;
			for (int i = 0; i < cConsLoopCount; i++) {
				//average length of current points
				averageLength = 0.0;
				for (const PointContext& pc : mConsList) {
					averageLength += pc.ptr->length;
				}
				averageLength /= mConsList.size();

				//transform for selected points
				trf1 = mAffineSolver->computeSimilar(mConsList);

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
		}
		mBestTransform = trf1;
		//std::cout << "frame " << frameIndex << " mConsList size " << mConsList.size() << std::endl;

		// STEP 2
		// using dbscan
		if (mConsList.size() < numValid / 5) {
			const int cClusterMinPts = 10;
			const int cClusterBest = 4;
			const double cClusterEps = sqr(0.75);

			//cluster -1: noise, 0: unclassified, >0: cluster index 
			//follow cluster sizes
			std::vector<ClusterSizes> clusterSizes = { {0, 0}, {1, 0} };
			int clusterIdx = 1;

			//calculate uu and vv
			int m = std::max(mData.h, mData.w);
			double mm = m * m;
			auto f = [&] (double d) { return -0.125 * d * d / mm + 0.375 * d; };

			for (PointContext& pc : mList2) {
				pc.uu = f(pc.ptr->u);
				pc.vv = f(pc.ptr->v);
			}

			auto distfun = [] (const PointContext& pc1, const PointContext& pc2) { return sqr(pc1.uu - pc2.uu) + sqr(pc1.vv - pc2.vv); };
			
			/*
			//iterate over unclassified values
			for (PointContext& pc : mList2) {
				if (pc.clusterIdx == 0) {
					mRegion.clear();

					//build neighborhood
					for (PointContext& pc2 : mList2) {
						if (distfun(pc, pc2) < cClusterEps) mRegion.push_back(&pc2);
					}

					if (mRegion.size() < cClusterMinPts) {
						pc.clusterIdx = -1; //classify as noise for now

					} else {
						for (size_t idx = 0; idx < mRegion.size(); idx++) {
							PointContext& pc2 = *mRegion[idx];
							if (pc2.clusterIdx <= 0) {
								pc2.clusterIdx = clusterIdx; //add to current cluster
								clusterSizes[clusterIdx].siz++;

								//build new neighborhood
								size_t siz = mRegion.size();
								for (PointContext& pc3 : mList2) {
									if (distfun(pc2, pc3) < cClusterEps) mRegion.push_back(&pc3);
								}

								//discard neighbors if below minPts
								if (mRegion.size() - siz < cClusterMinPts) {
									mRegion.erase(mRegion.begin() + siz, mRegion.end());
								}
							}
						}

						//begin next cluster
						clusterIdx++;
						clusterSizes.emplace_back(clusterIdx, 0);
					}
				}
			}

			//analyse clusters
			std::sort(clusterSizes.begin(), clusterSizes.end());

			//try best clusters
			std::vector<PointContext>::iterator it;
			for (int i = 0; i < cClusterBest && i < clusterSizes.size() - 2 ; i++) {
				int cluster = clusterSizes[i].index;
				auto pred = [&] (const PointContext& pc) { return pc.clusterIdx == cluster; };
				it = std::copy_if(mList2.begin(), mList2.end(), mList1.begin(), pred);
				trf2 = mAffineSolver->computeSimilar({ mList1.begin(), it });

				it = mList1.begin();
				for (PointContext& pc : mList2) {
					PointResult& pr = *pc.ptr;
					auto [tx, ty] = mAffineSolver->transform(pr.x, pr.y);
					pc.distance = sqr(pr.x + pr.u - tx) + sqr(pr.y + pr.v - ty);
					pc.distanceRelative = pc.distance / pr.length;
					if (pc.distance < cConsensDistanceSqr || pc.distanceRelative < cConsensDistRelative) {
						*it++ = pc;
					}
				}

				std::span<PointContext> span(mList1.begin(), it);
				if (span.size() > mConsList.size()) {
					mConsList = { span.begin(), span.end() };
					mBestTransform = trf2;
					std::cout << "cluster " << cluster << " consens " << span.size() << std::endl;
				}
			}
			*/

			/*
			//output results
			ImageBGR bgr(mData.h, mData.w);
			bgr.setValues({ 220, 230, 240 });
			std::vector<std::string> colorStrings = { "#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F" };
			std::vector<im::ColorBgr> colors(colorStrings.size());
			for (int idx = 0; idx < colorStrings.size(); idx++) colors[idx] = im::ColorBgr::webColor(colorStrings[idx]);

			for (int idx = 0; idx < 5 && idx < clusterSizes.size() - 2; idx++) {
				int i = clusterSizes[idx].index;

				for (const PointContext& pc : mConsList2) {
					if (pc.clusterIdx == i) {
						PointResult& pr = *pc.ptr;
						double px = pr.x + mData.w / 2.0;
						double py = pr.y + mData.h / 2.0;
						double x2 = px + pr.u;
						double y2 = py + pr.v;

						const im::ColorBgr& col = colors[idx];
						bgr.drawLine(px, py, x2, y2, col);
						bgr.drawMarker(x2, y2, col, 1.4);
					}
				}
			}

			std::string filename = std::format("f:/pic/out{:02}clusters.bmp", frameIndex);
			bgr.saveAsColorBMP(filename);
			//std::cout << filename << ", vectors: " << numPoints << std::endl;
			*/
		}

		for (PointContext& pc : mConsList) pc.ptr->isConsens = true;
	}

	//resultStore.emplace_back(frameIndex, std::vector<PointResult>(results.begin(), results.end())); //-----------
	//std::cout << "frame " << frameIndex << ", " << mBestTransform << std::endl;
	return mBestTransform;
}