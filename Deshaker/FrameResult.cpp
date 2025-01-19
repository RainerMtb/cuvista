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

// --------------------------------------
// ------------ UTILS -------------------
// --------------------------------------

constexpr double FrameResult::sqr(double value) { 
	return value * value; 
}

void FrameResult::writeResults(std::span<PointContext> pc, const std::string& filename) {
	std::ofstream file(filename);
	file << "idx;x;y;u;v;Type" << std::endl;

	for (const PointContext& p : pc) {
		const PointResult& pr = *p.ptr;
		file << pr.idx << ";" << pr.x << ";" << pr.y << ";" << pr.u << ";" << pr.v << ";" << pr.resultValue() << std::endl;
	}
}

void writeImage(std::span<PointContext> pc, const std::string& filename, int h, int w) {
	ImageBGR bgr(h, w);
	bgr.setValues({ 200, 230, 240 });

	for (const PointContext& p : pc) {
		PointResult& pr = *p.ptr;
		double px = pr.x + w / 2.0;
		double py = pr.y + h / 2.0;
		double x2 = px + pr.u;
		double y2 = py + pr.v;

		im::ColorBgr col = im::ColorBgr::BLUE;
		bgr.drawLine(px, py, x2, y2, col);
		bgr.drawMarker(x2, y2, col, 1.4);
	}

	bgr.saveAsColorBMP(filename);
}


// --------------------------------------
// ------------ MAIN --------------------
// --------------------------------------

FrameResult::FrameResult(MainData& data, ThreadPoolBase& threadPool) :
	mData { data }
{
	if (data.hasAvx512()) {
		mAffineSolver = std::make_unique<AffineSolverAvx>(data.resultCount);
	} else {
		mAffineSolver = std::make_unique<AffineSolverFast>(threadPool, data.resultCount);
	}

	mConsList1.reserve(data.resultCount);
	mConsList2.reserve(data.resultCount);
	mList1.resize(data.resultCount);
	mList2.resize(data.resultCount);
	mList3.resize(data.resultCount);
}

const AffineTransform& FrameResult::getTransform() const {
	return mBestTransform;
}

void FrameResult::reset() {
	mAffineSolver->reset();
	mAffineSolver->frameIndex = 0;
	mBestTransform.reset();
}

void FrameResult::computeTransform(std::vector<PointResult>& results, ThreadPoolBase& threadPool, int64_t frameIndex, SamplerPtr sampler) {
	const size_t cMinConsensPoints = 8;	              //min numbers of points for consensus set
	const int cConsLoopCount = 8;			          //max number of loops when searching for consensus set
	const int cConsLoopPercent = 95;		          //percentage of points for next loop 0..100
	const double cConsensDistanceSqr = sqr(1.25);     //max offset for a point to be in the consensus set
	const double cConsensDistRelative = 0.2;          //max offset normalized by length
	const double cClusterFraction = 0.15;             

	auto sortDist = [] (const PointContext& pc1, const PointContext& pc2) { return pc1.distance < pc2.distance; };
	auto sortDelta = [] (const PointContext& pc1, const PointContext& pc2) { return pc1.delta < pc2.delta; };
	auto sortRel = [] (const PointContext& pc1, const PointContext& pc2) { return pc1.distanceRelative < pc2.distanceRelative; };

	//util::ConsoleTimer ic("computeTransform");
	mAffineSolver->reset();
	mAffineSolver->frameIndex = frameIndex;
	mConsList1.clear();

	//only valid points
	for (PointResult& pr : results) {
		pr.isConsens = false;
		if (pr.isValid()) {
			mConsList1.emplace_back(pr);
		}
	}
	size_t numValid = mConsList1.size();

	if (numValid > cMinConsensPoints) {
		//calculate average displacement distance and cut off outsiders
		double averageLength = 0.0;
		for (PointContext& pc : mConsList1) {
			pc.ptr->length = std::sqrt(sqr(pc.ptr->u) + sqr(pc.ptr->v));
			averageLength += pc.ptr->length;
		}
		averageLength /= numValid;
		for (PointContext& pc : mConsList1) {
			//set delta value to deviation from average length
			pc.delta = std::abs(pc.ptr->length - averageLength);
		}
		std::sort(mConsList1.begin(), mConsList1.end(), sortDelta);
		mConsList1.resize(numValid * cConsLoopPercent / 100);
		mConsList2 = mConsList1;
		AffineTransform trf1, trf2;

		// STEP 1 traditional method
		{
			size_t numCons = 0;
			for (int i = 0; i < cConsLoopCount; i++) {
				//average length of current points
				averageLength = 0.0;
				for (auto it = mConsList1.begin(); it != mConsList1.end(); it++) {
					averageLength += it->ptr->length;
				}
				averageLength /= mConsList1.size();

				//transform for selected points
				trf1 = mAffineSolver->computeSimilar(mConsList1);

				size_t numConsAbsolute = 0;
				size_t numConsRelative = 0;
				//calculate error distance based on transform
				for (auto it = mConsList1.begin(); it != mConsList1.end(); it++) {
					PointResult& pr = *it->ptr;
					auto [tx, ty] = mAffineSolver->transform(pr.x, pr.y);
					it->distance = sqr(pr.x + pr.u - tx) + sqr(pr.y + pr.v - ty);
					it->distanceRelative = it->distance / pr.length;

					if (it->distance < cConsensDistanceSqr)
						numConsAbsolute++;
					else if (it->distanceRelative < cConsensDistRelative)
						numConsRelative++;
				}

				//only then rely on relatives when there are many more than absolute ones
				if (numConsAbsolute > 40 || numConsAbsolute * 10 > numConsRelative) {
					numCons = numConsAbsolute;
					std::sort(mConsList1.begin(), mConsList1.end(), sortDist);

				} else {
					//std::cout << data.status.frameInputIndex << " ABS " << numConsAbsolute << " REL " << numConsRelative << std::endl;
					numCons = numConsRelative;
					std::sort(mConsList1.begin(), mConsList1.end(), sortRel);
				}

				//stop if enough points are consens
				size_t cutoff = mConsList1.size() * cConsLoopPercent / 100;
				if (numCons > cutoff) {
					break;

				} else {
					mConsList1.resize(cutoff);
				}
			}
			mConsList1.resize(numCons);
		}

		/*
		// STEP 2 histogram approach
		{
			auto maxFcn = [] (PointContext& pc1, PointContext& pc2) { return pc1.ptr->length < pc2.ptr->length; };
			double maxLen = std::max_element(mConsList2.begin(), mConsList2.end(), maxFcn)->ptr->length;

			Cluster clusterByLength(0.0, maxLen, 256);
			Cluster clusterByAngle(-std::numbers::pi, std::numbers::pi, 256);
			for (PointContext& pc : mConsList2) {
				pc.distance = pc.ptr->length;
				pc.angle = std::atan2(pc.ptr->v, pc.ptr->u);
				clusterByLength.add(pc, pc.distance);
				clusterByAngle.add(pc, pc.angle);
			}

			auto fcnLess = [&] (const PointContext& pc1, const PointContext& pc2) { return pc1.ptr->idx < pc2.ptr->idx; };
			clusterByLength.sortBySize();
			auto iter1 = clusterByLength.getTopPoints(cClusterFraction, mList1.begin());
			std::sort(mList1.begin(), iter1, fcnLess);
			//writeImage({ mList1.begin(), iter1 }, "f:/x1.bmp", mData.h, mData.w);
			//std::cout << clusterByLength << std::endl;

			clusterByAngle.sortBySize();
			auto iter2 = clusterByAngle.getTopPoints(cClusterFraction, mList2.begin());
			std::sort(mList2.begin(), iter2, fcnLess);
			//writeImage({ mList2.begin(), iter2 }, "f:/x2.bmp", mData.h, mData.w);

			auto iter3 = std::set_intersection(mList1.begin(), iter1, mList2.begin(), iter2, mList3.begin(), fcnLess);
			//writeImage({ mList3.begin(), iter3 }, "f:/x3.bmp", mData.h, mData.w);

			trf2.reset();
			std::span pointsCommon = { mList3.begin(), iter3 };
			if (pointsCommon.size() >= 2) trf2 = mAffineSolver->computeSimilar(pointsCommon);

			for (PointContext& pc : mConsList2) {
				PointResult& pr = *pc.ptr;
				auto [tx, ty] = mAffineSolver->transform(pr.x, pr.y);
				pc.distance = sqr(pr.x + pr.u - tx) + sqr(pr.y + pr.v - ty);
				pc.distanceRelative = pc.distance / pr.length;
			}
			auto fcnCheck = [&] (const PointContext& pc) { return pc.distance >= cConsensDistanceSqr || pc.distanceRelative >= cConsensDistRelative; };
			std::erase_if(mConsList2, fcnCheck);
			//writeImage({ mConsList2.begin(), mConsList2.end()}, "f:/x4.bmp", mData.h, mData.w);
		}
		*/
		

		//ResultDetailsWriter::write(results, "f:/results.txt");

		// STEP 3 dbscan
		{
			int minPts = 10;
			double eps = sqr(0.75);

			size_t siz = mConsList2.size();
			int m = std::max(mData.h, mData.w);
			double mm = m * m;
			auto f = [&] (double d) { return -0.125 * d * d / mm + 0.375 * d; };

			//uu and vv
			for (auto pc = mConsList2.begin(); pc != mConsList2.end(); pc++) {
				pc->uu = f(pc->ptr->u);
				pc->vv = f(pc->ptr->v);
			}

			//dbscan
			//cluster -1: unvisited, -2: noise, >= 0: cluster index 
			int clusterIdx = 0;
			std::list<PointContext*> region1;

			for (auto pc = mConsList2.begin(); pc != mConsList2.end(); pc++) {
				if (pc->cluster == -1) {
					region1.clear();

					//build neighborhood
					for (auto it = mConsList2.begin(); it != mConsList2.end(); it++) {
						double dist = sqr(pc->uu - it->uu) + sqr(pc->vv - it->vv);
						if (dist < eps) region1.push_back(&(*it));
					}

					if (region1.size() < minPts) {
						pc->cluster = -2;

					} else {
						for (PointContext* pcptr : region1) {
							if (pcptr->cluster < 0) {
								pcptr->cluster = clusterIdx;

								//build neighborhood
								std::vector<PointContext*> region2;
								for (auto it2 = mConsList2.begin(); it2 != mConsList2.end(); it2++) {
									double dist2 = sqr(pcptr->uu - it2->uu) + sqr(pcptr->vv - it2->vv);
									if (dist2 < eps) region2.push_back(&(*it2));
								}

								//expand region
								if (region2.size() >= minPts) {
									for (PointContext* p : region2) region1.push_back(p);
								}
							}
						}

						clusterIdx++;
					}
				}
			}

			//results
			for (int i = 0; i < clusterIdx; i++) {
				ptrdiff_t num = std::count_if(mConsList2.begin(), mConsList2.end(), [&] (const PointContext& pc) { return pc.cluster == i; });
				std::cout << "cluster " << i << ": " << num << std::endl;

				ImageBGR bgr(mData.h, mData.w);
				bgr.setValues({ 200, 230, 240 });

				for (const PointContext& p : mConsList2) {
					if (p.cluster == i) {
						PointResult& pr = *p.ptr;
						double px = pr.x + mData.w / 2.0;
						double py = pr.y + mData.h / 2.0;
						double x2 = px + pr.u;
						double y2 = py + pr.v;

						im::ColorBgr col = im::ColorBgr::BLUE;
						bgr.drawLine(px, py, x2, y2, col);
						bgr.drawMarker(x2, y2, col, 1.4);
					}
				}

				bgr.saveAsColorBMP("f:/clusters" + std::to_string(i) + ".bmp");
			}
		}

		// pick best result
		if (mConsList2.size() > mConsList1.size()) {
			std::swap(mConsList1, mConsList2);
			mBestTransform = trf2;

		} else {
			mBestTransform = trf1;
		}

		for (PointContext& pc : mConsList1) pc.ptr->isConsens = true;
		std::exit(-1);
	}
	//std::cout << "frame " << frameIndex << ", " << mBestTransform << std::endl;
}