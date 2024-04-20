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

#include "FrameResult.hpp"
#include "Util.hpp"

FrameResult::FrameResult(MainData& data, ThreadPool& threadPool) :
	mData { data } 
{
	if (data.deviceInfoAvx.hasAvx512()) {
		mAffineSolver = std::make_unique<AffineSolverAvx>(data.resultCount);
	} else {
		mAffineSolver = std::make_unique<AffineSolverFast>(threadPool, data.resultCount);
	}
}

void FrameResult::computeTransform(std::vector<PointResult>& results, ThreadPool& threadPool, int64_t frameIndex, RNGbase* rng) {
	computeExperimental(results, threadPool, frameIndex, rng);
}

void FrameResult::computeExperimental(std::vector<PointResult>& results, ThreadPool& threadPool, int64_t frameIndex, RNGbase* rng) {
	const ptrdiff_t cMinConsensPoints = 8;	     //min numbers of points for consensus set
	const double cRetryPercentage = 8.0;         //run loop as long as percentage of valid points is not reached
	const int cConsLoopCount = 8;			     //max number of loops when searching for consensus set
	const int cCutoffPercent = 90;               //reduction of points per loop 0..100
	const std::vector<double> cConsensRadius = { 1.25, 2.5, 5.0 };     //max offset for a point to be in the consensus set

	//util::ConsoleTimer ic("transform");
	mAffineSolver->reset();
	mAffineSolver->frameIndex = frameIndex;
	mConsList.clear();

	//only valid points
	for (PointResult& pr : results) {
		pr.isConsens = false;
		if (pr.isValid()) {
			mConsList.emplace_back(pr);
		}
	}
	//std::cout << "frame " << frameIndex << ", finite " << siz << std::endl;

	if (mConsList.size() > cMinConsensPoints) {
		//calculate vector parameters once
		double averageLength = 0.0;
		for (PointContext& pc : mConsList) {
			pc.ptr->length = std::sqrt(sqr(pc.ptr->u) + sqr(pc.ptr->v));
			pc.ptr->heading = std::atan2(pc.ptr->v, pc.ptr->u) * 180.0 / std::numbers::pi;
			pc.ptr->stretch = 5.0 - 2.0 / std::sqrt(0.25 + 0.125 * pc.ptr->length); //length factor of ellipse
			pc.ptr->isConsens = false;
			averageLength += pc.ptr->length;
		}

		averageLength /= mConsList.size();
		for (PointContext& pc : mConsList) {
			//set delta value to deviation from average length
			pc.confidence = std::abs(pc.ptr->length - averageLength);
		}
		auto sortFun = [] (const PointContext& p1, const PointContext& p2) { return p1.delta < p2.delta; };
		std::sort(mConsList.begin(), mConsList.end(), sortFun);
		mConsList.resize(mConsList.size() * cCutoffPercent / 100);

		//try to find consens with smallest error ellipse, increasing area when no appropriate consens can be found
		bool continueLoop = true;
		ptrdiff_t numCons = 0;
		for (size_t k = 0; k < cConsensRadius.size() && continueLoop; k++) {
			double radius = cConsensRadius[k];
			ptrdiff_t siz = mConsList.size();
			numCons = 0;

			//try to find consensus with increasingly smaller point set
			auto limitFun = [] (const PointContext& p) { return p.delta < p.distanceEllipse; };
			for (int i = 0; i < cConsLoopCount && numCons < siz; i++) {
				std::span<PointContext> span = { mConsList.begin(), size_t(siz) };
				//similar transform based on current set
				mBestTransform = mAffineSolver->computeSimilar(span);
				//check consensus set
				computePointContext(span, mBestTransform, radius);
				std::sort(span.begin(), span.end(), sortFun);
				numCons = std::count_if(span.begin(), span.end(), limitFun);
				siz = siz * cCutoffPercent / 100;
				//std::cout << "frame " << frameIndex << ", loop " << k << "-" << i << ", numCons " << numCons << std::endl;
			}
			continueLoop = 100.0 * numCons / mConsList.size() < cRetryPercentage;
		}

		if (!continueLoop) {
			mConsList.resize(numCons);

		} else {
			//when no good result was found so far then try ransac
			const int cLoopCount = 100;
			const double cAngleLimit = 10.0;
			const double cMinLength = 10.0;
			const size_t cMinCons = 6;
			auto fcn = [&] (const PointContext& pc) { return pc.deltaAngle < cAngleLimit && pc.ptr->length > cMinLength; };

			mBestTransform.reset();
			numCons = 0;
			double radius = cConsensRadius.back();
			std::vector<PointContext> samples(2);
			std::map<size_t, std::pair<AffineTransform, std::vector<PointContext>>, std::greater<size_t>> bestMap;

			//test multiple samples and build map of candidates
			for (int k = 0; k < cLoopCount; k++) {
				//get random points and compute affine transform based on those points
				std::sample(mConsList.begin(), mConsList.end(), samples.begin(), samples.size(), *rng);
				PointResult s1 = *samples[0].ptr;
				PointResult s2 = *samples[1].ptr;
				const AffineTransform& trf = mAffineSolver->computeSimilarDirect(s1, s2);

				//build consensus set, check with first transform
				std::vector<PointContext> cons;
				computePointContext(mConsList, trf, radius);
				std::copy_if(mConsList.begin(), mConsList.end(), std::back_inserter(cons), fcn);
				//std::cout << k << " " << trf << " // " << cons.size() << " points" << std::endl;
				//for (auto& pr : cons) { std::cout << "u=" << pr.ptr->u << " v=" << pr.ptr->v << " // "; } std::cout << std::endl;

				//build second consensus set with refined transform
				if (cons.size() > cMinCons) {
					AffineTransform trf2 = mAffineSolver->computeSimilar(cons);
					computePointContext(mConsList, trf2, radius);
					cons.clear();
					std::copy_if(mConsList.begin(), mConsList.end(), std::back_inserter(cons), fcn);
					bestMap.emplace(cons.size(), std::make_pair(trf2, cons));
					//std::cout << k << " " << trf << " // " << cons.size() << " points" << std::endl;
				}
			}

			if (!bestMap.empty()) {
				auto it = bestMap.begin();
				mBestTransform = it->second.first;
				mConsList = it->second.second;
			}
			//std::cout << "fr " << frameIndex << " cons " << mConsList.size() << std::endl;
		}

		//mark consens points
		for (PointContext& p : mConsList) p.ptr->isConsens = true;
	}
}

void FrameResult::computePointContext(std::span<PointContext> points, const AffineTransform& trf, double radius) {
	for (PointContext& pc : points) {
		auto tp = trf.transform(pc.ptr->x, pc.ptr->y);                      //apply transform to point
		pc.tx = tp.first; pc.ty = tp.second;                                //store transformed point

		double px = pc.ptr->x + pc.ptr->u, py = pc.ptr->y + pc.ptr->v;      //actual computed point
		double dx = pc.tx - px, dy = pc.ty - py;                            //error vector
		pc.delta = std::sqrt(sqr(dx) + sqr(dy));                            //error distance
		pc.deltaAngleCos = (pc.ptr->u * dx + pc.ptr->v * dy) / (pc.delta * pc.ptr->length); //cos(angle) = (a*b)/(|a|*|b|)
		pc.deltaAngle = std::acos(pc.deltaAngleCos) * 180.0 / std::numbers::pi;

		//distance to ellipse, ellipse in polar coordinates d^2 = b^2/(1-e^2*cos(theta)^2)
		double epsilon2 = 1.0 - 1.0 / sqr(pc.ptr->stretch);
		pc.distanceEllipse = std::sqrt(sqr(radius) / (1.0 - epsilon2 * sqr(pc.deltaAngleCos)));

		pc.confidence = pc.delta / pc.distanceEllipse;
	}
}

const AffineTransform& FrameResult::getTransform() const {
	return mBestTransform;
}

void FrameResult::reset() {
	mAffineSolver->reset();
	mAffineSolver->frameIndex = 0;
	mBestTransform.reset();
}

//old style stabilization
void FrameResult::compute(std::vector<PointResult>& results, ThreadPool& threadPool, int64_t frameIndex, RNGbase* rng) {
	const size_t cMinConsensPoints = 8;	     //min numbers of points for consensus set
	const int cConsLoopCount = 8;			     //max number of loops when searching for consensus set
	const int cConsLoopPercent = 95;		     //percentage of points for next loop 0..100
	const double cConsensDistance = 1.25;        //max offset for a point to be in the consensus set
	const double cConsensDistRelative = 0.2;     //max offset normalized by length
	auto sortAbs = [] (const PointContext& pc1, const PointContext& pc2) { return pc1.distance < pc2.distance; };
	auto sortRel = [] (const PointContext& pc1, const PointContext& pc2) { return pc1.distanceRelative < pc2.distanceRelative; };

	//util::ConsoleTimer ic("transform");
	mAffineSolver->reset();
	mAffineSolver->frameIndex = frameIndex;
	mConsList.clear();

	//only valid points
	for (PointResult& pr : results) {
		pr.isConsens = false;
		if (pr.isValid()) {
			mConsList.emplace_back(pr);
		}
	}
	size_t siz = mConsList.size();

	if (siz > cMinConsensPoints) {
		//calculate average displacement distance and cut off 10% outsiders
		double averageLength = 0.0;
		for (PointContext& pc : mConsList) {
			pc.ptr->length = std::sqrt(sqr(pc.ptr->u) + sqr(pc.ptr->v));
			averageLength += pc.ptr->length;
		}
		averageLength /= siz;
		for (PointContext& pc : mConsList) {
			//set delta value to deviation from average length
			pc.delta = std::abs(pc.ptr->length - averageLength);
		}
		std::sort(mConsList.begin(), mConsList.end(), sortAbs);
		mConsList.resize(siz * cConsLoopPercent / 100);

		size_t numCons = 0;
		for (int i = 0; i < cConsLoopCount; i++) {
			//average length of current points
			averageLength = 0.0;
			for (auto it = mConsList.begin(); it != mConsList.end(); it++) {
				averageLength += it->ptr->length;
			}
			averageLength /= siz;

			//transform for selected points
			mBestTransform = mAffineSolver->computeSimilar(mConsList);

			size_t numConsAbsolute = 0;
			size_t numConsRelative = 0;
			//calculate error distance based on transform
			for (auto it = mConsList.begin(); it != mConsList.end(); it++) {
				PointResult& pr = *it->ptr;
				auto [tx, ty] = mAffineSolver->transform(pr.x, pr.y);
				it->distance = std::sqrt(sqr(pr.x + pr.u - tx) + sqr(pr.y + pr.v - ty));
				it->distanceRelative = it->distance / pr.length;

				if (it->distance < cConsensDistance)
					numConsAbsolute++;
				else if (it->distanceRelative < cConsensDistRelative)
					numConsRelative++;
			}

			//only then rely on relatives when there are many more than absolute ones
			if (numConsAbsolute > 40 || numConsAbsolute * 10 > numConsRelative) {
				numCons = numConsAbsolute;
				std::sort(mConsList.begin(), mConsList.end(), sortAbs);

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
		for (PointContext& pc : mConsList) pc.ptr->isConsens = true;
	}
}