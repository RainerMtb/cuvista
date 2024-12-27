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

double FrameResult::sqr(double value) { 
	return value * value; 
}

FrameResult::FrameResult(MainData& data, ThreadPoolBase& threadPool) :
	mData { data } 
{
	if (data.hasAvx512()) {
		mAffineSolver = std::make_unique<AffineSolverAvx>(data.resultCount);
	} else {
		mAffineSolver = std::make_unique<AffineSolverFast>(threadPool, data.resultCount);
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

void FrameResult::computeTransform(std::vector<PointResult>& results, ThreadPoolBase& threadPool, int64_t frameIndex, SamplerPtr rng) {
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