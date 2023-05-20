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

const AffineTransform& FrameResult::computeTransform(const std::vector<PointResult>& results, const MainData& data, ThreadPool& threadPool, RNGbase* rng) {
	const ptrdiff_t cMinConsensPoints = 8;	     //min numbers of points for consensus set
	const int cConsLoopCount = 8;			     //max number of loops when searching for consensus set
	const int cConsLoopPercent = 95;		     //percentage of points for next loop 0..100
	const double cConsensDistance = 1.25;        //max offset for a point to be in the consensus set
	const double cConsensDistRelative = 0.2;     //max offset normalized by length
	auto sortAbs = [] (const PointResult& pr1, const PointResult& pr2) { return pr1.distance < pr2.distance; };
	auto sortRel = [] (const PointResult& pr1, const PointResult& pr2) { return pr1.distanceRelative < pr2.distanceRelative; };

	//ConsoleTimer ic("transform");
	mTransform.reset();
	mTransformsList.clear();
	mCountConsens = 0;

	//only valid points
	auto finiteEnd = std::copy_if(results.cbegin(), results.cend(), mFiniteResults.begin(), [] (const PointResult& res) { return res.isValid(); });
	ptrdiff_t siz = mCountFinite = std::distance(mFiniteResults.begin(), finiteEnd);

	if (siz > cMinConsensPoints) {
		//calculate average displacement distance and cut off 10% outsiders
		auto iterEnd = finiteEnd;
		double averageLength = 0.0;
		for (auto it = mFiniteResults.begin(); it != iterEnd; it++) {
			it->length = std::sqrt(sqr(it->u) + sqr(it->v));
			averageLength += it->length;
		}
		averageLength /= mCountFinite;
		for (auto it = mFiniteResults.begin(); it != iterEnd; it++) {
			it->distance = std::abs(it->length - averageLength);
		}
		std::sort(mFiniteResults.begin(), iterEnd, sortAbs);
		siz = mCountFinite * 90 / 100;

		ptrdiff_t numCons = 0;
		for (int i = 0; i < cConsLoopCount; i++) {
			iterEnd = mFiniteResults.begin();
			std::advance(iterEnd, siz);

			//average length of current points
			averageLength = 0.0;
			for (auto it = mFiniteResults.begin(); it != iterEnd; it++) {
				it->length = std::sqrt(sqr(it->u) + sqr(it->v));
				averageLength += it->length;
			}
			averageLength /= mCountFinite;

			//transform for selected points
			mTransform.computeSimilarDirect(mFiniteResults.begin(), siz, threadPool);
			mTransformsList.push_back(mTransform); //save for debugging

			int numConsAbsolute = 0;
			int numConsRelative = 0;
			//calculate error distance based on transform
			for (auto it = mFiniteResults.begin(); it != iterEnd; it++) {
				auto [tx, ty] = mTransform.transform(it->x, it->y);
				it->distance = std::sqrt(sqr(it->x + it->u - tx) + sqr(it->y + it->v - ty));
				it->distanceRelative = it->distance / it->length;

				if (it->distance < cConsensDistance)
					numConsAbsolute++;
				else if (it->distanceRelative < cConsensDistRelative)
					numConsRelative++;
			}

			//only then rely on relatives when there are many more than absolute ones
			if (numConsAbsolute > 40 || numConsAbsolute * 10 > numConsRelative) {
				numCons = numConsAbsolute;
				std::sort(mFiniteResults.begin(), iterEnd, sortAbs);

			} else {
				//std::cout << data.status.frameInputIndex << " ABS " << numConsAbsolute << " REL " << numConsRelative << std::endl;
				numCons = numConsRelative;
				std::sort(mFiniteResults.begin(), iterEnd, sortRel);
			}

			//stop if enough points are consens
			if (numCons == siz) {
				break;

			} else {
				siz = siz * cConsLoopPercent / 100;
			}

			//debug output
			//std::cout << data.status.frameInputIndex << " " << i << " " << averageLength << " " <<
			//	numConsAbsolute << " " << numConsRelative << " " << siz << " " << numCons << " " << 
			//	(numConsAbsolute > numConsRelative ? "ABS" : "REL") << std::endl;
		}
		mCountConsens = numCons;
		//mTransform.computeSimilarDirect(mFiniteResults.begin(), numCons, threadPool);
		//mTransformsList.push_back(mTransform);
	}

	return mTransform;
}
