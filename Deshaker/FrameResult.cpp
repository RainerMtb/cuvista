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
	//ConsoleTimer ic("transform");
	mTransform.reset();
	mTransformsList.clear();
	mCountConsens = 0;

	//only valid points
	auto finiteEnd = std::copy_if(results.cbegin(), results.cend(), mFiniteResults.begin(), [] (const PointResult& res) { return res.isValid(); });
	ptrdiff_t siz = mCountFinite = std::distance(mFiniteResults.begin(), finiteEnd);

	if (siz > data.cMinConsensPoints) {
		//cut off longest 10% of displacement
		auto iterEnd = finiteEnd;
		for (auto it = mFiniteResults.begin(); it != iterEnd; it++) {
			it->distance = sqr(it->u) + sqr(it->v);
		}
		std::sort(mFiniteResults.begin(), iterEnd);
		siz = mCountFinite * 90 / 100;

		ptrdiff_t numCons = 0;
		for (int i = 0; i < data.cConsLoopCount; i++) {
			//transform for selected points
			mTransform.computeSimilarDirect(mFiniteResults.begin(), siz, threadPool);
			mTransformsList.push_back(mTransform); //save for eventual later analysis

			iterEnd = mFiniteResults.begin();
			std::advance(iterEnd, siz);
			//calculate distance based on transform
			for (auto it = mFiniteResults.begin(); it != iterEnd; it++) {
				auto [tx, ty] = mTransform.transform(it->x, it->y);
				it->distance = sqr(it->x + it->u - tx) + sqr(it->y + it->v - ty);
			}

			//sort by distance
			std::sort(mFiniteResults.begin(), iterEnd);

			//check how many PointResults are consens
			numCons = 0;
			while (mFiniteResults[numCons].distance < data.cConsensDistance && numCons < siz) {
				numCons++;
			}

			//stop if enough points are consens
			if (numCons == siz) {
				break;

			} else {
				siz = siz * data.cConsLoopPercent / 100;
			}
		}
		mCountConsens = siz;
	}
	return mTransform;
}
