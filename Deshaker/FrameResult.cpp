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

class Bucket {

private:
	size_t numel = 0;
	double minVal, maxVal, bucketSize;
	std::vector<std::vector<PointContext>> buckets;

public:
	Bucket(double minVal, double maxVal, int bucketCount) {
		this->minVal = minVal;
		this->maxVal = maxVal;
		this->bucketSize = (maxVal - minVal) / bucketCount;
		this->buckets.resize(bucketCount);
	}

	void add(const PointContext& pc, double value) {
		double bucketIndex = (value - minVal) / bucketSize;
		int idx = (int) bucketIndex;
		if (idx == bucketIndex && bucketIndex > 0) idx--;
		buckets[idx].push_back(pc);
		numel++;
	}

	void sortBySize() {
		std::sort(buckets.begin(), buckets.end(), [] (std::vector<PointContext>& v1, std::vector<PointContext>& v2) { return v1.size() > v2.size(); });
	}

	std::vector<PointContext> getTopPoints(double percentage) {
		size_t requiredCount = (size_t) (numel * percentage);
		std::vector<PointContext> out;
		for (auto it = buckets.begin(); it != buckets.end() && out.size() < requiredCount; it++) {
			size_t n = std::min(it->size(), requiredCount - out.size());
			std::copy_n(it->begin(), n, std::back_inserter(out));
		}
		return out;
	}
};

double FrameResult::sqr(double value) { 
	return value * value; 
}

void FrameResult::drawTransforms(std::span<PointContext> pc, const std::string& filename) {
	ImageBGR bgr(mData.h, mData.w);
	bgr.setValues({ 200, 230, 240 });

	for (const PointContext& p : pc) {
		PointResult& pr = *p.ptr;
		double px = pr.x + mData.w / 2.0;
		double py = pr.y + mData.h / 2.0;
		double x2 = px + pr.u;
		double y2 = py + pr.v;

		im::ColorBgr col = im::ColorBgr::BLUE;
		bgr.drawLine(px, py, x2, y2, col);
		bgr.drawDot(x2, y2, 1.5, 1.5, col);
	}

	bgr.saveAsColorBMP(filename);
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

void FrameResult::computeTransform(std::vector<PointResult>& results, ThreadPoolBase& threadPool, int64_t frameIndex, SamplerPtr sampler) {
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
	mConsList.clear();

	//only valid points
	for (PointResult& pr : results) {
		pr.isConsens = false;
		if (pr.isValid()) {
			mConsList.emplace_back(pr);
		}
	}
	size_t numValid = mConsList.size();

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
		mPointsList = mConsList;

		// STEP 1
		size_t numCons = 0;
		for (int i = 0; i < cConsLoopCount; i++) {
			//average length of current points
			averageLength = 0.0;
			for (auto it = mConsList.begin(); it != mConsList.end(); it++) {
				averageLength += it->ptr->length;
			}
			averageLength /= mConsList.size();

			//transform for selected points
			mBestTransform = mAffineSolver->computeSimilar(mConsList);

			size_t numConsAbsolute = 0;
			size_t numConsRelative = 0;
			//calculate error distance based on transform
			for (auto it = mConsList.begin(); it != mConsList.end(); it++) {
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

		// STEP 2 try ransac on second list
		/*
		if (100.0 * numCons / numValid < 10.0) {
			std::vector<PointContext> samples(3), bestList, consList;
			int consCount = 0, bestCount = 0;

			for (int i = 0; i < cRansacIterations; i++) {
				consList.clear();
				consCount = 0;

				sampler->sample(mConsListSecond, samples);
				AffineTransform trf = mAffineSolver->computeSimilar(samples);
				for (const PointContext& pc : mConsListSecond) {
					const PointResult& pr = *pc.ptr;
					auto [tx, ty] = trf.transform(pr.x, pr.y);
					double dist = sqr(pr.x + pr.u - tx) + sqr(pr.y + pr.v - ty);
					if (dist < cConsensDistanceSqr * 2) {
						consList.push_back(pc);
						consCount++;
					}
				}
				if (consCount > bestCount) {
					bestCount = consCount;
					std::swap(bestList, consList);
				}
			}

			if (bestCount > numCons) {
				mBestTransform = mAffineSolver->computeSimilar(bestList);
				std::swap(bestList, mConsList);
			}
		}
		*/

		//histogram
		auto maxFcn = [] (PointContext& pc1, PointContext& pc2) { return pc1.ptr->length < pc2.ptr->length; };
		double maxLen = std::max_element(mPointsList.begin(), mPointsList.end(), maxFcn)->ptr->length;
		
		Bucket bucketByLength(0.0, maxLen, 256);
		Bucket bucketByAngle(-std::numbers::pi, std::numbers::pi, 256);
		for (PointContext& pc : mPointsList) {
			pc.distance = pc.ptr->length;
			pc.angle = std::atan2(pc.ptr->v, pc.ptr->u);
			bucketByLength.add(pc, pc.distance);
			bucketByAngle.add(pc, pc.angle);
		}
		bucketByLength.sortBySize();
		bucketByAngle.sortBySize();
		std::vector<PointContext> topByLengthCount = bucketByLength.getTopPoints(0.2);
		std::vector<PointContext> topByAngleCount = bucketByAngle.getTopPoints(0.2);
		drawTransforms(topByLengthCount, "f:/x1.bmp");
		drawTransforms(topByAngleCount, "f:/x2.bmp");

		for (PointContext& pc : mConsList) pc.ptr->isConsens = true;
	}
}