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

#pragma once

#include <span>
#include "AffineTransform.hpp"
#include "MainData.hpp"
#include "ThreadPoolBase.h"


//compute and hold transform for one video frame
class FrameResult {

	using SamplerPtr = std::shared_ptr<SamplerBase<PointContext>>;

public:
	struct ClusterSize {
		int index, siz;

		friend std::ostream& operator << (std::ostream& out, const ClusterSize& cs);
	};

	struct DebugData {
		bool rundbscan = false;
		std::vector<PointContext> classic;
		std::vector<PointContext> dbscan;
		std::vector<ClusterSize> clusterSizes;
	};

	inline static DebugData debugData;
	inline static bool storeDebugData = false;

	//construct lists and solver class
	FrameResult(MainData& data, ThreadPoolBase& threadPool);

	//compute resulting transformation for this frame
	const AffineTransform& computeTransform(std::span<PointResult> results, int64_t frameIndex);

	//get the last computed treansform
	const AffineTransform& getTransform() const;

	//reset internal state of this class
	void reset();

private:
	const MainData& mData;
	ThreadPoolBase& mPool;
	std::unique_ptr<AffineSolver> mAffineSolver;
	AffineTransform mBestTransform;
	std::vector<PointContext> mConsList;
	
	std::vector<PointContext> mPointList;
	std::vector<PointContext> mBestCluster;
	std::vector<PointContext*> mWork;
	std::vector<ClusterSize> mClusterSizes;

	struct {
		int consLoopCount = 8;                      //max number of loops when searching for consensus set
		int consLoopPercent = 95;                   //percentage of points for next loop 0..100
		double consDistRelative = 0.2;              //max offset normalized by length
		double consDistanceSqr = util::sqr(1.25);   //max offset for a point to be in the consensus set

		size_t minConsPoints = 8;      //min numbers of points for consensus set
		double minDbScanRel = 0.08;    //when to try dbscan
		int minDbScanAbs = 80;         //when to start dbscan
		int minPts = 20;               //min cluster size
		int finalSizePercent = 75;     //reduction of best cluster

		double eps = -1.0;    //eps value for dbscan
		double f1 = -1.0;     //factor for angle delta
		double f2 = -1.0;     //factor for vector length
	} params;

	AffineTransform computeClassic(size_t numValid, int64_t frameIndex);
	void computeDbScan(int64_t frameIndex);
	bool clusterDistance(const PointContext& pc1, const PointContext& pc2) const;

	bool checkSizes() const;
	void writeVideo(std::span<PointContext> res, int64_t frameIndex, const std::string& title) const;
};
