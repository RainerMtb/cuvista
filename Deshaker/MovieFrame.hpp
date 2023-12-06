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

#include "MovieReader.hpp"
#include "CudaWriter.hpp"
#include "ThreadPool.h"
#include "ProgressDisplay.hpp"
#include "Diagnostics.hpp"
#include "Instrumentor.h"
#include "Diagnostics.hpp"
#include "UserInput.hpp"

#include "cuDeshaker.cuh"
#include "clMain.hpp"


//---------------------------------------------------------------------
//---------- MOVIE FRAME BASE CLASS -----------------------------------
//---------------------------------------------------------------------

class MovieFrame {

public:
	class DeshakerLoop {
	public:
		virtual void run(MovieFrame& mf, ProgressDisplay& progress, MovieReader& reader, MovieWriter& writer, UserInput& input) {}
	};

	class DeshakerLoopCombined : public DeshakerLoop {
	public:
		void run(MovieFrame& mf, ProgressDisplay& progress, MovieReader& reader, MovieWriter& writer, UserInput& input) override;
	};

	class DeshakerLoopFirst : public DeshakerLoop {
	public:
		void run(MovieFrame& mf, ProgressDisplay& progress, MovieReader& reader, MovieWriter& writer, UserInput& input) override;
	};

	class DeshakerLoopSecond : public DeshakerLoop {
	public:
		void run(MovieFrame& mf, ProgressDisplay& progress, MovieReader& reader, MovieWriter& writer, UserInput& input) override;
	};

	class DeshakerLoopClassic : public DeshakerLoop {
	public:
		void run(MovieFrame& mf, ProgressDisplay& progress, MovieReader& reader, MovieWriter& writer, UserInput& input) override;
	};

protected:
	const MainData& mData;
	Stats& mStatus;
	Trajectory mTrajectory;
	std::list<std::unique_ptr<DiagnosticItem>> diagsList;

public:
	FrameResult mFrameResult;
	ImageYuv inputFrame;
	ImageYuv bufferFrame;
	std::vector<PointResult> resultPointsOld;
	ThreadPool mPool;
	std::vector<PointResult> resultPoints;

	MovieFrame(const MovieFrame& other) = delete;
	MovieFrame(MovieFrame&& other) = delete;
	virtual ~MovieFrame();

	//get frame data from reader into frame object
	virtual void inputData(ImageYuv& frame) = 0;
	//set up image pyramid
	virtual void createPyramid() = 0;
	//start computation asynchronously for some part of a frame
	virtual void computeStart() = 0;
	//start computation asynchronously for second part and get results
	virtual void computeTerminate() = 0;
	//prepare data for output to writer
	virtual void outputData(const AffineTransform& trf, OutputContext outCtx) = 0;

	/*
	call transform calculation
	*/
	virtual const AffineTransform& computeTransform(std::vector<PointResult> resultPoints) final;

	/*
	* get transformed image as Mat<float> where YUV color planes are stacked vertically for debugging
	* warped output before unsharping
	*/
	virtual Mat<float> getTransformedOutput() const { return {}; }

	/*
	* get image pyramid as single Mat<float> where pyramids for Y, DX, DY are stacked vertically
	* useful for debugging
	*/
	virtual Mat<float> getPyramid(size_t idx) const { return {}; }

	/*
	* get input image as stored in frame buffers
	*/
	virtual ImageYuv getInput(int64_t index) const { return {}; }

	/*
	* get most recently read input frame to show progress
	*/
	virtual void getCurrentInputFrame(ImagePPM& image) {}

	/*
	* get most resently processed output frame, after warping, before unsharping
	*/
	virtual void getCurrentOutputFrame(ImagePPM& image) {}

	/*
	* read transforms from previous pass
	*/
	virtual std::map<int64_t, TransformValues> readTransforms() final;

	/*
	* run list of attached diagnostic methods
	*/
	virtual void runDiagnostics(int64_t frameIndex) final;

	/*
	* name MovieFrame used
	*/
	virtual std::string name() const { return "None"; }

protected:
	//only constructor for Frame class
	MovieFrame(MainData& data) :
		mData { data }, 
		mStatus { data.status },
		mFrameResult(data),
		inputFrame(data.h, data.w, data.cpupitch),
		bufferFrame(data.h, data.w, data.cpupitch),
		resultPointsOld(data.resultCount), 
		resultPoints(data.resultCount),
		mPool(data.cpuThreads)
	{
		//open and attach sinks for diagnostics
		if (data.pass != DeshakerPass::SECOND_PASS && data.trajectoryFile.empty() == false) {
			diagsList.push_back(std::make_unique<TransformsFile>(data.trajectoryFile, std::ios::out | std::ios::binary));
		}
		if (!data.resultsFile.empty()) {
			diagsList.push_back(std::make_unique<ResultDetailsFile>(data.resultsFile));
		}
		if (!data.resultImageFile.empty()) {
			diagsList.push_back(std::make_unique<ResultImage>(data, [&] (int64_t idx) { return getInput(idx); }));
		}
	}
};


//---------------------------------------------------------------------
//---------- DUMMY FRAME ----------------------------------------------
//---------------------------------------------------------------------

class DummyFrame : public MovieFrame {

private:
	std::vector<ImageYuv> frames;

public:
	DummyFrame(MainData& data) :
		MovieFrame(data), 
		frames(data.bufferCount, { data.h, data.w, data.w })
	{}

	void inputData(ImageYuv& frame) override;
	void createPyramid() override {}
	void computeStart() override {}
	void computeTerminate() override {}
	void outputData(const AffineTransform& trf, OutputContext outCtx) override;
	ImageYuv getInput(int64_t index) const override;
};


//---------------------------------------------------------------------
//---------- DEFAULT FRAME ----------------------------------------------
//---------------------------------------------------------------------

class DefaultFrame : public MovieFrame {
public:
	DefaultFrame(MainData& data) : MovieFrame(data) {}
	void inputData(ImageYuv& frame) override {}
	void createPyramid() override {}
	void computeStart() override {}
	void computeTerminate() override {}
	void outputData(const AffineTransform& trf, OutputContext outCtx) override {};
};