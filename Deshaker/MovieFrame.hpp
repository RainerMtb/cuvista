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
#include "UserInput.hpp"

#include "cuDeshaker.cuh"
#include "clMain.hpp"

using Writers = std::vector<std::unique_ptr<SecondaryWriter>>;


//---------------------------------------------------------------------
//---------- MOVIE FRAME BASE CLASS -----------------------------------
//---------------------------------------------------------------------

class MovieFrame {

protected:
	const MainData& mData;
	Stats& mStatus;
	Trajectory mTrajectory;

public:

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
	* run the stabilizing loop
	*/
	virtual void runLoop(DeshakerPass pass, ProgressDisplay& progress, MovieReader& reader, MovieWriter& writer, UserInput& input, Writers& secondaryWriters);

	/*
	* call transform calculation
	*/
	virtual const AffineTransform& computeTransform(std::vector<PointResult> resultPoints) final;

	/*
	* get transformed image as Mat<float> where YUV color planes are stacked vertically
	* image is warped output before unsharping, useful for debugging
	*/
	virtual Mat<float> getTransformedOutput() const { return {}; }

	/*
	* get image pyramid as single Mat<float> where images are stacked vertically from large to small
	* useful for debugging
	*/
	virtual Mat<float> getPyramid(size_t idx) const { return {}; }

	/*
	* get input image as stored in frame buffers
	*/
	virtual ImageYuv getInput(int64_t index) const { return {}; }

	/*
	* get most recently read input frame
	* use to show progress
	*/
	virtual void getCurrentInputFrame(ImagePPM& image) {}

	/*
	* get most resently processed output frame
	* image is warped output before unsharping
	* use to show progress
	*/
	virtual void getTransformedOutput(ImagePPM& image) {}

	/*
	* read transforms from previous pass
	*/
	virtual std::map<int64_t, TransformValues> readTransforms() final;

	/*
	* run list of attached diagnostic methods
	*/
	virtual void outputSecondary(Writers& writers, int64_t frameIndex) final;

	/*
	* name MovieFrame used
	*/
	virtual std::string name() const { return "None"; }

	FrameResult mFrameResult;
	ImageYuv inputFrame;
	ImageYuv bufferFrame;
	std::vector<PointResult> resultPointsOld;
	ThreadPool mPool;
	std::vector<PointResult> resultPoints;

	MovieFrame(const MovieFrame& other) = delete;
	MovieFrame(MovieFrame&& other) = delete;
	virtual ~MovieFrame();

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
	{}

private:
	void runLoopCombined(ProgressDisplay& progress, MovieReader& reader, MovieWriter& writer, UserInput& input, Writers& secondaryWriters);
	void runLoopFirst(ProgressDisplay& progress, MovieReader& reader, MovieWriter& writer, UserInput& input, Writers& secondaryWriters);
	void runLoopSecond(ProgressDisplay& progress, MovieReader& reader, MovieWriter& writer, UserInput& input, Writers& secondaryWriters);
	void runLoopConsecutive(ProgressDisplay& progress, MovieReader& reader, MovieWriter& writer, UserInput& input, Writers& secondaryWriters);
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