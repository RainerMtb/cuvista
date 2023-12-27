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

#include "CudaWriter.hpp"
#include "ThreadPool.h"
#include "UserInput.hpp"
#include "MainData.hpp"
#include "Trajectory.hpp"
#include "cuDeshaker.cuh"
#include "clMain.hpp"


//---------------------------------------------------------------------
//---------- MOVIE FRAME BASE CLASS -----------------------------------
//---------------------------------------------------------------------

class ProgressDisplay;

class MovieFrame {

protected:
	const MainData& mData;
	Trajectory mTrajectory;

public:
	MovieReader& mReader;
	MovieWriter& mWriter;

	//get frame data from reader into frame object
	virtual void inputData() = 0;
	//set up image pyramid
	virtual void createPyramid(int64_t frameIndex) = 0;
	//start computation asynchronously for some part of a frame
	virtual void computeStart(int64_t frameIndex) = 0;
	//start computation asynchronously for second part and get results
	virtual void computeTerminate(int64_t frameIndex) = 0;
	//prepare data for output to writer
	virtual void outputData(const AffineTransform& trf, OutputContext outCtx) = 0;

	/*
	* run the stabilizing loop
	*/
	virtual void runLoop(DeshakerPass pass, ProgressDisplay& progress, UserInput& input, AuxWriters& secondaryWriters);

	/*
	* call transform calculation
	*/
	virtual void computeTransform(int64_t frameIndex) final;

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
	virtual void getInputFrame(int64_t frameIndex, ImagePPM& image) {}

	/*
	* get most resently processed output frame
	* image is warped output before unsharping
	* use to show progress
	*/
	virtual void getTransformedOutput(int64_t frameIndex, ImagePPM& image) {}

	/*
	* read transforms from previous pass
	*/
	virtual std::map<int64_t, TransformValues> readTransforms() final;

	/*
	* name MovieFrame used
	*/
	virtual std::string className() const { return "None"; }

	ThreadPool mPool;
	FrameResult mFrameResult;
	ImageYuv mBufferFrame;
	std::vector<PointResult> mResultPoints;

	MovieFrame(const MovieFrame& other) = delete;
	MovieFrame(MovieFrame&& other) = delete;
	virtual ~MovieFrame();

protected:
	//only constructor for Frame class
	MovieFrame(MainData& data, MovieReader& reader, MovieWriter& writer) :
		mData { data },
		mReader { reader },
		mWriter { writer },
		mPool(data.cpuThreads),
		mFrameResult(data, mPool),
		mBufferFrame(data.h, data.w, data.cpupitch),
		mResultPoints(data.resultCount) {}

private:
	void read();
	void write();

	void runLoopCombined(ProgressDisplay& progress, UserInput& input, AuxWriters& auxWriters);
	void runLoopFirst(ProgressDisplay& progress, UserInput& input, AuxWriters& auxWriters);
	void runLoopSecond(ProgressDisplay& progress, UserInput& input, AuxWriters& auxWriters);
	void runLoopConsecutive(ProgressDisplay& progress, UserInput& input, AuxWriters& auxWriters);

	void loopInit(ProgressDisplay& progress, const std::string& message = "");
	void loopTerminate(ProgressDisplay& progress, UserInput& input, AuxWriters& auxWriters);
};


//---------------------------------------------------------------------
//---------- DUMMY FRAME ----------------------------------------------
//---------------------------------------------------------------------

class DummyFrame : public MovieFrame {

private:
	std::vector<ImageYuv> frames;

public:
	DummyFrame(MainData& data, MovieReader& reader, MovieWriter& writer) :
		MovieFrame(data, reader, writer),
		frames(data.bufferCount, { data.h, data.w, data.w }) {}

	void inputData() override;
	void createPyramid(int64_t frameIndex) override {}
	void computeStart(int64_t frameIndex) override {}
	void computeTerminate(int64_t frameIndex) override {}
	void outputData(const AffineTransform& trf, OutputContext outCtx) override;
	ImageYuv getInput(int64_t index) const override;
};


//---------------------------------------------------------------------
//---------- DEFAULT FRAME ----------------------------------------------
//---------------------------------------------------------------------

class DefaultFrame : public MovieFrame {
public:
	DefaultFrame(MainData& data, MovieReader& reader, MovieWriter& writer) : 
		MovieFrame(data, reader, writer) {}
	void inputData() override {}
	void createPyramid(int64_t frameIndex) override {}
	void computeStart(int64_t frameIndex) override {}
	void computeTerminate(int64_t frameIndex) override {}
	void outputData(const AffineTransform& trf, OutputContext outCtx) override {};
};