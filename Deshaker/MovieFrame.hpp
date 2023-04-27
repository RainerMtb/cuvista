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
#include "MovieWriter.hpp"
#include "ThreadPool.h"
#include "ProgressDisplay.hpp"
#include "Diagnostics.hpp"
#include "Instrumentor.h"
#include "Diagnostics.hpp"
#include "UserInput.hpp"

#include "cuDeshaker.cuh"


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
	FrameResult mFrameResult;
	Trajectory mTrajectory;
	std::list<std::unique_ptr<DiagnosticItem>> diagsList;

public:
	ThreadPool mPool;
	ImageYuv inputFrame;
	ImageYuv bufferFrame;
	std::vector<PointResult> resultPointsOld;
	std::vector<PointResult> resultPoints;

	MovieFrame(const MovieFrame& other) = delete;
	MovieFrame(MovieFrame&& other) = delete;
	virtual ~MovieFrame();

	//get frame data from reader into frame object
	virtual void inputData(ImageYuv& frame) = 0;
	//set up image pyramid
	virtual void createPyramid() = 0;
	//start computation asynchronously
	virtual void computeStart() = 0;
	//return displacement for each point
	virtual void computeTerminate() = 0;
	//prepare data for output to writer
	virtual void outputData(const AffineTransform& trf, OutputContext outCtx) = 0;

	/*
	* read transforms from previous pass
	*/
	std::map<int64_t, TransformValues> readTransforms();

	/*
	* run list of attached diagnostic methods
	*/
	void runDiagnostics(int64_t frameIndex);

	/*
	* get transformed image as Mat<float> where YUV color planes are stacked vertically
	* useful for debugging
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
	virtual bool getCurrentInputFrame(ImagePPM& image) { return false; }

	/*
	* get most resently processed output frame, after warping, before filtering and blending
	*/
	virtual bool getCurrentOutputFrame(ImagePPM& image) { return false; }

protected:
	//only constructor for Frame class
	MovieFrame(MainData& data) :
		mData { data }, 
		mStatus { data.status },
		mPool(data.cpuThreads), 
		mFrameResult(data),
		inputFrame(data.h, data.w, data.pitch),
		bufferFrame(data.h, data.w, data.pitch),
		resultPointsOld(data.resultCount), 
		resultPoints(data.resultCount)
	{
		//open and attach sinks for diagnostics
		if (data.pass == DeshakerPass::FIRST_PASS && data.trajectoryFile.empty() == false) {
			diagsList.push_back(std::make_unique<TransformsFile>(data.trajectoryFile, std::ios::out | std::ios::binary));
		}
		if (!data.resultsFile.empty()) {
			diagsList.push_back(std::make_unique<ResultDetailsFile>(data.resultsFile, data.fileDelimiter));
		}
		if (!data.resultImageFile.empty()) {
			diagsList.push_back(std::make_unique<ResultImage>(data, [&] (int64_t idx) { return getInput(idx); }));
		}
	}
};


//---------------------------------------------------------------------
//---------- GPU FRAME ------------------------------------------------
//---------------------------------------------------------------------


class GpuFrame : public MovieFrame {

public:
	GpuFrame(MainData& data) : MovieFrame(data) {
		cudaInit(data, inputFrame);
	}

	~GpuFrame();

	void inputData(ImageYuv& frame) override {
		cudaReadFrame(mStatus.frameInputIndex, mData, frame);
	}

	void createPyramid() override {
		cudaCreatePyramid(mStatus.frameInputIndex, mData);
	}

	void computeStart() override {
		cudaComputeStart(mStatus.frameInputIndex, mData);
	}

	void computeTerminate() override {
		cudaComputeTerminate(mData, resultPoints);
	}

	void outputData(const AffineTransform& trf, OutputContext outCtx) override {
		cudaOutput(mStatus.frameWriteIndex, mData, outCtx, trf.toCuAffine());
	}

	Mat<float> getTransformedOutput() const override {
		Mat<float> warped = Mat<float>::allocate(3LL * mData.h, mData.w);
		cudaGetTransformedOutput(warped.data(), mData);
		return warped;
	}

	Mat<float> getPyramid(size_t idx) const override {
		Mat<float> out = Mat<float>::allocate(mData.pyramidRows * 3LL, mData.w);
		cudaGetPyramid(out.data(), idx, mData);
		return out;
	}

	ImageYuv getInput(int64_t index) const override {
		return cudaGetInput(index, mData);
	}

	bool getCurrentInputFrame(ImagePPM& image) override {
		bool state = mStatus.frameReadIndex > 0;
		if (state) {
			int idx = (mStatus.frameReadIndex - 1) % mData.bufferCount;
			cudaGetCurrentInputFrame(image, mData, idx);
		}
		return state;
	}

	bool getCurrentOutputFrame(ImagePPM& image) override {
		bool state = mStatus.frameWriteIndex > 0;
		if (state) {
			cudaGetCurrentOutputFrame(image, mData);
		}
		return state;
	}
};


//---------------------------------------------------------------------
//---------- CPU FRAME ------------------------------------------------
//---------------------------------------------------------------------

class CpuFrame : public MovieFrame {

public:
	CpuFrame(MainData& data);

	void inputData(ImageYuv& frame) override;
	void createPyramid() override;
	void computeStart() override {}
	void computeTerminate() override;
	void outputData(const AffineTransform& trf, OutputContext outCtx) override;
	Mat<float> getTransformedOutput() const override;
	Mat<float> getPyramid(size_t idx) const override;
	ImageYuv getInput(int64_t index) const override;
	bool getCurrentInputFrame(ImagePPM& image) override;
	bool getCurrentOutputFrame(ImagePPM& image) override;

protected:

	class CpuFrameItem {

	public:
		int64_t frameIndex = -1;
		std::vector<Mat<float>> Y, DX, DY;

		CpuFrameItem(MainData& data);
	};

	//frame input buffer, number of frames = frameBufferCount
	std::vector<ImageYuv> YUV;

	//holds image pyramids
	std::vector<CpuFrameItem> pyr;

	//buffers the last output frame, 3 mats, to be used to blend background of next frame
	std::vector<Mat<float>> prevOut;

	//buffer for generating output from input yuv and transformation
	std::vector<Mat<float>> buffer;
	Mat<float> yuv, filterBuffer, filterResult;
};


//---------------------------------------------------------------------
//---------- DUMMY FRAME ----------------------------------------------
//---------------------------------------------------------------------

class DummyFrame : public MovieFrame {

private:
	std::vector<ImageYuv> frames;
	std::vector<unsigned char> nv12;

public:
	DummyFrame(MainData& data, size_t nv12pitch) :
		MovieFrame(data), 
		frames(data.bufferCount, { data.h, data.w, data.pitch }), 
		nv12(nv12pitch * data.h * 3 / 2)
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
	void inputData(ImageYuv& frame) {}
	void createPyramid() {}
	void computeStart() {}
	void computeTerminate() {}
	void outputData(const AffineTransform& trf, OutputContext outCtx) {};
};