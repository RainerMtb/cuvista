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
	virtual void computePartOne() = 0;
	//start computation asynchronously for second part
	virtual void computePartTwo() = 0;
	//return displacement for each point
	virtual void computeTerminate() = 0;
	//prepare data for output to writer
	virtual void outputData(const AffineTransform& trf, OutputContext outCtx) = 0;

	/*
	call transform calculation
	*/
	const AffineTransform& computeTransform(std::vector<PointResult> resultPoints);

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
	std::map<int64_t, TransformValues> readTransforms();

	/*
	* run list of attached diagnostic methods
	*/
	void runDiagnostics(int64_t frameIndex);

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
//---------- CUDA FRAME -----------------------------------------------
//---------------------------------------------------------------------


class CudaFrame : public MovieFrame {

public:
	CudaFrame(MainData& data) : MovieFrame(data) {
		DeviceInfo* dev = data.deviceList[data.deviceSelected];
		size_t devIdx = dev->targetIndex;
		const cudaDeviceProp& prop = data.deviceListCuda[devIdx].props;
		cudaInit(data, (int) devIdx, prop, inputFrame);
	}

	~CudaFrame() {
		//retrieve debug data from device
		DebugData data = cudaShutdown(mData);
		std::vector<double>& debugData = data.debugData;
		size_t siz = (size_t) debugData[0];
		double* ptr = debugData.data() + 1;
		double* ptrEnd = debugData.data() + siz + 1;
		while (ptr != ptrEnd) {
			size_t h = (size_t) *ptr++;
			size_t w = (size_t) *ptr++;
			std::cout << std::endl << "Debug Data found, mat [" << h << " x " << w << "]" << std::endl;
			//Mat<double>::fromArray(h, w, ptr).saveAsCSV("f:/gpu.txt", true);
			Mat<double>::fromArray(h, w, ptr, false).toConsole("", 16);
			ptr += h * w;
		}

		//data.kernelTimings.saveAsBMP("f:/kernel.bmp");
	}

	void inputData(ImageYuv& frame) override {
		cudaReadFrame(mStatus.frameInputIndex, mData, frame);
	}

	void createPyramid() override {
		cudaCreatePyramid(mStatus.frameInputIndex, mData);
	}

	void computePartOne() override {
		DeviceInfo* dev = mData.deviceList[mData.deviceSelected];
		const DeviceInfoCuda& dic = mData.deviceListCuda[dev->targetIndex];
		cudaCompute1(mStatus.frameInputIndex, mData, dic.props);
	}

	void computePartTwo() override {
		cudaCompute2(mStatus.frameInputIndex, mData);
	}

	void computeTerminate() override {
		cudaComputeTerminate(mData, resultPoints);
	}

	void outputData(const AffineTransform& trf, OutputContext outCtx) override {
		cudaOutput(mStatus.frameWriteIndex, mData, outCtx, trf.toArray());
	}

	Mat<float> getTransformedOutput() const override {
		Mat<float> warped = Mat<float>::allocate(3LL * mData.h, mData.w);
		cudaGetTransformedOutput(warped.data(), mData);
		return warped;
	}

	Mat<float> getPyramid(size_t idx) const override {
		Mat<float> out = Mat<float>::allocate(mData.pyramidRowCount * 3LL, mData.w);
		cudaGetPyramid(out.data(), idx, mData);
		return out;
	}

	ImageYuv getInput(int64_t index) const override {
		return cudaGetInput(index, mData);
	}

	void getCurrentInputFrame(ImagePPM& image) override {
		cudaGetCurrentInputFrame(image, mData, mStatus.frameReadIndex - 1);
	}

	void getCurrentOutputFrame(ImagePPM& image) override {
		cudaGetCurrentOutputFrame(image, mData);
	}
};


//---------------------------------------------------------------------
//---------- OPENCL FRAME ---------------------------------------------
//---------------------------------------------------------------------

class OpenClFrame : public MovieFrame {

public:
	OpenClFrame(MainData& data) : MovieFrame(data) {
		DeviceInfo* dev = data.deviceList[data.deviceSelected];
		size_t devIdx = dev->targetIndex;
		cl::init(data, inputFrame, devIdx);
	}

	~OpenClFrame() {}

	void inputData(ImageYuv& frame) override { 
		cl::inputData(mStatus.frameInputIndex, mData, frame);
	}

	void createPyramid() override { 
		cl::createPyramid(mStatus.frameInputIndex, mData);
	}

	void computePartOne() override { 
		cl::computePartOne(); 
	}

	void computePartTwo() override { 
		cl::computePartTwo(); 
	}

	void computeTerminate() override { 
		cl::computeTerminate(); 
	}

	void outputData(const AffineTransform& trf, OutputContext outCtx) override { 
		cl::outputData(mStatus.frameWriteIndex, mData, outCtx, trf.toArray());
	}

	Mat<float> getPyramid(size_t idx) const override {
		Mat<float> out = Mat<float>::zeros(mData.pyramidRowCount * 3LL, mData.w);
		cl::getPyramid(out.data(), idx, mData);
		return out;
	}

	Mat<float> getTransformedOutput() const override { 
		return cl::getTransformedOutput(mData); 
	}

	ImageYuv getInput(int64_t idx) const override { 
		return cl::getInput(idx, mData); 
	}

	void getCurrentInputFrame(ImagePPM& image) override { 
		cl::getCurrentInputFrame(image, mStatus.frameReadIndex - 1);
	}

	void getCurrentOutputFrame(ImagePPM& image) override { 
		cl::getCurrentOutputFrame(image); 
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
	void computePartOne() override {}
	void computePartTwo() override {}
	void computeTerminate() override;
	void outputData(const AffineTransform& trf, OutputContext outCtx) override;
	Mat<float> getTransformedOutput() const override;
	Mat<float> getPyramid(size_t idx) const override;
	ImageYuv getInput(int64_t index) const override;
	void getCurrentInputFrame(ImagePPM& image) override;
	void getCurrentOutputFrame(ImagePPM& image) override;

protected:

	class CpuFrameItem {

	public:
		int64_t frameIndex = -1;
		std::vector<Mat<float>> mY, mDX, mDY;

		CpuFrameItem(MainData& data);
	};

	//frame input buffer, number of frames = frameBufferCount
	std::vector<ImageYuv> mYUV;

	//holds image pyramids
	std::vector<CpuFrameItem> mPyr;

	//buffers the last output frame, 3 mats, to be used to blend background of next frame
	std::vector<Mat<float>> mPrevOut;

	//buffer for generating output from input yuv and transformation
	std::vector<Mat<float>> mBuffer;
	Mat<float> mYuv, mFilterBuffer, mFilterResult;
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
	void computePartOne() override {}
	void computePartTwo() override {}
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
	void computePartOne() {}
	void computePartTwo() {}
	void computeTerminate() {}
	void outputData(const AffineTransform& trf, OutputContext outCtx) {};
};