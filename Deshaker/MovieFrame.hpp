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
#include "ThreadPool.hpp"
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
	virtual void outputData(const AffineTransform& trf) = 0;
	//prepare data for encoding on cpu
	virtual void outputCpu(int64_t frameIndex, ImageYuv& image) = 0;
	//prepare data for encoding on cuda
	virtual void outputCuda(int64_t frameIndex, unsigned char* cudaNv12ptr, int cudaPitch) = 0;
	//output rgb data warped but not unsharped
	virtual void outputRgbWarped(int64_t frameIndex, ImagePPM& image) = 0;

	/*
	* run the stabilizing loop
	*/
	virtual void runLoop(DeshakerPass pass, ProgressDisplay& progress, UserInput& input, AuxWriters& secondaryWriters);

	/*
	* call transform calculation
	*/
	virtual void computeTransform(int64_t frameIndex) final;

	/*
	* get the current frame transform
	*/
	virtual const AffineTransform& getTransform() const final;

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
	virtual void getInput(int64_t frameIndex, ImageYuv& image) const {}

	/*
	* get most recently read input frame
	* use to show progress
	*/
	virtual void getInput(int64_t frameIndex, ImagePPM& image) {}

	/*
	* read transforms from previous pass
	*/
	virtual std::map<int64_t, TransformValues> readTransforms() final;

	/*
	* printable name of MovieFrame instance in use
	*/
	virtual std::string getClassName() const { return "None"; }

	/*
	* short id string for frame instance
	*/
	virtual std::string getClassId() const { return "None"; }

	/*
	* play time for given frame
	*/
	std::string getTimeForFrame(uint64_t frameIndex);


	ThreadPool mPool;
	FrameResult mFrameResult;
	ImageYuv mBufferFrame;
	std::vector<PointResult> mResultPoints;
	std::unique_ptr<RNGbase> rng = std::make_unique<RNG<PseudoRandomSource>>();

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
		mResultPoints(data.resultCount) {
		//set a reference to this frame class into writer object
		writer.movieFrame = this;

		//set PointResult indizes
		int idx = 0;
		for (int y = 0; y < data.iyCount; y++) {
			for (int x = 0; x < data.ixCount; x++) {
				mResultPoints[idx].idx = idx;
				mResultPoints[idx].ix0 = x;
				mResultPoints[idx].iy0 = y;
				idx++;
			}
		}
	}

private:
	void read();
	std::future<void> readAsync();
	void write();
	bool doLoop(UserInput& input);

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
	void outputData(const AffineTransform& trf) override;
	void outputCpu(int64_t frameIndex, ImageYuv& image) override;
	void outputCuda(int64_t frameIndex, unsigned char* cudaNv12ptr, int cudaPitch) override;
	void outputRgbWarped(int64_t frameIndex, ImagePPM& image) override;
	void getInput(int64_t index, ImageYuv& image) const override;
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
	void outputData(const AffineTransform& trf) override {};
	void outputCpu(int64_t frameIndex, ImageYuv& image) override {};
	void outputCuda(int64_t frameIndex, unsigned char* cudaNv12ptr, int cudaPitch) override {};
	void outputRgbWarped(int64_t frameIndex, ImagePPM& image) override {};
};
