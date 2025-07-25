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

#include "MovieWriter.hpp"
#include "ThreadPool.hpp"
#include "SystemStuff.hpp"
#include "MainData.hpp"
#include "Trajectory.hpp"
#include "ProgressBase.hpp"
#include "FrameExecutor.hpp"


 //---------------------------------------------------------------------
 //---------- MOVIE FRAME BASE CLASS -----------------------------------
 //---------------------------------------------------------------------


class MovieFrame {

protected:
	enum class StateCombined {
		READ_FIRST,
		READ_SECOND,
		FILL_BUFFER,
		MAIN_LOOP,
		LAST_COMPUTE,
		DRAIN_BUFFER,
		END,
		QUIT,
		FLUSH,
		CLOSE,
		DONE,
	};

	enum class StateConsecutive {
		READ_FIRST,
		READ_SECOND,
		READ,
		WRITE_INIT,
		WRITE,
		QUIT,
		FLUSH,
		CLOSE,
		DONE,
	};

	enum class InputState {
		NONE,
		SIGNAL,
		HANDLED,
	};

	const MainData& mData;

public:
	MovieReader& mReader;
	MovieWriter& mWriter;
	ThreadPool mPool;
	FrameResult mFrameResult;
	ImageYuv mBufferFrame;
	Trajectory mTrajectory;
	std::vector<PointResult> mResultPoints;

	//only constructor for Frame class
	MovieFrame(MainData& data, MovieReader& reader, MovieWriter& writer);

	MovieFrame(const MovieFrame& other) = delete;
	MovieFrame(MovieFrame&& other) = delete;
	virtual ~MovieFrame();

	// call transform calculation
	virtual void computeTransform(int64_t frameIndex) final;

	// read transforms from previous pass
	virtual std::map<int64_t, TransformValues> readTransforms() final;

	// get the current frame transform
	virtual const AffineTransform& getTransform() const final;

	// play time for given frame
	std::string ptsForFrameAsString(int64_t frameIndex) const;

	//run loop in subclass
	virtual void runLoop(std::shared_ptr<ProgressBase> progress, UserInput& input, std::shared_ptr<FrameExecutor> executor) {}

	//run loop with default progress and input, used for development
	void runLoop(std::shared_ptr<FrameExecutor> executor);
};


class MovieFrameCombined : public MovieFrame {
public:
	MovieFrameCombined(MainData& data, MovieReader& reader, MovieWriter& writer) :
		MovieFrame(data, reader, writer) {}

	void runLoop(std::shared_ptr<ProgressBase> progress, UserInput& input, std::shared_ptr<FrameExecutor> executor) override;
};


class MovieFrameConsecutive : public MovieFrame {
public:
	MovieFrameConsecutive(MainData& data, MovieReader& reader, MovieWriter& writer) :
		MovieFrame(data, reader, writer) {}

	void runLoop(std::shared_ptr<ProgressBase> progress, UserInput& input, std::shared_ptr<FrameExecutor> executor) override;
};
