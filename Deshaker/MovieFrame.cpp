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

#include "MovieFrame.hpp"
#include "MovieReader.hpp"
#include "ProgressDisplay.hpp"
#include "Util.hpp"


//only constructor for Frame class
MovieFrame::MovieFrame(MainData& data, MovieReader& reader, MovieWriter& writer) :
	mData { data },
	mReader { reader },
	mWriter { writer },
	mPool(data.cpuThreads),
	mFrameResult(data, mPool),
	mBufferFrame(data.h, data.w, data.cpupitch),
	mResultPoints(data.resultCount)
{
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

	//allocate trajectory list
	mTrajectory.reserve(reader.frameCount);
}

std::string MovieFrame::ptsForFrameAsString(int64_t frameIndex) const {
	return mReader.ptsForFrameAsString(frameIndex).value_or("");
}

//shutdown
MovieFrame::~MovieFrame() {
	mPool.shutdown(); //shutdown threads
}

//read transform parameters from file, usually for second run
std::map<int64_t, TransformValues> MovieFrame::readTransforms() {
	return TransformsFile::readTransformMap(mData.trajectoryFile);
}

const AffineTransform& MovieFrame::getTransform() const {
	return mFrameResult.getTransform();
}

void MovieFrame::runLoop(std::shared_ptr<FrameExecutor> executor) {
	std::shared_ptr<ProgressBase> progress = std::make_shared<ProgressDefault>();
	UserInputDefault input;
	runLoop(progress, input, executor);
}

void MovieFrame::progressUpdate(ProgressInfo& progressInfo, std::shared_ptr<ProgressBase> progress, double totalProgress, bool forceUpdate) {
	if (mReader.frameCount > 0) progressInfo.totalProgress = std::clamp(totalProgress, 0.0, 100.0);
	else progressInfo.totalProgress = std::numeric_limits<double>::quiet_NaN();
	std::unique_lock<std::mutex> lock(mWriter.mStatsMutex);
	progressInfo.readIndex = mReader.frameIndex;
	progressInfo.writeIndex = mWriter.frameIndex;
	progressInfo.encodeIndex = mWriter.frameEncoded;
	progressInfo.outputBytesWritten = mWriter.outputBytesWritten;
	lock.unlock();
	progress->update(progressInfo, forceUpdate);
}

MovieFrameCombined::MovieFrameCombined(MainData& data, MovieReader& reader, MovieWriter& writer) :
	MovieFrame(data, reader, writer)
{
	data.bufferCount = 2 * data.radius + 2;
}

MovieFrameConsecutive::MovieFrameConsecutive(MainData& data, MovieReader& reader, MovieWriter& writer) :
	MovieFrame(data, reader, writer)
{
	data.bufferCount = 2;
}


//---------------------------------------------------------------
//---------- DESHAKER LOOP COMBINED -----------------------------
//---------------------------------------------------------------

void MovieFrameCombined::runLoop(std::shared_ptr<ProgressBase> progress, UserInput& input, std::shared_ptr<FrameExecutor> executor) {
	InputState inputState = InputState::NONE;
	StateCombined state = StateCombined::READ_FIRST;
	bool hasFramesToFlush = false;
	ProgressInfo progressInfo = { mReader.frameCount };

	while (state != StateCombined::DONE) {
		//handle current state
		if (state == StateCombined::READ_FIRST) {
			//show program header on console
			if (mData.printHeader) mData.showIntro(executor->mDeviceInfo.getName(), mReader);
			//init progress display
			progress->init();
			progress->updateStatus("Processing Frames...");
			//read first frame from input into buffer
			mReader.read(mBufferFrame);
			mReader.startOfInput = false;

		} else if (state == StateCombined::READ_SECOND) {
			executor->inputData(mReader.frameIndex, mBufferFrame); //input first frame
			executor->createPyramid(mReader.frameIndex);
			mReader.read(mBufferFrame); //read second frame

		} else if (state == StateCombined::FILL_BUFFER) {
			//process current frame
			executor->inputData(mReader.frameIndex, mBufferFrame);
			executor->createPyramid(mReader.frameIndex);
			//transform for previous frame
			mFrameResult.computeTransform(mResultPoints, mReader.frameIndex - 1);
			mTrajectory.addTrajectoryTransform(mFrameResult.getTransform());
			//begin computing smooth transform
			if (mReader.frameIndex > mData.radius) mTrajectory.computeSmoothTransform(mData, mReader.frameIndex - mData.radius - 1);
			mWriter.writeInput(*executor);
			//compute flow for current frame
			executor->computeStart(mReader.frameIndex, mResultPoints);
			executor->computeTerminate(mReader.frameIndex, mResultPoints);
			//read next frame into buffer
			mReader.read(mBufferFrame);

		} else if (state == StateCombined::MAIN_LOOP) {
			//process current frame, read next frame, output previous frame
			//util::ConsoleTimer timer_fr("frame");
			//reader will update frameIndex async
			int64_t readIndex = mReader.frameIndex;
			assert(readIndex % mData.bufferCount != mWriter.frameIndex % mData.bufferCount && "accessing the same buffer for read and write");
			//process current frame
			executor->inputData(readIndex, mBufferFrame);
			executor->createPyramid(readIndex);
			executor->computeStart(readIndex, mResultPoints);
			//read next frame async
			std::future<void> f = mReader.readAsync(mBufferFrame);
			//now compute transform for previous frame while results for current frame are potentially computed on device
			mFrameResult.computeTransform(mResultPoints, readIndex - 1);
			const AffineTransform& currentTransform = mFrameResult.getTransform();
			mTrajectory.addTrajectoryTransform(currentTransform);
			mTrajectory.computeSmoothTransform(mData, readIndex - mData.radius - 1);
			mTrajectory.computeSmoothZoom(mData, mWriter.frameIndex);
			const AffineTransform& finalTransform = mTrajectory.getTransform(mData, mWriter.frameIndex);
			executor->outputData(mWriter.frameIndex, finalTransform);
			mWriter.prepareOutput(*executor);
			mWriter.writeOutput(*executor);
			mWriter.writeInput(*executor);
			//get computed flow for current frame
			executor->computeTerminate(readIndex, mResultPoints);
			//wait for async read to complete
			f.wait();

		} else if (state == StateCombined::LAST_COMPUTE) {
			//compute last transform
			mFrameResult.computeTransform(mResultPoints, mReader.frameIndex - 1);
			//add last transform
			mTrajectory.addTrajectoryTransform(mFrameResult.getTransform());
			//compute all remaining transforms
			for (int64_t idx = mWriter.frameIndex; idx < mTrajectory.size(); idx++) {
				if (mTrajectory.isComputed(idx) == false) mTrajectory.computeSmoothTransform(mData, idx);
			}
			mTrajectory.computeSmoothZoom(mData, mWriter.frameIndex);
			const AffineTransform& finalTransform = mTrajectory.getTransform(mData, mWriter.frameIndex);
			executor->outputData(mWriter.frameIndex, finalTransform);
			mWriter.prepareOutput(*executor);
			mWriter.writeOutput(*executor);
			mWriter.writeInput(*executor);

		} else if (state == StateCombined::DRAIN_BUFFER) {
			mTrajectory.computeSmoothZoom(mData, mWriter.frameIndex);
			const AffineTransform& tf = mTrajectory.getTransform(mData, mWriter.frameIndex);
			executor->outputData(mWriter.frameIndex, tf);
			mWriter.prepareOutput(*executor);
			mWriter.writeOutput(*executor);

		} else if (state == StateCombined::QUIT) {
			//start to flush the writer
			hasFramesToFlush = mWriter.startFlushing();

		} else if (state == StateCombined::FLUSH) {
			//flush the writer as long as there are frames left
			hasFramesToFlush = mWriter.flush();
		}

		//update progress
		double frameCount = mReader.frameCount;
		double p = 75.0 * mReader.frameIndex / frameCount + 25.0 * mWriter.frameIndex / frameCount;
		progressUpdate(progressInfo, progress, p, state == StateCombined::CLOSE);

		//prevent system sleep
		keepSystemAlive();

		//stop signal must be handled exactly once
		if (inputState == InputState::NONE && (mReader.endOfInput || mReader.frameIndex >= mData.maxFrames)) {
			inputState = InputState::SIGNAL;
		}

		//check user input and set state
		UserInputEnum e = input.checkState();
		if (e == UserInputEnum::END) {
			progress->writeMessage("[e] command received. Stop reading input.");
			state = StateCombined::LAST_COMPUTE;

		} else if (e == UserInputEnum::QUIT) {
			progress->writeMessage("[q] command received. Stop writing output.");
			state = StateCombined::QUIT;

		} else if (e == UserInputEnum::HALT) {
			progress->writeMessage("[x] command received. Terminating.");
			state = StateCombined::DONE;

		} else if (errorLogger().hasError()) {
			state = StateCombined::DONE;

		} else if (inputState == InputState::SIGNAL) {
			inputState = InputState::HANDLED;
			state = (state == StateCombined::READ_FIRST) ? state = StateCombined::QUIT : StateCombined::LAST_COMPUTE;

		} else if (state == StateCombined::READ_FIRST) {
			state = StateCombined::READ_SECOND;

		} else if (state == StateCombined::READ_SECOND) {
			state = StateCombined::FILL_BUFFER;

		} else if (state == StateCombined::FILL_BUFFER && mReader.frameIndex == mData.bufferCount - 1LL) {
			state = StateCombined::MAIN_LOOP;

		} else if (state == StateCombined::LAST_COMPUTE) {
			state = mWriter.frameIndex == mReader.frameIndex ? StateCombined::QUIT : StateCombined::DRAIN_BUFFER;

		} else if (state == StateCombined::DRAIN_BUFFER) {
			state = mWriter.frameIndex == mReader.frameIndex ? StateCombined::QUIT : StateCombined::DRAIN_BUFFER;

		} else if (state == StateCombined::QUIT) {
			state = hasFramesToFlush ? StateCombined::FLUSH : StateCombined::CLOSE;

		} else if (state == StateCombined::FLUSH) {
			state = hasFramesToFlush ? StateCombined::FLUSH : StateCombined::CLOSE;

		} else if (state == StateCombined::CLOSE) {
			state = StateCombined::DONE;
		}
	}
	
	//std::cout << mTrajectory << std::endl;
}

// -----------------------------------------
// -------- DESHAKER LOOP CONSECUTIVE-------
// -----------------------------------------

void MovieFrameConsecutive::runLoop(std::shared_ptr<ProgressBase> progress, UserInput& input, std::shared_ptr<FrameExecutor> executor) {
	InputState inputState = InputState::NONE;
	StateConsecutive state = StateConsecutive::READ_FIRST;
	bool hasFramesToFlush = false;
	int currentPass = 1;
	int64_t maxFrames = mData.maxFrames;
	ProgressInfo progressInfo = { mReader.frameCount };

	while (state != StateConsecutive::DONE) {
		//handle current state
		if (state == StateConsecutive::READ_FIRST) {
			//start the loop, read first frame into buffer
			mReader.mStoreSidePackets = false; //we do not want packets to pile up on first iteration
			//show program header on console
			if (mData.printHeader) mData.showIntro(executor->mDeviceInfo.getName(), mReader);
			//read first frame from input into buffer
			//init progress display
			progress->init();
			progress->updateStatus("Analyzing Video...");
			mReader.read(mBufferFrame);
			mReader.startOfInput = false;

		} else if (state == StateConsecutive::READ_SECOND) {
			//create pyramid for first frame, read second frame
			executor->inputData(mReader.frameIndex, mBufferFrame); //input first frame
			executor->createPyramid(mReader.frameIndex);
			mReader.read(mBufferFrame); //read second frame
			mTrajectory.addTrajectoryTransform({}); //first frame has no transform applied
			mWriter.writeInput(*executor);

		} else if (state == StateConsecutive::READ) {
			//main loop, compute frame, read one frame ahead async
			//reader will update frameIndex async
			int64_t idx = mReader.frameIndex;
			executor->inputData(idx, mBufferFrame);
			std::future<void> f = mReader.readAsync(mBufferFrame);
			executor->createPyramid(idx);
			executor->computeStart(idx, mResultPoints);
			executor->computeTerminate(idx, mResultPoints);
			mFrameResult.computeTransform(mResultPoints, idx);
			mTrajectory.addTrajectoryTransform(mFrameResult.getTransform());
			mWriter.writeInput(*executor);
			f.wait();

		} else if (state == StateConsecutive::WRITE_PREPARE) {
			//prepare for second pass, rewind reader and read first frame
			mReader.rewind();
			mReader.mStoreSidePackets = true;
			progress->updateStatus("Generating Output...");
			mReader.read(mBufferFrame);
			currentPass = 2;
			inputState = InputState::NONE;
			//compute smooth transforms
			for (int64_t idx = 0; idx < mTrajectory.size(); idx++) mTrajectory.computeSmoothTransform(mData, idx);
			for (int64_t idx = 0; idx < mTrajectory.size(); idx++) mTrajectory.computeSmoothZoom(mData, idx);

		} else if (state == StateConsecutive::WRITE) {
			//output stabilized frame, read next frame
			executor->inputData(mReader.frameIndex, mBufferFrame);
			const AffineTransform& tf = mTrajectory.getTransform(mData, mWriter.frameIndex);
			executor->outputData(mWriter.frameIndex, tf);
			mWriter.prepareOutput(*executor);
			mWriter.writeOutput(*executor);
			mReader.read(mBufferFrame);

		} else if (state == StateConsecutive::QUIT) {
			//start to flush the writer
			hasFramesToFlush = mWriter.startFlushing();

		} else if (state == StateConsecutive::FLUSH) {
			//flush the writer as long as there are frames left
			hasFramesToFlush = mWriter.flush();
		}

		//update progress
		double frameCount = mReader.frameCount;
		double p;
		if (currentPass == 1) p = 85.0 * mReader.frameIndex / frameCount;
		else p = 85.0 + 15.0 * mWriter.frameIndex / frameCount;
		progressUpdate(progressInfo, progress, p, state == StateConsecutive::CLOSE);
		
		//prevent system sleep
		keepSystemAlive();

		//stop signal must be handled exactly once
		if (inputState == InputState::NONE && (mReader.endOfInput || mReader.frameIndex >= maxFrames)) {
			inputState = InputState::SIGNAL;
		}

		//check user input and set state
		UserInputEnum e = input.checkState();
		if (e == UserInputEnum::END && currentPass == 1) {
			progress->writeMessage("[e] command received. Stop reading input.");
			maxFrames = mReader.frameIndex;
			state = StateConsecutive::WRITE_PREPARE;

		} else if (e == UserInputEnum::QUIT) {
			progress->writeMessage("[q] command received. Stopping.");
			state = StateConsecutive::QUIT;

		} else if (e == UserInputEnum::HALT) {
			progress->writeMessage("[x] command received. Terminating.");
			state = StateConsecutive::DONE;

		} else if (errorLogger().hasError()) {
			state = StateConsecutive::DONE;

		} else if (inputState == InputState::SIGNAL) {
			inputState = InputState::HANDLED;
			state = currentPass == 1 ? StateConsecutive::WRITE_PREPARE : StateConsecutive::QUIT;

		} else if (state == StateConsecutive::READ_FIRST) {
			state = StateConsecutive::READ_SECOND;

		} else if (state == StateConsecutive::READ_SECOND) {
			state = StateConsecutive::READ;

		} else if (state == StateConsecutive::WRITE_PREPARE) {
			state = StateConsecutive::WRITE;

		} else if (state == StateConsecutive::QUIT) {
			state = hasFramesToFlush ? StateConsecutive::FLUSH : StateConsecutive::CLOSE;

		} else if (state == StateConsecutive::FLUSH) {
			state = hasFramesToFlush ? StateConsecutive::FLUSH : StateConsecutive::CLOSE;

		} else if (state == StateConsecutive::CLOSE) {
			state = StateConsecutive::DONE;
		}
	}

	//std::cout << mTrajectory << std::endl;
}
