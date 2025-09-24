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

LoopResult MovieFrame::runLoop(std::shared_ptr<FrameExecutor> executor) {
	std::shared_ptr<ProgressBase> progress = std::make_shared<ProgressDefault>();
	UserInputDefault input;
	return runLoop(progress, input, executor);
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

LoopResult MovieFrame::runLoop(std::shared_ptr<ProgressBase> progress, UserInput& input, std::shared_ptr<FrameExecutor> executor) {
	return LoopResult::LOOP_NONE;
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

LoopResult MovieFrameCombined::runLoop(std::shared_ptr<ProgressBase> progress, UserInput& input, std::shared_ptr<FrameExecutor> executor) {
	assert(mData.mode == 0 && "mode must be == 0 here");
	InputState inputState = InputState::NONE;
	StateCombined state = StateCombined::READ_FIRST;
	bool hasFramesToFlush = false;
	ProgressInfo progressInfo = { mReader.frameCount };
	LoopResult loopResult = LoopResult::LOOP_SUCCESS;

	while (state != StateCombined::DONE) {
		//handle current state
		if (state == StateCombined::READ_FIRST) {
			mWriter.start();
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
			executor->createPyramid(mReader.frameIndex, {}, false);
			mReader.read(mBufferFrame); //read second frame

		} else if (state == StateCombined::FILL_BUFFER) {
			//process current frame
			executor->inputData(mReader.frameIndex, mBufferFrame);
			executor->createPyramid(mReader.frameIndex, {}, false);
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
			executor->createPyramid(readIndex, {}, false);
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
			mWriter.writeOutput(*executor);
			mWriter.writeInput(*executor);

		} else if (state == StateCombined::DRAIN_BUFFER) {
			mTrajectory.computeSmoothZoom(mData, mWriter.frameIndex);
			const AffineTransform& tf = mTrajectory.getTransform(mData, mWriter.frameIndex);
			executor->outputData(mWriter.frameIndex, tf);
			mWriter.writeOutput(*executor);

		} else if (state == StateCombined::QUIT) {
			//start to flush the writer
			hasFramesToFlush = mWriter.startFlushing();

		} else if (state == StateCombined::FLUSH) {
			//flush the writer as long as there are frames left
			hasFramesToFlush = mWriter.flush();

		} else if (state == StateCombined::CLOSE) {
			mWriter.close();
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
			loopResult = LoopResult::LOOP_CANCELLED;

		} else if (e == UserInputEnum::QUIT) {
			progress->writeMessage("[q] command received. Stop writing output.");
			state = StateCombined::QUIT;
			loopResult = LoopResult::LOOP_CANCELLED;

		} else if (e == UserInputEnum::HALT) {
			progress->writeMessage("[x] command received. Terminating.");
			state = StateCombined::DONE;
			loopResult = LoopResult::LOOP_CANCELLED;

		} else if (errorLogger().hasError()) {
			state = StateCombined::DONE;
			loopResult = LoopResult::LOOP_ERROR;

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
	return loopResult;
}

// -----------------------------------------
// -------- DESHAKER LOOP CONSECUTIVE-------
// -----------------------------------------

LoopResult MovieFrameConsecutive::runLoop(std::shared_ptr<ProgressBase> progress, UserInput& input, std::shared_ptr<FrameExecutor> executor) {
	assert(mData.mode > 0 && "mode must be > 0 here");
	InputState inputState = InputState::NONE;
	StateConsecutive state = StateConsecutive::READ_FIRST_FRAME;
	bool hasFramesToFlush = false;
	int currentPass = 1;
	int64_t maxFrames = mData.maxFrames;
	ProgressInfo progressInfo = { mReader.frameCount };
	double progressPerReaderPass = 80.0 / (80.0 * mData.mode + 20.0);
	LoopResult loopResult = LoopResult::LOOP_SUCCESS;

	while (state != StateConsecutive::DONE) {
		//handle current state
		if (state == StateConsecutive::READ_FIRST_FRAME) {
			//start the loop, read first frame into buffer
			mReader.mStoreSidePackets = false; //we do not want packets to pile up on first iteration
			//show program header on console
			if (mData.printHeader) mData.showIntro(executor->mDeviceInfo.getName(), mReader);
			//read first frame from input into buffer
			//init progress display
			progress->init();
			progress->updateStatus(std::format("Analyzing Video {}/{}...", currentPass, mData.mode));
			mReader.read(mBufferFrame);
			mReader.startOfInput = false;

		} else if (state == StateConsecutive::READ_SECOND_FRAME) {
			//create pyramid for first frame, read second frame
			executor->inputData(mReader.frameIndex, mBufferFrame); //input first frame
			executor->createPyramid(mReader.frameIndex, {}, false);
			mReader.read(mBufferFrame); //read second frame
			mTrajectory.addTrajectoryTransform({}); //first frame has no transform applied
			mWriter.writeInput(*executor);

		} else if (state == StateConsecutive::READ) {
			//main loop, compute frame, read one frame ahead async
			//reader will update frameIndex async
			int64_t idx = mReader.frameIndex;
			executor->inputData(idx, mBufferFrame);
			std::future<void> f = mReader.readAsync(mBufferFrame);
			executor->createPyramid(idx, {}, false);
			executor->computeStart(idx, mResultPoints);
			executor->computeTerminate(idx, mResultPoints);
			mFrameResult.computeTransform(mResultPoints, idx);
			mTrajectory.addTrajectoryTransform(mFrameResult.getTransform());
			mWriter.writeInput(*executor);
			f.wait();

		} else if (state == StateConsecutive::ITERATION_PREPARE) {
			progressUpdate(progressInfo, progress, progressPerReaderPass * currentPass * 100.0, true);
			//prepare next iteration over input
			mReader.rewind();
			mReader.read(mBufferFrame);
			currentPass++;
			progress->updateStatus(std::format("Analyzing Video {}/{}...", currentPass, mData.mode));
			inputState = InputState::NONE;
			//compute smooth transforms and leave zoom at 1.0
			for (int64_t idx = 0; idx < mTrajectory.size(); idx++) mTrajectory.computeSmoothTransform(mData, idx);
			mTrajectory.reset();
			mFrameResult.reset();
			//std::cout << mTrajectory << std::endl;

		} else if (state == StateConsecutive::ITERATION_SECOND_FRAME) {
			executor->inputData(mReader.frameIndex, mBufferFrame); //input first frame
			executor->createPyramid(mReader.frameIndex, mTrajectory.getTransform(mData, 0), true);
			mReader.read(mBufferFrame); //read second frame
			mTrajectory.setTrajectoryTransform({});

		} else if (state == StateConsecutive::ITERATION) {
			//compute again using transformed input frames
			int64_t idx = mReader.frameIndex;
			executor->inputData(idx, mBufferFrame);
			std::future<void> f = mReader.readAsync(mBufferFrame);
			const AffineTransform& trf = mTrajectory.getTransform(mData, idx);
			executor->createPyramid(idx, trf, true);
			executor->computeStart(idx, mResultPoints);
			executor->computeTerminate(idx, mResultPoints);
			const AffineTransform& newTransform = mFrameResult.computeTransform(mResultPoints, idx);
			mTrajectory.setTrajectoryTransform(newTransform);
			mWriter.writeInput(*executor);
			f.wait();

		} else if (state == StateConsecutive::WRITE_PREPARE) {
			mWriter.start();
			progressUpdate(progressInfo, progress, progressPerReaderPass * currentPass * 100.0, true);
			//prepare for second pass, rewind reader and read first frame
			mReader.rewind();
			mReader.mStoreSidePackets = true;
			mReader.read(mBufferFrame);
			currentPass++;
			progress->updateStatus("Generating Output...");
			inputState = InputState::NONE;
			//compute smooth transforms
			for (int64_t idx = 0; idx < mTrajectory.size(); idx++) mTrajectory.computeSmoothTransform(mData, idx);
			for (int64_t idx = 0; idx < mTrajectory.size(); idx++) mTrajectory.computeSmoothZoom(mData, idx);
			//std::cout << mTrajectory << std::endl;

		} else if (state == StateConsecutive::WRITE) {
			//output stabilized frame, read next frame
			executor->inputData(mReader.frameIndex, mBufferFrame);
			const AffineTransform& tf = mTrajectory.getTransform(mData, mWriter.frameIndex);
			executor->outputData(mWriter.frameIndex, tf);
			mWriter.writeOutput(*executor);
			mReader.read(mBufferFrame);

		} else if (state == StateConsecutive::QUIT) {
			//start to flush the writer
			hasFramesToFlush = mWriter.startFlushing();

		} else if (state == StateConsecutive::FLUSH) {
			//flush the writer as long as there are frames left
			hasFramesToFlush = mWriter.flush();

		} else if (state == StateConsecutive::CLOSE) {
			mWriter.close();
		}

		//update progress
		double frameCount = mReader.frameCount;
		double p = progressPerReaderPass * (currentPass - 1);
		if (currentPass <= mData.mode) p += progressPerReaderPass * mReader.frameIndex / frameCount;
		else p += (1.0 - p) * mWriter.frameIndex / frameCount;
		progressUpdate(progressInfo, progress, p * 100.0, state == StateConsecutive::CLOSE);
		
		//prevent system sleep
		keepSystemAlive();

		//stop signal must be handled exactly once
		if (inputState == InputState::NONE && (mReader.endOfInput || mReader.frameIndex >= maxFrames)) {
			inputState = InputState::SIGNAL;
		}

		//check user input and set state
		UserInputEnum e = input.checkState();
		if (e == UserInputEnum::END && currentPass == mData.mode) {
			progress->writeMessage("[e] command received. Stop reading input.");
			maxFrames = mReader.frameIndex;
			state = StateConsecutive::WRITE_PREPARE;
			loopResult = LoopResult::LOOP_CANCELLED;

		} else if (e == UserInputEnum::END) {
			progress->writeMessage("[e] command received. Stop reading more input.");
			maxFrames = mReader.frameIndex;
			state = StateConsecutive::ITERATION_PREPARE;
			loopResult = LoopResult::LOOP_CANCELLED;

		} else if (e == UserInputEnum::QUIT) {
			progress->writeMessage("[q] command received. Stopping.");
			state = StateConsecutive::QUIT;
			loopResult = LoopResult::LOOP_CANCELLED;

		} else if (e == UserInputEnum::HALT) {
			progress->writeMessage("[x] command received. Terminating.");
			state = StateConsecutive::DONE;
			loopResult = LoopResult::LOOP_CANCELLED;

		} else if (errorLogger().hasError()) {
			state = StateConsecutive::DONE;
			loopResult = LoopResult::LOOP_ERROR;

		} else if (inputState == InputState::SIGNAL && currentPass > mData.mode) {
			inputState = InputState::HANDLED;
			state = StateConsecutive::QUIT;

		} else if (inputState == InputState::SIGNAL && currentPass == mData.mode) {
			inputState = InputState::HANDLED;
			state = StateConsecutive::WRITE_PREPARE;

		} else if (inputState == InputState::SIGNAL) {
			inputState = InputState::HANDLED;
			state = StateConsecutive::ITERATION_PREPARE;

		} else if (state == StateConsecutive::READ_FIRST_FRAME) {
			state = StateConsecutive::READ_SECOND_FRAME;

		} else if (state == StateConsecutive::READ_SECOND_FRAME) {
			state = StateConsecutive::READ;

		} else if (state == StateConsecutive::ITERATION_PREPARE) {
			state = StateConsecutive::ITERATION_SECOND_FRAME;

		} else if (state == StateConsecutive::ITERATION_SECOND_FRAME) {
			state = StateConsecutive::ITERATION;

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
	return loopResult;
}
