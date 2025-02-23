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

std::string MovieFrame::ptsForFrameAsString(int64_t frameIndex) {
	return mReader.ptsForFrameAsString(frameIndex).value_or("");
}

//shutdown
MovieFrame::~MovieFrame() {
	mPool.shutdown(); //shutdown threads
}

//compute affine transform parameters given the indiviual point results
void MovieFrame::computeTransform(int64_t frameIndex) {
	mFrameResult.computeTransform(mResultPoints, mPool, frameIndex, mData.sampler);
}

//read transform parameters from file, usually for second run
std::map<int64_t, TransformValues> MovieFrame::readTransforms() {
	return TransformsFile::readTransformMap(mData.trajectoryFile);
}

const AffineTransform& MovieFrame::getTransform() const {
	return mFrameResult.getTransform();
}

//try to read next frame from reader class into input buffer
void MovieFrame::read() {
	mReader.read(mBufferFrame);
}

std::future<void> MovieFrame::readAsync() {
	return mReader.readAsync(mBufferFrame);
}

//check if we should continue with the next frame
bool MovieFrame::doLoop(UserInput& input) {
	return errorLogger().hasNoError() && input.doContinue() && mReader.endOfInput == false && mReader.frameIndex < mData.maxFrames;
}


//---------------------------------------------------------------------
//---------- DESHAKER LOOPS  ------------------------------------------
//---------------------------------------------------------------------

void MovieFrame::loopInit(std::shared_ptr<ProgressBase> progress, std::shared_ptr<FrameExecutor> executor, const std::string& message) {
	//read first frame from input into buffer
	if (errorLogger().hasNoError()) read();
	//show program header on console
	if (errorLogger().hasNoError() && mData.printHeader) mData.showIntro(executor->mDeviceInfo.getName(), mReader);
	//init progress display
	progress->init();
	progress->writeMessage(message);
	progress->update();

	//first pyramid
	if (errorLogger().hasNoError() && mReader.endOfInput == false) {
		executor->inputData(mReader.frameIndex, mBufferFrame); //input first frame
		executor->createPyramid(mReader.frameIndex);
		read(); //read second frame
		progress->update();
	}
}

void MovieFrame::loopTerminate(std::shared_ptr<ProgressBase> progress, UserInput& input, AuxWriters& auxWriters, std::shared_ptr<FrameExecutor> executor) {
	//flush writer buffer
	bool hasFrame = mWriter.startFlushing();
	while (errorLogger().hasNoError() && input.mCurrentInput < UserInputEnum::HALT && hasFrame) {
		hasFrame = mWriter.flush();
		progress->update();
	}
	progress->forceUpdate();
}

//loop to handle input - stabilization - output in one go
void MovieFrameCombined::runLoop(std::shared_ptr<ProgressBase> progress, UserInput& input, AuxWriters& auxWriters, std::shared_ptr<FrameExecutor> executor) {
	loopInit(progress, executor);

	//fill input buffer and compute transformations
	while (doLoop(input) && mReader.frameIndex < mData.bufferCount - 1LL) {
		//process current frame
		executor->inputData(mReader.frameIndex, mBufferFrame);
		executor->createPyramid(mReader.frameIndex);
		//transform for previous frame
		computeTransform(mReader.frameIndex - 1);
		mTrajectory.addTrajectoryTransform(mFrameResult.getTransform());
		auxWriters.writeAll(*executor);
		//compute flow for current frame
		executor->computeStart(mReader.frameIndex, mResultPoints);
		executor->computeTerminate(mReader.frameIndex, mResultPoints);
		//read next frame into buffer
		read();

		//check if there is input on the console
		input.checkState();
		progress->update();
	}

	//main loop
	//read data and compute frame results
	while (doLoop(input)) {
		//util::ConsoleTimer timer_fr("frame");
		int64_t readIndex = mReader.frameIndex;
		assert(readIndex % mData.bufferCount != mWriter.frameIndex % mData.bufferCount && "accessing the same buffer for read and write");
		
		//process current frame
		executor->inputData(mReader.frameIndex, mBufferFrame);
		executor->createPyramid(readIndex);
		executor->computeStart(readIndex, mResultPoints);
		//read next frame async
		std::future<void> futRead = readAsync();
		
		//now compute transform for previous frame while results for current frame are potentially computed on device
		computeTransform(readIndex - 1);
		const AffineTransform& currentTransform = mFrameResult.getTransform();
		mTrajectory.addTrajectoryTransform(currentTransform);
		const AffineTransform& finalTransform = mTrajectory.computeSmoothTransform(mData, mWriter.frameIndex);
		executor->outputData(mWriter.frameIndex, finalTransform);
		mWriter.prepareOutput(*executor);
		
		//write output
		mWriter.write(*executor);
		auxWriters.writeAll(*executor);
		
		//get computed flow for current frame
		executor->computeTerminate(readIndex, mResultPoints);

		//wait for async to complete
		futRead.wait();

		//check if there is input on the console
		input.checkState();
		progress->update();
	}

	//process last frame in buffer
	if (errorLogger().hasNoError() && input.mCurrentInput <= UserInputEnum::END) {
		computeTransform(mReader.frameIndex);
		mTrajectory.addTrajectoryTransform(mFrameResult.getTransform());
		const AffineTransform& finalTransform = mTrajectory.computeSmoothTransform(mData, mWriter.frameIndex);
		executor->outputData(mWriter.frameIndex, finalTransform);
		mWriter.prepareOutput(*executor);
		mWriter.write(*executor);
		auxWriters.writeAll(*executor);
		progress->update();
	}

	//write remaining frames
	while (errorLogger().hasNoError() && mWriter.frameIndex < mReader.frameIndex && input.mCurrentInput <= UserInputEnum::END) {
		const AffineTransform& tf = mTrajectory.computeSmoothTransform(mData, mWriter.frameIndex);
		executor->outputData(mWriter.frameIndex, tf);
		mWriter.prepareOutput(*executor);
		mWriter.write(*executor);

		//check if there is input on the console
		input.checkState();
		progress->update();
	}

	loopTerminate(progress, input, auxWriters, executor);
}

void MovieFrameFirst::runLoop(std::shared_ptr<ProgressBase> progress, UserInput& input, AuxWriters& auxWriters, std::shared_ptr<FrameExecutor> executor) {
	mReader.storePackets = false; //do not keep any secondary packets when we do not write output anyway
	loopInit(progress, executor);
	mFrameResult.reset();
	mTrajectory.addTrajectoryTransform(mFrameResult.getTransform()); //first frame has no transform applied
	mWriter.write(*executor);
	auxWriters.writeAll(*executor);

	while (doLoop(input)) {
		executor->inputData(mReader.frameIndex, mBufferFrame);
		executor->createPyramid(mReader.frameIndex);

		executor->computeStart(mReader.frameIndex, mResultPoints);
		executor->computeTerminate(mReader.frameIndex, mResultPoints);
		computeTransform(mReader.frameIndex);
		mWriter.write(*executor);
		auxWriters.writeAll(*executor);
		read();

		//check if there is input on the console
		input.checkState();
		progress->update();
	}
	progress->forceUpdate();
}

void MovieFrameSecond::runLoop(std::shared_ptr<ProgressBase> progress, UserInput& input, AuxWriters& auxWriters, std::shared_ptr<FrameExecutor> executor) {
	//setup list of transforms from file
	auto map = readTransforms();
	if (errorLogger().hasNoError()) mTrajectory.readTransforms(map);

	//init
	if (errorLogger().hasNoError()) read();
	if (errorLogger().hasNoError() && mData.printHeader) mData.showIntro(executor->mDeviceInfo.getName(), mReader);
	progress->init();
	progress->update();

	//looping for output
	while (doLoop(input)) {
		executor->inputData(mReader.frameIndex, mBufferFrame);
		read();

		const AffineTransform& tf = mTrajectory.computeSmoothTransform(mData, mWriter.frameIndex);
		executor->outputData(mWriter.frameIndex, tf);
		mWriter.prepareOutput(*executor);
		mWriter.write(*executor);

		//check if there is input on the console
		input.checkState();
		progress->update();
	}

	loopTerminate(progress, input, auxWriters, executor);
}

void MovieFrameConsecutive::runLoop(std::shared_ptr<ProgressBase> progress, UserInput& input, AuxWriters& auxWriters, std::shared_ptr<FrameExecutor> executor) {
	mReader.storePackets = false; //we do not want packets to pile up on first iteration
	loopInit(progress, executor, "first pass - analyzing input\n");
	mFrameResult.reset();
	mTrajectory.addTrajectoryTransform(mFrameResult.getTransform()); //first frame has no transform applied
	auxWriters.writeAll(*executor);

	//first run - analyse
	while (doLoop(input)) {
		executor->inputData(mReader.frameIndex, mBufferFrame);
		executor->createPyramid(mReader.frameIndex);

		executor->computeStart(mReader.frameIndex, mResultPoints);
		executor->computeTerminate(mReader.frameIndex, mResultPoints);
		computeTransform(mReader.frameIndex);
		mTrajectory.addTrajectoryTransform(mFrameResult.getTransform());
		auxWriters.writeAll(*executor);
		read();

		//check if there is input on the console
		input.checkState();
		progress->update();
	}
	progress->forceUpdate();

	//rewind input
	mReader.rewind();
	mReader.storePackets = true;
	progress->writeMessage("\nsecond pass - generating output\n");
	progress->update();

	read();
	while (doLoop(input)) {
		executor->inputData(mReader.frameIndex, mBufferFrame);

		const AffineTransform& tf = mTrajectory.computeSmoothTransform(mData, mWriter.frameIndex);
		executor->outputData(mWriter.frameIndex, tf);
		mWriter.prepareOutput(*executor);
		mWriter.write(*executor);
		read();

		//check if there is input on the console
		input.checkState();
		progress->update();
	}

	loopTerminate(progress, input, auxWriters, executor);
}
