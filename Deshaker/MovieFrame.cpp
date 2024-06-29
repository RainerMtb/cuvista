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
#include "ProgressDisplay.hpp"
#include "Util.hpp"

std::string MovieFrame::ptsForFrameString(int64_t frameIndex) {
	return mReader.ptsForFrameString(frameIndex).value_or("");
}

//shutdown
MovieFrame::~MovieFrame() {
	mPool.shutdown(); //shutdown threads
}

//compute affine transform parameters given the indiviual point results
void MovieFrame::computeTransform(int64_t frameIndex) {
	mFrameResult.computeTransform(mResultPoints, mPool, frameIndex, rng.get());
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
	return errorLogger.hasNoError() && input.doContinue() && mReader.endOfInput == false && mReader.frameIndex < mData.maxFrames;
}


//---------------------------------------------------------------------
//---------- DUMMY FRAME ----------------------------------------------
//---------------------------------------------------------------------

void DummyFrame::inputData() {
	size_t idx = mBufferFrame.index % mFrames.size();
	mBufferFrame.copyTo(mFrames[idx], mPool);
}

void DummyFrame::outputData(const AffineTransform& trf) {
	size_t idx = trf.frameIndex % mFrames.size();
	assert(mFrames[idx].index == trf.frameIndex && "invalid frame index to output");
}

void DummyFrame::getOutput(int64_t frameIndex, ImageYuv& image) {
	size_t idx = frameIndex % mFrames.size();
	mFrames[idx].copyTo(image, mPool);
}

void DummyFrame::getOutput(int64_t frameIndex, unsigned char* cudaNv12ptr, int cudaPitch) {
	static std::vector<unsigned char> nv12(cudaPitch * mData.h * 3 / 2);
	size_t idx = frameIndex % mFrames.size();
	mFrames[idx].toNV12(nv12, cudaPitch, mPool);
	encodeNvData(nv12, cudaNv12ptr);
}

void DummyFrame::getOutput(int64_t frameIndex, ImageRGBA& image) {
	size_t idx = frameIndex % mFrames.size();
	mFrames[idx].toRGBA(image, mPool);
}

void DummyFrame::getWarped(int64_t frameIndex, ImageRGBA& image) {
	size_t idx = frameIndex % mFrames.size();
	mFrames[idx].toRGBA(image, mPool);
}

void DummyFrame::getInput(int64_t frameIndex, ImageYuv& image) const {
	size_t idx = frameIndex % mFrames.size();
	mFrames[idx].copyTo(image);
}


//---------------------------------------------------------------------
//---------- DESHAKER LOOPS  ------------------------------------------
//---------------------------------------------------------------------

void MovieFrame::runLoop(DeshakerPass pass, ProgressBase& progress, UserInput& input, AuxWriters& auxWriters) {
	if (pass == DeshakerPass::COMBINED) runLoopCombined(progress, input, auxWriters);
	else if (pass == DeshakerPass::FIRST_PASS) runLoopFirst(progress, input, auxWriters);
	else if (pass == DeshakerPass::SECOND_PASS) runLoopSecond(progress, input, auxWriters);
	else if (pass == DeshakerPass::CONSECUTIVE) runLoopConsecutive(progress, input, auxWriters);
	else throw AVException("loop not implemented");
}

void MovieFrame::loopInit(ProgressBase& progress, const std::string& message) {
	//read first frame from input into buffer
	if (errorLogger.hasNoError()) read();
	//show program header on console
	if (errorLogger.hasNoError() && mData.showHeader) mData.showIntro(getId().nameLong, mReader);
	//init progress display
	progress.init();
	progress.writeMessage(message);
	progress.update();

	//first pyramid
	if (errorLogger.hasNoError() && mReader.endOfInput == false) {
		inputData(); //input first frame
		createPyramid(mReader.frameIndex);
		read(); //read second frame
		progress.update();
	}
}

void MovieFrame::loopTerminate(ProgressBase& progress, UserInput& input, AuxWriters& auxWriters) {
	//flush writer buffer
	bool hasFrame = mWriter.startFlushing();
	while (errorLogger.hasNoError() && input.mCurrentInput < UserInputEnum::HALT && hasFrame) {
		hasFrame = mWriter.flush();
		progress.update();
	}
	progress.forceUpdate();
}

//loop to handle input - stabilization - output in one go
void MovieFrame::runLoopCombined(ProgressBase& progress, UserInput& input, AuxWriters& auxWriters) {
	loopInit(progress);

	//fill input buffer and compute transformations
	while (doLoop(input) && mReader.frameIndex < mData.bufferCount - 1LL) {
		//process current frame
		inputData();
		createPyramid(mReader.frameIndex);
		//transform for previous frame
		computeTransform(mReader.frameIndex - 1);
		mTrajectory.addTrajectoryTransform(mFrameResult.getTransform());
		auxWriters.writeAll(*this);
		//compute flow for current frame
		computeStart(mReader.frameIndex);
		computeTerminate(mReader.frameIndex);
		//read next frame into buffer
		read();

		//check if there is input on the console
		input.checkState();
		progress.update();
	}

	//main loop
	//read data and compute frame results
	while (doLoop(input)) {
		//util::ConsoleTimer timer_fr("frame");
		int64_t readIndex = mReader.frameIndex;
		assert(readIndex % mData.bufferCount != mWriter.frameIndex % mData.bufferCount && "accessing the same buffer for read and write");
		
		//process current frame
		inputData();
		createPyramid(readIndex);
		computeStart(readIndex);
		//read next frame async
		std::future<void> futRead = readAsync();
		
		//now compute transform for previous frame while results for current frame are potentially computed on device
		computeTransform(readIndex - 1);
		const AffineTransform& currentTransform = mFrameResult.getTransform();
		const TrajectoryItem& item = mTrajectory.addTrajectoryTransform(currentTransform);
		const AffineTransform& finalTransform = mTrajectory.computeSmoothTransform(mData, mWriter.frameIndex);
		outputData(finalTransform);
		mWriter.prepareOutput(*this);
		
		//write output
		mWriter.write(*this);
		auxWriters.writeAll(*this);
		
		//get computed flow for current frame
		computeTerminate(readIndex);

		//wait for async to complete
		futRead.wait();

		//check if there is input on the console
		input.checkState();
		progress.update();
	}

	//process last frame in buffer
	if (errorLogger.hasNoError() && input.mCurrentInput <= UserInputEnum::END) {
		computeTransform(mReader.frameIndex);
		mTrajectory.addTrajectoryTransform(mFrameResult.getTransform());
		const AffineTransform& finalTransform = mTrajectory.computeSmoothTransform(mData, mWriter.frameIndex);
		outputData(finalTransform);
		mWriter.prepareOutput(*this);
		mWriter.write(*this);
		auxWriters.writeAll(*this);
		progress.update();
	}

	//write remaining frames
	while (errorLogger.hasNoError() && mWriter.frameIndex < mReader.frameIndex && input.mCurrentInput <= UserInputEnum::END) {
		const AffineTransform& tf = mTrajectory.computeSmoothTransform(mData, mWriter.frameIndex);
		outputData(tf);
		mWriter.prepareOutput(*this);
		mWriter.write(*this);

		//check if there is input on the console
		input.checkState();
		progress.update();
	}

	loopTerminate(progress, input, auxWriters);
}

void MovieFrame::runLoopFirst(ProgressBase& progress, UserInput& input, AuxWriters& auxWriters) {
	mReader.storePackets = false; //do not keep any secondary packets when we do not write output anyway
	loopInit(progress);
	mFrameResult.reset();
	mTrajectory.addTrajectoryTransform(mFrameResult.getTransform()); //first frame has no transform applied
	mWriter.write(*this);
	auxWriters.writeAll(*this);

	while (doLoop(input)) {
		inputData();
		createPyramid(mReader.frameIndex);

		computeStart(mReader.frameIndex);
		computeTerminate(mReader.frameIndex);
		computeTransform(mReader.frameIndex);
		mWriter.write(*this);
		auxWriters.writeAll(*this);
		read();

		//check if there is input on the console
		input.checkState();
		progress.update();
	}
	progress.forceUpdate();
}

void MovieFrame::runLoopSecond(ProgressBase& progress, UserInput& input, AuxWriters& auxWriters) {
	//setup list of transforms from file
	auto map = readTransforms();
	if (errorLogger.hasNoError()) mTrajectory.readTransforms(map);

	//init
	if (errorLogger.hasNoError()) read();
	if (errorLogger.hasNoError() && mData.showHeader) mData.showIntro(getId().nameShort, mReader);
	progress.init();
	progress.update();

	//looping for output
	while (doLoop(input)) {
		inputData();
		read();

		const AffineTransform& tf = mTrajectory.computeSmoothTransform(mData, mWriter.frameIndex);
		outputData(tf);
		mWriter.prepareOutput(*this);
		mWriter.write(*this);

		//check if there is input on the console
		input.checkState();
		progress.update();
	}

	loopTerminate(progress, input, auxWriters);
}

void MovieFrame::runLoopConsecutive(ProgressBase& progress, UserInput& input, AuxWriters& auxWriters) {
	mReader.storePackets = false; //we do not want packets to pile up on first iteration
	loopInit(progress, "first pass - analyzing input\n");
	mFrameResult.reset();
	mTrajectory.addTrajectoryTransform(mFrameResult.getTransform()); //first frame has no transform applied
	auxWriters.writeAll(*this);

	//first run - analyse
	while (doLoop(input)) {
		inputData();
		createPyramid(mReader.frameIndex);

		computeStart(mReader.frameIndex);
		computeTerminate(mReader.frameIndex);
		computeTransform(mReader.frameIndex);
		mTrajectory.addTrajectoryTransform(mFrameResult.getTransform());
		auxWriters.writeAll(*this);
		read();

		//check if there is input on the console
		input.checkState();
		progress.update();
	}
	progress.forceUpdate();

	//rewind input
	mReader.rewind();
	mReader.storePackets = true;
	progress.writeMessage("\nsecond pass - generating output\n");
	progress.update();

	read();
	while (doLoop(input)) {
		inputData();

		const AffineTransform& tf = mTrajectory.computeSmoothTransform(mData, mWriter.frameIndex);
		outputData(tf);
		mWriter.prepareOutput(*this);
		mWriter.write(*this);
		read();

		//check if there is input on the console
		input.checkState();
		progress.update();
	}

	loopTerminate(progress, input, auxWriters);
}
