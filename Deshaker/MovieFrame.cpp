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

MovieFrame::~MovieFrame() {
	mPool.shutdown(); //shutdown threads
}

void MovieFrame::computeTransform(int64_t frameIndex) {
	mFrameResult.computeTransform(mResultPoints, mPool, frameIndex);
}

std::map<int64_t, TransformValues> MovieFrame::readTransforms() {
	return TransformsWriterMain::readTransformMap(mData.trajectoryFile);
}

void MovieFrame::read() {
	mReader.read(mBufferFrame);
}

void MovieFrame::write() {
	mWriter.write(*this);
}


//---------------------------------------------------------------------
//---------- DUMMY FRAME ----------------------------------------------
//---------------------------------------------------------------------

void DummyFrame::inputData() {
	size_t idx = mBufferFrame.index % frames.size();
	frames[idx] = mBufferFrame;
}

void DummyFrame::outputData(const AffineTransform& trf, OutputContext outCtx) {
	ImageYuv& frameToEncode = frames[trf.frameIndex % frames.size()];
	assert(frameToEncode.index == trf.frameIndex && "invalid frame index to output");

	if (outCtx.encodeCpu) {
		*outCtx.outputFrame = frameToEncode;
	}

	if (outCtx.encodeCuda) {
		encodeNvData(frameToEncode.toNV12(outCtx.cudaPitch), outCtx.cudaNv12ptr);
	}
}

ImageYuv DummyFrame::getInput(int64_t index) const {
	return frames[index % frames.size()];
}


//---------------------------------------------------------------------
//---------- DESHAKER LOOPS  ------------------------------------------
//---------------------------------------------------------------------

void MovieFrame::runLoop(DeshakerPass pass, ProgressDisplay& progress, UserInput& input, AuxWriters& auxWriters) {
	if (pass == DeshakerPass::COMBINED) runLoopCombined(progress, input, auxWriters);
	else if (pass == DeshakerPass::FIRST_PASS) runLoopFirst(progress, input, auxWriters);
	else if (pass == DeshakerPass::SECOND_PASS) runLoopSecond(progress, input, auxWriters);
	else if (pass == DeshakerPass::CONSECUTIVE) runLoopConsecutive(progress, input, auxWriters);
	else throw AVException("loop not implemented");
}

void MovieFrame::loopInit(ProgressDisplay& progress, const std::string& message) {
	//read first frame from input into buffer
	if (errorLogger.hasNoError()) read();
	//show program header on console
	if (errorLogger.hasNoError() && mData.showHeader) mData.showIntro(className(), mReader);
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

void MovieFrame::loopTerminate(ProgressDisplay& progress, UserInput& input, AuxWriters& auxWriters) {
	//flush writer buffer
	bool hasFrame = mWriter.startFlushing();
	while (errorLogger.hasNoError() && input.current < UserInputEnum::HALT && hasFrame) {
		hasFrame = mWriter.flush();
		progress.update();
	}
	progress.update(true);
}

//loop to handle input - stabilization - output in one go
void MovieFrame::runLoopCombined(ProgressDisplay& progress, UserInput& input, AuxWriters& auxWriters) {
	loopInit(progress);

	//fill input buffer and compute transformations
	while (errorLogger.hasNoError() && mReader.endOfInput == false && input.doContinue() && mReader.frameIndex < mData.bufferCount - 1) {
		//process current frame
		inputData();
		createPyramid(mReader.frameIndex);
		//transform for previous frame
		computeTransform(mReader.frameIndex - 1);
		mTrajectory.addTrajectoryTransform(mFrameResult.transform());
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
	while (errorLogger.hasNoError() && mReader.endOfInput == false && input.doContinue()) {
		//util::ConsoleTimer t_fr("frame");
		int64_t readIndex = mReader.frameIndex;
		assert(readIndex % mData.bufferCount != mWriter.frameIndex % mData.bufferCount && "accessing the same buffer for read and write");
		
		//process current frame
		inputData();
		createPyramid(readIndex);
		computeStart(readIndex);
		//read next frame async
		std::future<void> futRead = mReader.readAsync(mBufferFrame);
		
		//now compute transform for previous frame while results for current frame are potentially computed on device
		computeTransform(readIndex - 1);
		mTrajectory.addTrajectoryTransform(mFrameResult.transform());
		const AffineTransform& finalTransform = mTrajectory.computeSmoothTransform(mData, mWriter.frameIndex);
		outputData(finalTransform, mWriter.getOutputContext());
		
		//write output
		std::future<void> futWrite = mWriter.writeAsync(*this);
		auxWriters.writeAll(*this);
		
		//get computed flow for current frame
		computeTerminate(readIndex);

		//wait for async to complete
		futRead.get();
		futWrite.get();

		//check if there is input on the console
		input.checkState();
		progress.update();
	}

	//process last frame in buffer
	if (errorLogger.hasNoError() && input.current <= UserInputEnum::END) {
		computeTransform(mReader.frameIndex);
		mTrajectory.addTrajectoryTransform(mFrameResult.transform());
		const AffineTransform& finalTransform = mTrajectory.computeSmoothTransform(mData, mWriter.frameIndex);
		outputData(finalTransform, mWriter.getOutputContext());
		write();
		auxWriters.writeAll(*this);
		progress.update();
	}

	//write remaining frames
	while (errorLogger.hasNoError() && mWriter.frameIndex < mReader.frameIndex && input.current <= UserInputEnum::END) {
		const AffineTransform& tf = mTrajectory.computeSmoothTransform(mData, mWriter.frameIndex);
		outputData(tf, mWriter.getOutputContext());
		write();

		//check if there is input on the console
		input.checkState();
		progress.update();
	}

	loopTerminate(progress, input, auxWriters);
}

void MovieFrame::runLoopFirst(ProgressDisplay& progress, UserInput& input, AuxWriters& auxWriters) {
	mReader.storePackets = false; //do not keep any secondary packets when we do not write output anyway
	loopInit(progress);
	mFrameResult.transformReset();
	mTrajectory.addTrajectoryTransform(mFrameResult.transform()); //first frame has no transform applied
	write();
	auxWriters.writeAll(*this);

	while (errorLogger.hasNoError() && mReader.endOfInput == false && input.doContinue()) {
		inputData();
		createPyramid(mReader.frameIndex);

		computeStart(mReader.frameIndex);
		computeTerminate(mReader.frameIndex);
		computeTransform(mReader.frameIndex);
		write();
		auxWriters.writeAll(*this);
		read();

		//check if there is input on the console
		input.checkState();
		progress.update();
	}
	progress.update(true);
}

void MovieFrame::runLoopSecond(ProgressDisplay& progress, UserInput& input, AuxWriters& auxWriters) {
	//setup list of transforms from file
	auto map = readTransforms();
	mTrajectory.readTransforms(map);

	//init
	if (errorLogger.hasNoError()) read();
	if (errorLogger.hasNoError() && mData.showHeader) mData.showIntro(className(), mReader);
	progress.init();
	progress.update();

	//looping for output
	while (errorLogger.hasNoError() && mReader.endOfInput == false && input.doContinue()) {
		inputData();
		read();

		const AffineTransform& tf = mTrajectory.computeSmoothTransform(mData, mWriter.frameIndex);
		outputData(tf, mWriter.getOutputContext());
		write();

		//check if there is input on the console
		input.checkState();
		progress.update();
	}

	loopTerminate(progress, input, auxWriters);
}

void MovieFrame::runLoopConsecutive(ProgressDisplay& progress, UserInput& input, AuxWriters& auxWriters) {
	mReader.storePackets = false; //we do not want packets to pile up on first iteration
	loopInit(progress, "first pass - analyzing input\n");
	mFrameResult.transformReset();
	mTrajectory.addTrajectoryTransform(mFrameResult.transform()); //first frame has no transform applied
	auxWriters.writeAll(*this);

	//first run - analyse
	while (errorLogger.hasNoError() && mReader.endOfInput == false && input.doContinue()) {
		inputData();
		createPyramid(mReader.frameIndex);

		computeStart(mReader.frameIndex);
		computeTerminate(mReader.frameIndex);
		computeTransform(mReader.frameIndex);
		mTrajectory.addTrajectoryTransform(mFrameResult.transform());
		auxWriters.writeAll(*this);
		read();

		//check if there is input on the console
		input.checkState();
		progress.update();
	}
	progress.update(true);

	//rewind input
	mReader.rewind();
	mReader.storePackets = true;
	progress.writeMessage("\nsecond pass - generating output\n");
	progress.update();

	read();
	while (errorLogger.hasNoError() && mReader.endOfInput == false && input.doContinue()) {
		inputData();

		const AffineTransform& tf = mTrajectory.computeSmoothTransform(mData, mWriter.frameIndex);
		outputData(tf, mWriter.getOutputContext());
		write();
		read();

		//check if there is input on the console
		input.checkState();
		progress.update();
	}

	loopTerminate(progress, input, auxWriters);
}
