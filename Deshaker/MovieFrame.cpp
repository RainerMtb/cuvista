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

MovieFrame::~MovieFrame() {
	mPool.shutdown(); //shutdown threads
}

const AffineTransform& MovieFrame::computeTransform(std::vector<PointResult> resultPoints) {
	return mFrameResult.computeTransform(resultPoints, mData, mPool, mData.rng.get());
}

void MovieFrame::outputSecondary(Writers& writers, int64_t frameIndex) {
	for (auto& writer : writers) {
		writer->write(this, frameIndex);
	}
}

std::map<int64_t, TransformValues> MovieFrame::readTransforms() {
	return TransformsWriter::readTransformMap(mData.trajectoryFile);
}


//---------------------------------------------------------------------
//---------- DUMMY FRAME ----------------------------------------------
//---------------------------------------------------------------------

void DummyFrame::inputData(ImageYuv& frame) {
	size_t idx = mStatus.frameInputIndex % frames.size();
	frames[idx] = frame;
}

void DummyFrame::outputData(const AffineTransform& trf, OutputContext outCtx) {
	ImageYuv& frameToEncode = frames[mStatus.frameWriteIndex % frames.size()];

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

void MovieFrame::runLoop(DeshakerPass pass, ProgressDisplay& progress, MovieReader& reader, MovieWriter& writer, UserInput& input, Writers& writers) {
	if (pass == DeshakerPass::COMBINED) runLoopCombined(progress, reader, writer, input, writers);
	else if (pass == DeshakerPass::FIRST_PASS) runLoopFirst(progress, reader, writer, input, writers);
	else if (pass == DeshakerPass::SECOND_PASS) runLoopSecond(progress, reader, writer, input, writers);
	else if (pass == DeshakerPass::CONSECUTIVE) runLoopConsecutive(progress, reader, writer, input, writers);
	else throw AVException("loop not implemented");
}

void MovieFrame::runLoopCombined(ProgressDisplay& progress, MovieReader& reader, MovieWriter& writer, UserInput& input, Writers& secondaryWriters) {
	//init
	if (errorLogger.hasNoError()) reader.read(bufferFrame, mStatus);
	if (errorLogger.hasNoError() && mData.showHeader) mData.showIntro(name());
	progress.init();
	progress.update();

	//first pyramid
	if (errorLogger.hasNoError() && mStatus.endOfInput == false) {
		inputData(bufferFrame); //input first frame
		createPyramid();
		mStatus.frameReadIndex++;
		reader.read(bufferFrame, mStatus); //read second frame

		mStatus.frameInputIndex++;
		progress.update();
	}

	//fill input buffer and compute transformations
	while (mStatus.doContinue() && input.doContinue() && mStatus.frameReadIndex < mData.maxFrames && mStatus.frameInputIndex + 1 < mData.bufferCount) {
		inputData(bufferFrame);
		createPyramid();
		computeTransform(resultPointsOld);
		mTrajectory.addTrajectoryTransform(mFrameResult.mTransform, mStatus.frameInputIndex - 1);
		outputSecondary(secondaryWriters, mStatus.frameInputIndex - 1);

		computeStart();
		computeTerminate();
		std::swap(resultPointsOld, resultPoints);

		mStatus.frameReadIndex++;
		reader.read(bufferFrame, mStatus);
		mStatus.frameInputIndex++;
		progress.update();

		//check if there is input on the console
		input.setCurrentState();
	}

	//main loop
	//read data and compute frame results
	while (mStatus.doContinue() && input.doContinue() && mStatus.frameReadIndex < mData.maxFrames) {
		//util::ConsoleTimer t_fr("frame");
		assert(mStatus.frameInputIndex % mData.bufferCount != mStatus.frameWriteIndex % mData.bufferCount && "accessing the same buffer for read and write");
		std::vector<std::future<void>> futures;
		std::swap(bufferFrame, inputFrame);
		mStatus.frameReadIndex++;
		futures.push_back(reader.readAsync(bufferFrame, mStatus));

		inputData(inputFrame);
		createPyramid();
		computeStart();
		computeTransform(resultPointsOld);
		mTrajectory.addTrajectoryTransform(mFrameResult.mTransform, mStatus.frameInputIndex - 1);
		const AffineTransform& finalTransform = mTrajectory.computeTransformForFrame(mData, mStatus.frameWriteIndex);
		outputData(finalTransform, writer.getOutputContext());
		outputSecondary(secondaryWriters, mStatus.frameInputIndex - 1);
		futures.push_back(writer.writeAsync());

		computeTerminate();

		futures.clear(); //wait for futures to terminate
		std::swap(resultPointsOld, resultPoints);
		mStatus.frameWriteIndex++;
		mStatus.frameInputIndex++;
		progress.update();

		//check if there is input on the console
		input.setCurrentState();
	}

	//process last frame in buffer
	if (errorLogger.hasNoError() && input.current <= UserInputEnum::END) {
		computeTransform(resultPointsOld);
		mTrajectory.addTrajectoryTransform(mFrameResult.mTransform, mStatus.frameInputIndex - 1);

		assert(mStatus.frameInputIndex % mData.bufferCount != mStatus.frameWriteIndex % mData.bufferCount && "accessing the same buffer for read and write");
		const AffineTransform& finalTransform = mTrajectory.computeTransformForFrame(mData, mStatus.frameWriteIndex);
		outputData(finalTransform, writer.getOutputContext());
		outputSecondary(secondaryWriters, mStatus.frameInputIndex - 1);
		writer.write();
		mStatus.frameWriteIndex++;
		progress.update();
	}

	//write remaining frames
	while (errorLogger.hasNoError() && mStatus.frameWriteIndex < mStatus.frameInputIndex && input.current <= UserInputEnum::END) {
		assert(mStatus.frameInputIndex % mData.bufferCount != mStatus.frameWriteIndex % mData.bufferCount && "accessing the same buffer for read and write");
		const AffineTransform& tf = mTrajectory.computeTransformForFrame(mData, mStatus.frameWriteIndex);
		outputData(tf, writer.getOutputContext());
		writer.write();
		mStatus.frameWriteIndex++;
		progress.update();

		//check if there is input on the console
		input.setCurrentState();
	}

	//flush writer buffer
	bool hasFrame = writer.terminate(true);
	while (errorLogger.hasNoError() && input.current < UserInputEnum::HALT && hasFrame) {
		hasFrame = writer.terminate();
		progress.update();
	}
	progress.update(true);
}

void MovieFrame::runLoopFirst(ProgressDisplay& progress, MovieReader& reader, MovieWriter& writer, UserInput& input, Writers& secondaryWriters) {
	//init
	if (errorLogger.hasNoError()) reader.read(bufferFrame, mStatus);
	if (errorLogger.hasNoError() && mData.showHeader) mData.showIntro(name());
	progress.init();
	progress.update();

	while (mStatus.doContinue() && input.doContinue() && mStatus.frameReadIndex < mData.maxFrames) {
		inputData(bufferFrame);
		createPyramid();
		mStatus.frameReadIndex++;
		reader.read(bufferFrame, mStatus);

		computeStart();
		computeTerminate();
		computeTransform(resultPoints);
		outputSecondary(secondaryWriters, mStatus.frameInputIndex);

		mStatus.frameInputIndex++;
		mStatus.frameWriteIndex++;
		progress.update();

		//check if there is input on the console
		input.setCurrentState();
	}
	progress.update(true);
}

void MovieFrame::runLoopSecond(ProgressDisplay& progress, MovieReader& reader, MovieWriter& writer, UserInput& input, Writers& secondaryWriters) {
	//setup list of transforms from file
	auto map = readTransforms();
	mTrajectory.readTransforms(map);

	//init
	if (errorLogger.hasNoError()) reader.read(bufferFrame, mStatus);
	if (errorLogger.hasNoError() && mData.showHeader) mData.showIntro(name());
	progress.init();
	progress.update();

	//looping for output
	while (mStatus.doContinue() && input.doContinue() && mStatus.frameReadIndex < mData.maxFrames) {
		inputData(bufferFrame);
		mStatus.frameReadIndex++;
		mStatus.frameInputIndex++;
		reader.read(bufferFrame, mStatus);

		const AffineTransform& tf = mTrajectory.computeTransformForFrame(mData, mStatus.frameWriteIndex);
		outputData(tf, writer.getOutputContext());
		writer.write();

		mStatus.frameWriteIndex++;
		progress.update();

		//check if there is input on the console
		input.setCurrentState();
	}

	//flush writer buffer
	bool hasFrame = writer.terminate(true);
	while (errorLogger.hasNoError() && input.current < UserInputEnum::HALT && hasFrame) {
		hasFrame = writer.terminate();
		progress.update();
	}
	progress.update(true);
}

void MovieFrame::runLoopConsecutive(ProgressDisplay& progress, MovieReader& reader, MovieWriter& writer, UserInput& input, Writers& secondaryWriters) {
	//init
	if (errorLogger.hasNoError()) reader.read(bufferFrame, mStatus);
	if (errorLogger.hasNoError() && mData.showHeader) mData.showIntro(name());
	progress.init();
	progress.writeMessage("first pass - analyzing input\n");
	progress.update();

	//first run - analyse
	while (mStatus.doContinue() && input.doContinue() && mStatus.frameReadIndex < mData.maxFrames) {
		inputData(bufferFrame);
		createPyramid();
		mStatus.frameReadIndex++;
		reader.read(bufferFrame, mStatus);

		computeStart();
		computeTerminate();
		computeTransform(resultPoints);
		mTrajectory.addTrajectoryTransform(mFrameResult.mTransform, mStatus.frameInputIndex);
		outputSecondary(secondaryWriters, mStatus.frameInputIndex);

		mStatus.frameInputIndex++;
		progress.update();

		//check if there is input on the console
		input.setCurrentState();
	}
	progress.update(true);

	mStatus.reset();
	//rewind input
	reader.rewind();
	//second run - compute transform and write output
	progress.writeMessage("\nsecond pass - generating output\n");
	progress.update();

	reader.read(bufferFrame, mStatus);
	while (mStatus.doContinue() && input.doContinue() && mStatus.frameReadIndex < mData.maxFrames) {
		inputData(bufferFrame);
		mStatus.frameReadIndex++;
		mStatus.frameInputIndex++;
		reader.read(bufferFrame, mStatus);

		const AffineTransform& tf = mTrajectory.computeTransformForFrame(mData, mStatus.frameWriteIndex);
		outputData(tf, writer.getOutputContext());
		writer.write();

		mStatus.frameWriteIndex++;
		progress.update();

		//check if there is input on the console
		input.setCurrentState();
	}

	//flush writer buffer
	bool hasFrame = writer.terminate(true);
	while (errorLogger.hasNoError() && input.current < UserInputEnum::HALT && hasFrame) {
		hasFrame = writer.terminate();
		progress.update();
	}
	progress.update(true);
}
