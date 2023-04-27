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

//combined loop to read and write
void MovieFrame::DeshakerLoopCombined::run(MovieFrame& mf, ProgressDisplay& progress, MovieReader& reader, MovieWriter& writer, UserInput& input) {
	const MainData& data = mf.mData;
	Stats& status = mf.mStatus;

	//init
	if (errorLogger.hasNoError()) reader.read(mf.bufferFrame, status);
	if (errorLogger.hasNoError()) data.showIntro();
	progress.init();
	progress.update();

	//first pyramid
	if (errorLogger.hasNoError() && status.endOfInput == false) {
		mf.inputData(mf.bufferFrame); //input first frame
		mf.createPyramid();
		status.frameReadIndex++;
		reader.read(mf.bufferFrame, status); //read second frame

		status.frameInputIndex++;
		progress.update();
	}

	//fill input buffer and compute transformations
	while (status.doContinue() && input.doContinue() && status.frameReadIndex < data.maxFrames && status.frameInputIndex + 1 < data.bufferCount) {
		mf.inputData(mf.bufferFrame);
		mf.createPyramid();
		mf.computeStart();
		mf.mFrameResult.computeTransform(mf.resultPointsOld, data, mf.mPool, data.rng.get());
		mf.mTrajectory.addTrajectoryTransform(mf.mFrameResult.mTransform, status.frameInputIndex - 1);
		mf.runDiagnostics(status.frameInputIndex - 1);

		mf.computeTerminate();
		std::swap(mf.resultPointsOld, mf.resultPoints);

		status.frameReadIndex++;
		reader.read(mf.bufferFrame, status);
		status.frameInputIndex++;
		progress.update();

		//check if there is input on the console
		input.setCurrentState();
	}

	//main loop
	//read data and compute frame results
	while (status.doContinue() && input.doContinue() && status.frameReadIndex < data.maxFrames) {
		TIMER("frame");
		std::vector<std::future<void>> futures;
		std::swap(mf.bufferFrame, mf.inputFrame);
		auto funcRead = [&] { 
			TIMER("read"); 
			status.frameReadIndex++;
			reader.read(mf.bufferFrame, status); 
		};
		futures.push_back(std::async(std::launch::async, funcRead));

		mf.inputData(mf.inputFrame);
		mf.createPyramid();
		mf.computeStart();
		mf.mFrameResult.computeTransform(mf.resultPointsOld, data, mf.mPool, data.rng.get());
		mf.mTrajectory.addTrajectoryTransform(mf.mFrameResult.mTransform, status.frameInputIndex - 1);
		mf.runDiagnostics(status.frameInputIndex - 1);

		assert(status.frameInputIndex % data.bufferCount != status.frameWriteIndex % data.bufferCount && "accessing the same buffer for read and write");
		const AffineTransform finalTransform = mf.mTrajectory.computeTransformForFrame(data, status.frameWriteIndex);
		mf.outputData(finalTransform, writer.getOutputData());
		auto funcWrite = [&] { 
			TIMER("write");
			writer.write(); 
		};
		futures.push_back(std::async(std::launch::async, funcWrite));
		mf.computeTerminate();

		futures.clear(); //wait for futures to terminate
		std::swap(mf.resultPointsOld, mf.resultPoints);
		status.frameWriteIndex++;
		status.frameInputIndex++;
		progress.update();

		//check if there is input on the console
		input.setCurrentState();
	}

	//process last frame in buffer
	if (errorLogger.hasNoError() && input.current <= UserInputEnum::END) {
		mf.mFrameResult.computeTransform(mf.resultPointsOld, data, mf.mPool, data.rng.get());
		mf.mTrajectory.addTrajectoryTransform(mf.mFrameResult.mTransform, status.frameInputIndex - 1);
		mf.runDiagnostics(status.frameInputIndex - 1);

		assert(status.frameInputIndex % data.bufferCount != status.frameWriteIndex % data.bufferCount && "accessing the same buffer for read and write");
		const AffineTransform finalTransform = mf.mTrajectory.computeTransformForFrame(data, status.frameWriteIndex);
		mf.outputData(finalTransform, writer.getOutputData());
		writer.write();
		status.frameWriteIndex++;
		progress.update();
	}

	//write remaining frames
	while (errorLogger.hasNoError() && status.frameWriteIndex < status.frameInputIndex && input.current <= UserInputEnum::END) {
		assert(status.frameInputIndex % data.bufferCount != status.frameWriteIndex % data.bufferCount && "accessing the same buffer for read and write");
		const AffineTransform tf = mf.mTrajectory.computeTransformForFrame(data, status.frameWriteIndex);
		mf.outputData(tf, writer.getOutputData());
		writer.write();
		status.frameWriteIndex++;
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

//fist pass to read and analyze--------------------------------
void MovieFrame::DeshakerLoopFirst::run(MovieFrame& mf, ProgressDisplay& progress, MovieReader& reader, MovieWriter& writer, UserInput& input) {
	const MainData& data = mf.mData;
	Stats& status = mf.mStatus;

	//init
	if (errorLogger.hasNoError()) reader.read(mf.bufferFrame, status);
	if (errorLogger.hasNoError()) data.showIntro();
	progress.init();
	progress.update();

	while (status.doContinue() && input.doContinue() && status.frameReadIndex < data.maxFrames) {
		mf.inputData(mf.bufferFrame);
		mf.createPyramid();
		status.frameReadIndex++;
		reader.read(mf.bufferFrame, status); 

		mf.computeStart();
		mf.computeTerminate();
		mf.mFrameResult.computeTransform(mf.resultPoints, data, mf.mPool, data.rng.get());
		mf.runDiagnostics(status.frameInputIndex);

		status.frameInputIndex++;
		status.frameWriteIndex++;
		progress.update();

		//check if there is input on the console
		input.setCurrentState();
	}
	progress.update(true);
}

//second pass to transform video--------------------------------
void MovieFrame::DeshakerLoopSecond::run(MovieFrame& mf, ProgressDisplay& progress, MovieReader& reader, MovieWriter& writer, UserInput& input) {
	//setup list of transforms from file
	auto map = mf.readTransforms();
	mf.mTrajectory.readTransforms(map);
	const MainData& data = mf.mData;
	Stats& status = mf.mStatus;

	//init
	if (errorLogger.hasNoError()) reader.read(mf.bufferFrame, status);
	if (errorLogger.hasNoError()) data.showIntro();
	progress.init();
	progress.update();

	//looping for output
	while (status.doContinue() && input.doContinue() && status.frameReadIndex < data.maxFrames) {
		mf.inputData(mf.bufferFrame);
		status.frameReadIndex++;
		status.frameInputIndex++;
		reader.read(mf.bufferFrame, status);

		const AffineTransform tf = mf.mTrajectory.computeTransformForFrame(data, status.frameWriteIndex);
		mf.outputData(tf, writer.getOutputData());
		writer.write();

		status.frameWriteIndex++;
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

//---------------------------------
//run first pass then second pass
//---------------------------------
void MovieFrame::DeshakerLoopClassic::run(MovieFrame& mf, ProgressDisplay& progress, MovieReader& reader, MovieWriter& writer, UserInput& input) {
	const MainData& data = mf.mData;
	Stats& status = mf.mStatus;

	//init
	if (errorLogger.hasNoError()) reader.read(mf.bufferFrame, status);
	if (errorLogger.hasNoError()) data.showIntro();
	progress.init();
	progress.writeMessage("first pass - analyzing input\n");
	progress.update();

	//first run - analyse
	while (status.doContinue() && input.doContinue() && status.frameReadIndex < data.maxFrames) {
		mf.inputData(mf.bufferFrame);
		mf.createPyramid();
		status.frameReadIndex++;
		reader.read(mf.bufferFrame, status);

		mf.computeStart();
		mf.computeTerminate();
		mf.mFrameResult.computeTransform(mf.resultPoints, data, mf.mPool, data.rng.get());
		mf.mTrajectory.addTrajectoryTransform(mf.mFrameResult.mTransform, status.frameInputIndex);
		mf.runDiagnostics(status.frameInputIndex);

		status.frameInputIndex++;
		progress.update();

		//check if there is input on the console
		input.setCurrentState();
	}
	progress.update(true);

	status.reset();
	//rewind input
	reader.rewind();
	//second run - compute transform and write output
	progress.writeMessage("\nsecond pass - generating output\n");
	progress.update();

	reader.read(mf.bufferFrame, status);
	while (status.doContinue() && input.doContinue() && status.frameReadIndex < data.maxFrames) {
		mf.inputData(mf.bufferFrame);
		status.frameReadIndex++;
		status.frameInputIndex++;
		reader.read(mf.bufferFrame, status);

		const AffineTransform tf = mf.mTrajectory.computeTransformForFrame(data, status.frameWriteIndex);
		mf.outputData(tf, writer.getOutputData());
		writer.write();

		status.frameWriteIndex++;
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