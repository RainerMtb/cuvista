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
#include "MovieReader.hpp"
#include "Trajectory.hpp"
#include "FrameExecutor.hpp"


//-----------------------------------------------------------------------------------
class MovieWriterCollection : public NullWriter {

private:
	std::vector<std::shared_ptr<MovieWriter>> writers;
	std::vector<int> hasFrames;

	void updateStats();

public:
	MovieWriterCollection(MainData& data, MovieReader& reader, std::vector<std::shared_ptr<MovieWriter>> writers);

	void open(OutputOption outputOption) override;
	void start() override;
	void writeInput(const FrameExecutor& executor) override;
	void writeOutput(const FrameExecutor& executor) override;
	bool flush() override;
	void close() override;
};


//-----------------------------------------------------------------------------------
class OutputWriter : public NullWriter {

protected:
	ImageVuyx outputFrame; //frame to get from main loop and to send to output

public:
	OutputWriter(MainData& data, MovieReader& reader, int outputStride);
	OutputWriter(MainData& data, MovieReader& reader);

	const ImageVuyx& getOutputFrame();
	void writeOutput(const FrameExecutor& executor) override;
};


//-----------------------------------------------------------------------------------
class RawMemoryStoreWriter : public MovieWriter {

protected:
	size_t maxFrameCount;
	int inputFrameIndex = 0;
	bool doWriteInput;
	bool doWriteOutput;

public:
	std::list<ImageVuyx> outputFramesYuv;
	std::list<ImageRGBA> outputFramesRgba;
	std::list<ImageBGRA> outputFramesBgra;
	std::list<ImageVuyx> inputFrames;
	std::list<std::vector<PointResult>> results;

	RawMemoryStoreWriter(size_t maxFrameCount = 250, bool writeInput = true, bool writeOutput = true);

	void writeOutput(const FrameExecutor& executor) override;
	void writeInput(const FrameExecutor& executor) override;
	
	void writeYuvFiles(const std::string& inputFile, const std::string& outputFile, int maxFrames);
	void writeInputFile(const std::string& inputFile, int maxFrames, ThreadPoolBase& pool = defaultPool);
	void writeOutputFile(const std::string& outputFile, int maxFrames, ThreadPoolBase& pool = defaultPool);
};


//-----------------------------------------------------------------------------------
class BmpImageWriter : public ImageWriter {

private:
	ImageBGRA imageBgra;
	ImageStretcher stretcher;
	std::jthread worker;

public:
	BmpImageWriter(MainData& data, MovieReader& reader);

	void close() override;
	void writeOutput(const FrameExecutor& executor) override;
};


//-----------------------------------------------------------------------------------
class RawNv12Writer : public NullWriter {

private:
	std::ofstream file;
	ImageNV12 nv12;

public:
	RawNv12Writer(MainData& data, MovieReader& reader);

	void open(OutputOption outputOption) override;
	void writeOutput(const FrameExecutor& executor) override;
};


//-----------------------------------------------------------------------------------
class RawYuvWriter : public NullWriter {

private:
	std::ofstream file;
	ImageYuv yuv;

public:
	RawYuvWriter(MainData& data, MovieReader& reader);

	void open(OutputOption outputOption) override;
	void writeOutput(const FrameExecutor& executor) override;
};


//-----------------------------------------------------------------------------------
class RawPipeWriter : public NullWriter, public PipeWriter {

private:
	ImageYuv output;

public:
	RawPipeWriter(MainData& data, MovieReader& reader);

	void open(OutputOption outputOption) override;
	void writeOutput(const FrameExecutor& executor) override;
	void close() override;
};


//------------- transformation file -------------------------------------------------
class TransformsFile {

protected:
	inline static std::string id = "CUVI";
	std::ofstream mFile;

	template <class T> void writeValue(T val) {
		mFile.write(reinterpret_cast<const char*>(&val), sizeof(val));
	}

public:
	static std::map<int64_t, TransformValues> readTransformMap(const std::string& trajectoryFile);

	void open(const std::string& trajectoryFile);
	void writeTransform(const Affine2D& transform, int64_t frameIndex);
};


//--------------- write transforms --------------------------------------------------
class TransformsWriter : public MainWriter, public TransformsFile {

public:
	TransformsWriter(MainData& data);

	void start() override;
	void writeInput(const FrameExecutor& executor) override;
};


//--------------- write point results as large text file ----------------------------
class ResultDetailsWriter : public MainWriter {

private:
	std::string mDelim = ";";
	std::ofstream mFile;

	void write(std::span<PointResult> results, int64_t frameIndex);

public:
	ResultDetailsWriter(MainData& data);

	static void write(std::span<PointResult> results, const std::string& filename);

	void start() override;
	void writeInput(const FrameExecutor& executor) override;
};


//--------------- write individual images to show point results ---------------------
class ResultImageWriter : public MainWriter {

private:
	ImageVuyx yuv;
	ImageBGRA bgra;

public:
	ResultImageWriter(MainData& data);
	ResultImageWriter(MainData& data, MovieReader& reader);

	void start() override {}
	void writeInput(const FrameExecutor& executor) override;

	static void writeImage(const FrameResultData& resultData, std::span<PointResult> res, int64_t idx, Image8& dest, ThreadPoolBase& pool = defaultPool, bool drawTransformed = true);
};


//--------------- write individual images to show point results ---------------------
class ResultVideoWriter : public MainWriter {

private:
	std::ofstream file;
	ImageNV12 nv12;
	ImageBGRA bgra;

public:
	ResultVideoWriter(MainData& data, MovieReader& reader);

	void open(OutputOption outputOption) override;
	void writeInput(const FrameExecutor& executor) override;
};


//--------------------------------------------------------------------------------------
//simple writer for debugging, writes YUV data to file, cannot be used in executor loop
class SimpleYuvWriter {

private:
	std::ofstream os;

public:
	SimpleYuvWriter(const std::string& file);

	void write(ImageVuyx& image);
};