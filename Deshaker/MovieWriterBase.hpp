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

#include "Stats.hpp"
#include "FFmpegUtil.hpp"
#include "OutputOption.hpp"

class MovieReader;
class MainData;
class MovieFrame;


 //-----------------------------------------------------------------------------------
 // each writer must increment frame counter when beeing called to write
 //-----------------------------------------------------------------------------------

class MovieWriter : public WriterStats {

public:
	virtual ~MovieWriter() = default;
	virtual void open(OutputOption outputOption) {}
	virtual void start() {}
	virtual void writeInput(const FrameExecutor& executor) {}
	virtual void writeOutput(const FrameExecutor& executor) {}
	virtual bool flush() { return false; }
	virtual void close() {}
};


//-----------------------------------------------------------------------------------
class MovieWriterBase : public MovieWriter {

protected:
	const MainData& mData;

public:
	MovieWriterBase(MainData& data) :
		mData(data)
	{}
};


//-----------------------------------------------------------------------------------
class NullWriter : public MovieWriterBase {

protected:
	MovieReader& mReader;

public:
	NullWriter(MainData& data, MovieReader& reader) :
		MovieWriterBase(data),
		mReader { reader }
	{}

	void writeOutput(const FrameExecutor& executor) override;
};


//-----------------------------------------------------------------------------------
class ImageWriter : public NullWriter {

protected:
	ImageWriter(MainData& data, MovieReader& reader) :
		NullWriter(data, reader)
	{}

	std::string makeFilename(const std::string& extension) const;

public:
	static std::string makeFilename(const std::string& pattern, int64_t index, const std::string& extension);
	static std::string makeFilenameSamples(const std::string& pattern, const std::string& extension);
};


//-----------------------------------------------------------------------------------
class PipeWriter {

protected:
	virtual void openPipe();
	virtual void closePipe();
};
