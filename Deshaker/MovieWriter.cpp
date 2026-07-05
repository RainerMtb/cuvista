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

#include "MovieWriter.hpp"
#include "MainData.hpp"

#include <filesystem>


void NullWriter::writeOutput(const FrameExecutor& executor) {
	this->frameIndex++;
}


//-----------------------------------------------------------------------------------
// Image Helpers
//-----------------------------------------------------------------------------------

std::string ImageWriter::makeFilename(const std::string& pattern, int64_t index, const std::string& extension) {
	namespace fs = std::filesystem;
	fs::path out;

	if (pattern.empty() == false && fs::is_directory(pattern)) {
		//file in the given directory
		std::string file = std::format("im{:04d}.{}", index, extension);
		out = fs::path(pattern) / fs::path(file);

	} else {
		//apply pattern as is
		const int siz = 512;
		char fname[siz];
		std::snprintf(fname, siz, pattern.c_str(), index);
		out = fs::path(fname);
	}
	return out.make_preferred().string();
}

std::string ImageWriter::makeFilenameSamples(const std::string& pattern, const std::string& extension) {
	std::string samples = "";
	std::string file = makeFilename(pattern, 0, extension);
	int idx = 0;
	while (samples.size() + file.size() < 100 && idx < 3) {
		samples += file;
		samples += ", ";
		idx++;
		file = makeFilename(pattern, idx, extension);
	}
	samples += "...";
	return samples;
}

std::string ImageWriter::makeFilename(const std::string& extension) const {
	return makeFilename(mData.fileOut, this->frameIndex, extension);
}
