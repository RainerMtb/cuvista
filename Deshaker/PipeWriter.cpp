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
#include <io.h>
#include <fcntl.h>

void PipeWriter::open() {
	//set stdout to binary mode
	int result = _setmode(_fileno(stdout), _O_BINARY);
	if (result < 0)
		throw AVException("Pipe: error setting stdout to binary");
}

void PipeWriter::write() {
	packYuv();
	size_t siz = fwrite(yuvPacked.data(), 1, yuvPacked.size(), stdout);
	if (siz != yuvPacked.size()) {
		errorLogger.logError("Pipe: error writing data");
	}
	mStatus.outputBytesWritten += siz;

	//static std::ofstream out("f:/test.yuv", std::ios::binary);
	//out.write(yuvPacked.data(), yuvPacked.size());
}

PipeWriter::~PipeWriter() {
	int result = _setmode(_fileno(stdout), _O_TEXT);
	if (result < 0) {
		errorLogger.logError("Pipe: error setting stdout back to text");
	}
}
