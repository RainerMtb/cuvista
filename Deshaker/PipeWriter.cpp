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
#include "MovieFrame.hpp"
#include <fcntl.h>

/*
* write raw video data via pipe to receiving program
* example usage:
* cuvista ... | ffmpeg -f rawvideo -pix_fmt yuv444p -video_size ww:hh -r xx -i pipe:0 -pix_fmt yuv420p outfile.mp4 -y
* make sure on the ffmpeg side the -i pipe:0 follows all the input format specifiers
* on windows do not use PowerShell!!
*/

#if defined(_WIN64)

//in windows we need to set pipe mode
#include <io.h>

void PipeWriter::open(EncodingOption videoCodec) {
	//set stdout to binary mode
	int result = _setmode(_fileno(stdout), _O_BINARY);
	if (result < 0) {
		throw AVException("Pipe: error setting stdout to binary mode");
	}
}

PipeWriter::~PipeWriter() {
	int result = _setmode(_fileno(stdout), _O_TEXT);
	if (result < 0) {
		errorLogger().logError("Pipe: error setting stdout to text mode", ErrorSource::WRITER);
	}
}

#else

void PipeWriter::open(EncodingOption videoCodec) {}

PipeWriter::~PipeWriter() {}

#endif

void PipeWriter::writeOutput(const FrameExecutor& executor) {
	const unsigned char* src = outputFrame.data();
	size_t bytes = 0;
	
	for (int i = 0; i < 3ull * outputFrame.h; i++) {
		bytes += fwrite(src, 1, outputFrame.w, stdout);
		src += outputFrame.stride;
	}
	
	if (bytes != 3ull * outputFrame.w * outputFrame.h) {
		errorLogger().logError("Pipe: error writing data", ErrorSource::WRITER);
	}

	std::unique_lock<std::mutex> lock(mStatsMutex);
	outputBytesWritten += bytes;
	this->frameIndex++;
}
