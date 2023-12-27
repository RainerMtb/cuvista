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

#include "FFmpegUtil.hpp"


class ReaderStats {
public:
	int h = 0, w = 0;
	int fpsNum = -1, fpsDen = -1;
	int64_t timeBaseNum = -1, timeBaseDen = -1;
	int64_t avformatDuration = -1;
	std::string_view sourceName;

	int64_t frameIndex = -1;
	int64_t frameCount = -1;
	bool endOfInput = true;

	AVStream* videoStream = nullptr;

	double fps() const;
	StreamInfo videoStreamInfo() const;
	StreamInfo streamInfo(AVStream* stream) const;
};


class WriterStats {
public:
	int64_t frameIndex = 0;
	int frameEncoded = 0;
	int64_t encodedBytesTotal = 0;
	int64_t outputBytesWritten = 0;

	int64_t encodedDts = 0;
	int64_t encodedPts = 0;
};