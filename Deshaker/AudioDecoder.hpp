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

#include <optional>
#include <mutex>

#include "FFmpegUtil.hpp"

class AudioDecoder {

protected:
	std::mutex mMutex;
	std::vector<uint8_t> mBuffer;
	StreamContext* mStreamCtx = nullptr;

	int64_t mBytesPerSample = 0;
	int64_t mBufferLimit = 0;
	int64_t mPlayerLimit = 0;

	size_t mWriteIndex = 0;
	int64_t mSamplesWritten = 0;

	int64_t mReadIndex = 0;
	int64_t mSamplesRead = 0;

public:
	void openFFmpeg(StreamContext* sc, double audioBufferSecs);
	void decodePackets();
	void setAudioLimit(std::optional<int64_t> millis);
};