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
#include "ErrorLogger.hpp"


//status of progress, singleton class
class Stats {

public:
	int64_t frameReadIndex;
	int64_t frameInputIndex;
	int64_t frameWriteIndex;
	int frameEncodeIndex;
	int64_t encodedBytes;
	int64_t encodedBytesTotal;
	int64_t outputBytesWritten;

	int64_t encodedDts;
	int64_t encodedPts;
	VideoPacketContext encodedFrame;

	std::list<VideoPacketContext> packetList;
	bool endOfInput;
	std::vector<StreamContext> inputStreams;

	//disable copies and assignments, moving is allowed
	Stats(const Stats& other) = delete;
	Stats(Stats&& other) = delete;
	Stats& operator = (const Stats& other) = delete;

	Stats() {
		reset();
	}

	void reset() {
		frameReadIndex = 0;
		frameInputIndex = 0;
		frameWriteIndex = 0;
		frameEncodeIndex = 0;
		encodedBytes = 0;
		encodedBytesTotal = 0;
		outputBytesWritten = 0;

		encodedDts = 0;
		encodedPts = 0;
		encodedFrame = {};

		packetList.clear();
		endOfInput = true;

		for (StreamContext& sc : inputStreams) {
			sc.packets.clear();
			sc.packetsWritten = 0;
		}
	}

	bool doContinue() const {
		return errorLogger.hasNoError() && endOfInput == false;
	}
};
