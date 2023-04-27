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

#include <format>
#include "ProgressDisplayConsole.hpp"
#include "Util.hpp"

//print a new line to the console
void ProgressDisplayConsole::update(bool force) {
	if (isDue(force)) {
		double done = progressPercent();
		out << "frames in " << data.status.frameInputIndex << ", frames out " << data.status.frameWriteIndex;
		if (done >= 0) out << done << " %";
		out << std::endl;
	}
}

void ProgressDisplayConsole::writeMessage(const std::string& str) {
	*outstream << str;
}

ProgressDisplayConsole::ProgressDisplayConsole(MainData& data) : ProgressDisplay(data, 500), outstream { data.console } {
	out.precision(1);
	out << std::showpoint;
	out << std::fixed;
	timePoint = std::chrono::steady_clock::now();
}

//---------------------------
// Graph
//---------------------------

void ProgressDisplayGraph::init() {
	std::string leadin = "  |-------- progress ";
	*outstream << leadin;
	for (size_t i = leadin.length() - 3; i < numStars; i++) *outstream << "-";
	*outstream << "|" << std::endl << "  |";
}

//printing stars in one line
void ProgressDisplayGraph::update(bool force) {
	double done = progressPercent();
	if (done > 0 && done <= 100.0) {
		int stars = int(0.01 * done * numStars);
		for (; numPrinted < stars; numPrinted++) *outstream << "*";
	}
}

void ProgressDisplayGraph::terminate() {
	for (; numPrinted < numStars; numPrinted++) *outstream << "*";
	*outstream << "|" << std::endl;
}

//---------------------------
// Rewrite same line
//---------------------------

//rewrite one line on the console
void ProgressDisplayRewriteLine::update(bool force) {
	if (isDue(force)) {
		//overwrite existing line
		line.assign(line.size(), ' ');
		line[0] = '\r';
		*outstream << line;

		//new line
		double done = progressPercent();
		if (isfinite(done)) line = std::format("\rframes in {}, out {}, done {:.1f}%, data written {}",
			data.status.frameInputIndex, data.status.frameWriteIndex, done, util::byteSizeToString(data.status.outputBytesWritten));
		else line = std::format("\rframes in {}, out {}, data written {}",
			data.status.frameInputIndex, data.status.frameWriteIndex, util::byteSizeToString(data.status.outputBytesWritten));
		*outstream << line;
	}
}

void ProgressDisplayRewriteLine::terminate() {
	*outstream << std::endl;
}

//---------------------------
// Detailed timings
//---------------------------

//show detailed report
void ProgressDisplayDetailed::update(bool force) {
	std::vector<std::string> strings;

	//input stats
	if (data.status.frameReadIndex != lastReadFrame && data.status.endOfInput == false) {
		VideoPacketContext stats = data.status.packetList.back();
		strings.push_back(std::format("read idx={}, dts={}, pts={}, duration={}",
			data.status.frameReadIndex, stats.dts, stats.pts, stats.duration));
		lastReadFrame = data.status.frameReadIndex;
	}

	//encoding stats
	if (data.status.frameEncodeIndex != lastEncodedFrame) {
		strings.push_back(std::format("encode idx={}, dts={}, pts={}", data.status.frameEncodeIndex - 1, data.status.encodedDts, data.status.encodedPts));
		lastEncodedFrame = data.status.frameEncodeIndex;
	}

	//send to display
	*outstream << util::concatStrings(strings, " // ", "", "\n");
}