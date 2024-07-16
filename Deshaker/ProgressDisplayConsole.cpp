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
#include "MovieFrame.hpp"

ProgressDisplayConsole::ProgressDisplayConsole(MovieFrame& frame, std::ostream* outstream) :
	ProgressDisplay(frame, 500), 
	outstream { outstream } 
{
	outBuffer.precision(1);
	outBuffer << std::showpoint;
	outBuffer << std::fixed;
	timePoint = std::chrono::steady_clock::now();
}

void ProgressDisplayConsole::writeMessage(const std::string& str) {
	*outstream << str << std::flush;
}

std::stringstream& ProgressDisplayConsole::buildMessage() {
	outBuffer.str("");
	double donePercent = progressPercent();
	if (isfinite(donePercent))
		outBuffer << "done " << donePercent << "% ";
	if (frame.mReader.frameIndex > 0)
		outBuffer << "frames in " << frame.mReader.frameIndex;
	if (frame.mWriter.frameIndex > 0)
		outBuffer << ", frames out " << frame.mWriter.frameIndex;
	if (frame.mWriter.outputBytesWritten > 0)
		outBuffer << ", written " << util::byteSizeToString(frame.mWriter.outputBytesWritten);
	return outBuffer;
}

//---------------------------
// New Line per frame
//---------------------------

//print a new line to the console
void ProgressDisplayNewLine::update(bool force) {
	buildMessage();
	*outstream << outBuffer.str() << std::endl;
}

//---------------------------
// Rewrite same line
//---------------------------

//rewrite one line on the console
void ProgressDisplayRewriteLine::update(bool force) {
	if (isDue(force)) {
		//overwrite existing line
		output.assign(output.length() + 1, ' ');
		output[0] = '\r';
		*outstream << output;

		//new line
		output = "\r" + buildMessage().str();
		*outstream << output << std::flush;
	}
}

void ProgressDisplayRewriteLine::terminate() {
	*outstream << std::endl;
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
// Detailed timings
//---------------------------

//show detailed report
void ProgressDisplayDetailed::update(bool force) {
	std::vector<std::string> strings;

	//input stats
	if (frame.mReader.frameIndex != lastReadFrame && frame.mReader.endOfInput == false) {
		VideoPacketContext stats = frame.mReader.packetList.back();
		strings.push_back(std::format("read idx={}, dts={}, pts={}, duration={}", frame.mReader.frameIndex, stats.dts, stats.pts, stats.duration));
		lastReadFrame = frame.mReader.frameIndex;
	}

	//encoding stats
	if (frame.mWriter.frameEncoded != lastEncodedFrame) {
		strings.push_back(std::format("encode idx={}, dts={}, pts={}", frame.mWriter.frameEncoded, frame.mWriter.encodedDts, frame.mWriter.encodedPts));
		lastEncodedFrame = frame.mWriter.frameEncoded;
	}

	//send to display
	*outstream << util::concatStrings(strings, " // ", "", "\n");
}