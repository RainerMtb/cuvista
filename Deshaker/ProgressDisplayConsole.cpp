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
#include "MovieReader.hpp"

ProgressDisplayConsole::ProgressDisplayConsole(MovieFrame& frame, std::ostream* outstream, int interval) :
	ProgressDisplay(frame, interval), 
	outstream { outstream } 
{
	outBuffer.precision(1);
	outBuffer << std::showpoint;
	outBuffer << std::fixed;
	timePoint = std::chrono::steady_clock::now();
}

void ProgressDisplayConsole::writeMessage(const std::string& str) {
	*outstream << str << std::endl;
}

std::stringstream& ProgressDisplayConsole::buildMessageLine(double totalPercentage, std::stringstream& buffer) {
	if (isfinite(totalPercentage))
		buffer << "done " << totalPercentage << "% ";
	if (frame.mReader.frameIndex > 0)
		buffer << "frames in " << frame.mReader.frameIndex;
	if (frame.mWriter.frameIndex > 0)
		buffer << ", frames out " << frame.mWriter.frameIndex;

	std::unique_lock<std::mutex> lock(frame.mWriter.mStatsMutex);
	if (frame.mWriter.outputBytesWritten > 0)
		buffer << ", output written " << util::byteSizeToString(frame.mWriter.outputBytesWritten);
	return buffer;
}

//---------------------------
// New Line per frame
//---------------------------

//print a new line to the console
void ProgressDisplayNewLine::update(double totalPercentage, bool force) {
	if (isDue(force)) {
		outBuffer.str("");
		buildMessageLine(totalPercentage, outBuffer);
		*outstream << outBuffer.str() << std::endl;
	}
}

//---------------------------
// Rewrite same line
//---------------------------

//rewrite one line on the console
void ProgressDisplayRewriteLine::update(double totalPercentage, bool force) {
	if (isDue(force)) {
		outBuffer.str("");
		outBuffer << "\x0D\x1B[2K";
		buildMessageLine(totalPercentage, outBuffer);
		*outstream << outBuffer.str() << std::flush;
	}
}

void ProgressDisplayRewriteLine::terminate() {
	*outstream << std::endl;
}

void ProgressDisplayRewriteLine::writeMessage(const std::string& msg) {
	*outstream << "\x0D\x1B[2K" << msg << std::endl;
}

//---------------------------
// Graph
//---------------------------

void ProgressDisplayGraph::init() {
	std::string leadin = "|-------- progress indicator ";
	*outstream << leadin;
	for (size_t i = leadin.length() - 1; i < numStars; i++) *outstream << "-";
	*outstream << "|" << std::endl << "|";
}

//printing stars in one line
void ProgressDisplayGraph::update(double totalPercentage, bool force) {
	double done = totalPercentage;
	if (done > 0 && done <= 100.0) {
		int stars = int(0.01 * done * numStars);
		for (; numPrinted < stars; numPrinted++) *outstream << "*";
	}
}

void ProgressDisplayGraph::terminate() {
	for (; numPrinted < numStars; numPrinted++) *outstream << "*";
	*outstream << "|" << std::endl;
}

//------------------------------
// Default mode multiple lines
//------------------------------

void ProgressDisplayMultiLine::init() {
	*outstream << buildMessage() << std::flush;
}

void ProgressDisplayMultiLine::update(double totalPercentage, bool force) {
	if (isDue(force)) {
		*outstream << "\x1B[5A" << buildMessage() << std::flush;
	}
}

void ProgressDisplayMultiLine::writeMessage(const std::string& str) {
	//set message
	statusMessage = str;
	//move curser up and reprint lines
	*outstream << "\x1B[5A" << buildMessage() << std::flush;
}


int ProgressDisplayMultiLine::getConsoleWidth() {
	return std::clamp(getSystemConsoleWidth(), 40, 200);
}

std::string ProgressDisplayMultiLine::buildLine(int64_t frameIndex, int64_t frameCount, int64_t graphLength) {
	std::string line(graphLength + 15, ' ');
	if (frameCount > 0) {
		int64_t hashNum = std::clamp(graphLength * frameIndex / frameCount, int64_t(0), graphLength);
		double percent = std::clamp(100.0 * frameIndex / frameCount, 0.0, 100.0);
		std::format_to(line.begin(), "{:6d} [{}{}] {:4.0f}%", frameIndex, std::string(hashNum, '#'), std::string(graphLength - hashNum, '.'), percent);

	} else {
		int64_t loopDuration = 500;
		std::format_to(line.begin(), "{:6d} [{}]", frameIndex, std::string(graphLength, '.'));
		int64_t pos = (frameIndex % loopDuration) * graphLength / loopDuration;
		line[pos + 8] = '#';
	}
	return line;
}

std::string ProgressDisplayMultiLine::buildMessage() {
	std::unique_lock<std::mutex> lock(frame.mWriter.mStatsMutex);
	int64_t graphLength = getConsoleWidth() - 25LL;
	return std::format("\x0D\x1B[2K{}\n\x0D\x1B[2Kinput   {}\n\x0D\x1B[2Koutput  {}\n\x0D\x1B[2Kencoded {}\n\x0D\x1B[2Koutput written {}\n",
		statusMessage,
		buildLine(frame.mReader.frameIndex, frame.mReader.frameCount, graphLength),
		buildLine(frame.mWriter.frameIndex, frame.mReader.frameCount, graphLength),
		buildLine(frame.mWriter.frameEncoded, frame.mReader.frameCount, graphLength),
		util::byteSizeToString(frame.mWriter.outputBytesWritten))
		;
}