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
#include <algorithm>

#include "ProgressDisplayConsole.hpp"
#include "Util.hpp"
#include "SystemStuff.hpp"

ProgressDisplayConsole::ProgressDisplayConsole(std::ostream* outstream, int interval) :
	ProgressDisplay(interval), 
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

std::stringstream& ProgressDisplayConsole::buildMessageLine(const ProgressInfo& progress, std::stringstream& buffer) {
	if (isfinite(progress.totalProgress))
		buffer << "done " << progress.totalProgress << "% ";
	if (progress.readIndex > 0)
		buffer << "frames in " << progress.readIndex;
	if (progress.writeIndex > 0)
		buffer << ", frames out " << progress.writeIndex;

	if (progress.outputBytesWritten > 0)
		buffer << ", output written " << util::byteSizeToString(progress.outputBytesWritten);
	return buffer;
}

//---------------------------
// New Line per frame
//---------------------------

//print a new line to the console
void ProgressDisplayNewLine::update(const ProgressInfo& progress, bool force) {
	if (isDue(force)) {
		outBuffer.str("");
		buildMessageLine(progress, outBuffer);
		*outstream << outBuffer.str() << std::endl;
	}
}

//---------------------------
// Rewrite same line
//---------------------------

//rewrite one line on the console
void ProgressDisplayRewriteLine::update(const ProgressInfo& progress, bool force) {
	if (isDue(force)) {
		outBuffer.str("");
		outBuffer << "\x0D\x1B[2K";
		buildMessageLine(progress, outBuffer);
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
void ProgressDisplayGraph::update(const ProgressInfo& progress, bool force) {
	double done = progress.totalProgress;
	if (done > 0 && done <= 100.0) {
		int stars = int(0.01 * done * numStars);
		for (; numPrinted < stars; numPrinted++) *outstream << "*";
	}
}

void ProgressDisplayGraph::terminate() {
	*outstream << "|" << std::endl;
}

//------------------------------
// Default mode multiple lines
//------------------------------

void ProgressDisplayMultiLine::init() {
	*outstream << buildMessage({}) << std::flush;
}

void ProgressDisplayMultiLine::update(const ProgressInfo& progress, bool force) {
	if (isDue(force)) {
		*outstream << "\x1B[5A" << buildMessage(progress) << std::flush;
	}
}

void ProgressDisplayMultiLine::writeMessage(const std::string& str) {
	//set message
	mStatusMessage = str;
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

std::string ProgressDisplayMultiLine::buildMessage(const ProgressInfo& progress) {
	int64_t graphLength = getConsoleWidth() - 25LL;
	return std::format("\x0D\x1B[2K{}\n\x0D\x1B[2Kinput   {}\n\x0D\x1B[2Koutput  {}\n\x0D\x1B[2Kencoded {}\n\x0D\x1B[2Koutput written {}\n",
		mStatusMessage,
		buildLine(progress.readIndex, progress.frameCount, graphLength),
		buildLine(progress.writeIndex, progress.frameCount, graphLength),
		buildLine(progress.encodeIndex, progress.frameCount, graphLength),
		util::byteSizeToString(progress.outputBytesWritten))
		;
}