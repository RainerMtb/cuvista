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

#include "ProgressDisplay.hpp"

//base class for display on console
class ProgressDisplayConsole : public ProgressDisplay {

protected:
	std::stringstream outBuffer;
	std::ostream* outstream;

	ProgressDisplayConsole(MovieFrame& frame, std::ostream* outstream);
	void writeMessage(const std::string& str) override;
	std::stringstream& buildMessage();
};

//display a character graph between 0% and 100%
class ProgressDisplayGraph : public ProgressDisplayConsole {

private:
	int numStars = 75;
	int numPrinted = 0;

public:
	ProgressDisplayGraph(MovieFrame& frame, std::ostream* outstream) :
		ProgressDisplayConsole(frame, outstream) {}
	void init() override;
	void update(bool force) override;
	void terminate() override;
	void writeMessage(const std::string& msg) override {}
};

//new line for every frame
class ProgressDisplayNewLine : public ProgressDisplayConsole {

public:
	ProgressDisplayNewLine(MovieFrame& frame, std::ostream* outstream) :
		ProgressDisplayConsole(frame, outstream) {}
	void update(bool force = false) override;
};

//rewrite one line with updated status
class ProgressDisplayRewriteLine : public ProgressDisplayConsole {

private:
	std::string output;

public:
	ProgressDisplayRewriteLine(MovieFrame& frame, std::ostream* outstream) :
		ProgressDisplayConsole(frame, outstream) {}
	void update(bool force) override;
	void terminate() override;
};

//detailed progress report
class ProgressDisplayDetailed : public ProgressDisplayConsole {

private:
	int64_t lastReadFrame = -1;
	int64_t lastEncodedFrame = 0;

public:
	ProgressDisplayDetailed(MovieFrame& frame, std::ostream* outstream) :
		ProgressDisplayConsole(frame, outstream) {}
	void update(bool force) override;
};

//silent progress
class ProgressDisplayNone : public ProgressDisplay {

public:
	ProgressDisplayNone(MovieFrame& frame) :
		ProgressDisplay(frame) {}
	void update(bool force) override {}
};