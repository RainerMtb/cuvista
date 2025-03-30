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

	ProgressDisplayConsole(MovieFrame& frame, std::ostream* outstream, int interval = 500);
	void writeMessage(const std::string& str) override;
	std::stringstream& buildMessageLine(double totalPercentage, std::stringstream& buffer);
};

//option 4
//display a character graph between 0% and 100%
class ProgressDisplayGraph : public ProgressDisplayConsole {

private:
	int numStars = 75;
	int numPrinted = 0;

public:
	ProgressDisplayGraph(MovieFrame& frame, std::ostream* outstream) :
		ProgressDisplayConsole(frame, outstream) {}
	
	void init() override;
	void update(double totalPercentage, bool force) override;
	void terminate() override;
	void writeMessage(const std::string& msg) override {}
};

//option 3
//new line for every frame
class ProgressDisplayNewLine : public ProgressDisplayConsole {

public:
	ProgressDisplayNewLine(MovieFrame& frame, std::ostream* outstream) :
		ProgressDisplayConsole(frame, outstream) {}
	
	void update(double totalPercentage, bool force = false) override;
};

//option 2
//rewrite one line with updated status
class ProgressDisplayRewriteLine : public ProgressDisplayConsole {

public:
	ProgressDisplayRewriteLine(MovieFrame& frame, std::ostream* outstream) :
		ProgressDisplayConsole(frame, outstream) {}
	
	void update(double totalPercentage, bool force) override;
	void terminate() override;
	void writeMessage(const std::string& msg) override;
};

//option 1
//rewrite multiple lines
class ProgressDisplayMultiLine : public ProgressDisplayConsole {

private:
	std::string statusMessage = "";

	int getConsoleWidth();
	std::string buildMessage();
	std::string buildLine(int64_t frameIndex, int64_t frameCount, int64_t graphLength);

public:
	ProgressDisplayMultiLine(MovieFrame& frame, std::ostream* outstream) :
		ProgressDisplayConsole(frame, outstream, 200) {}
	
	void init() override;
	void update(double totalPercentage, bool force) override;
	void writeMessage(const std::string& msg) override;
};

//option 0
//silent progress
class ProgressDisplayNone : public ProgressDisplay {

public:
	ProgressDisplayNone(MovieFrame& frame) :
		ProgressDisplay(frame) {}
	
	void update(double totalPercentage, bool force) override {}
	void writeMessage(const std::string& msg) override {}
};
