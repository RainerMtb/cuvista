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

#include <iostream>
#include <chrono>
#include "MainData.hpp"
#include "Stats.hpp"

//base class
class ProgressDisplay {

protected:
	MainData& data;
	std::chrono::steady_clock::time_point timePoint;
	std::chrono::milliseconds interval;

	ProgressDisplay(MainData& data, int interval) : 
		data { data }, 
		interval { interval } 
	{}

	bool isDue(bool forceUpdate);
	double progressPercent();
	bool isFinite();

public:
	virtual void init() {}
	virtual void update(bool force = false) = 0;
	virtual void terminate() {}
	virtual void writeMessage(const std::string& msg) {}
};
