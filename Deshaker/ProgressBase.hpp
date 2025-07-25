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

class ProgressBase {

public:
	virtual void init() {}
	virtual void update(double totalPercentage, bool force = false) = 0;
	virtual void terminate() {}
	virtual void writeMessage(const std::string& msg) {}
	virtual ~ProgressBase() {}
};


class ProgressDefault : public ProgressBase {

public:
	void update(double totalPercentage, bool force = false) override {}
};