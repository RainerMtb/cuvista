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

#include <thread>
#include <functional>
#include <vector>
#include <future>

//base class syncronously executes jobs
class ThreadPoolBase {

public:
	virtual ~ThreadPoolBase() = default;
	virtual void wait() const {}
	virtual void cancel() {}
	virtual void shutdown() {}

	//execute one job
	virtual std::future<void> add(std::function<void()> job) const;

	//iterate over job array
	virtual void addAndWait(std::function<void(size_t)> job, size_t iterStart, size_t iterEnd) const;

	//number of threads
	virtual size_t size() const;
};

inline ThreadPoolBase defaultPool;