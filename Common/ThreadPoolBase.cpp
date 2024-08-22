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

#include <functional>
#include <mutex>
#include <vector>
#include <queue>
#include <cassert>
#include "ThreadPoolBase.h"

 //execute one job
std::future<void> ThreadPoolBase::add(std::function<void()> job) const {
	job();
	return { std::async([] {}) };
}

//iterate over job array
void ThreadPoolBase::addAndWait(std::function<void(size_t)> job, size_t iterStart, size_t iterEnd) const {
	for (size_t i = iterStart; i < iterEnd; i++) {
		std::bind(job, i)();
	}
}

//number of threads
size_t ThreadPoolBase::size() const {
	return mThreads.size();
}