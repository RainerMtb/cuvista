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

#include "ThreadPoolBase.h"
#include <queue>

//actual async thread pool
class ThreadPool : public ThreadPoolBase {

private:
	std::vector<int> mBusyArray; //do not use vector<bool> here
	mutable std::queue<std::packaged_task<void()>> mJobs;
	mutable std::mutex mMutex;
	mutable std::condition_variable mCV, mBusy;
	bool mCancelRequest = false;

	bool isIdle() const;
	bool isBusy() const;

public:
	static inline ThreadPoolBase defaultPool;

	ThreadPool(size_t numThreads = 1);
	~ThreadPool() override;

	//wait for all pending jobs to execute
	void wait() const override;

	//add job to queue
	std::future<void> add(std::function<void()> job) const override;

	//create jobs and add to queue by iterating from a to b and calling func
	void addAndWait(std::function<void(size_t)> job, size_t iterStart, size_t iterEnd) const override;

	//cancel all jobs
	void cancel() override;

	//shutdown all threads
	void shutdown() override;
};