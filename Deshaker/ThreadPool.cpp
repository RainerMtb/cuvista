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

#include "ThreadPool.hpp"
#include <algorithm>

ThreadPool::ThreadPool(size_t numThreads) :
	mBusyArray(numThreads) 
{
	//define function to be executed by each thread
	std::function<void(size_t)> loopFunction = [&] (size_t idx) {
		while (true) {
			//get next pending job
			std::packaged_task<void()> pt;
			{
				std::unique_lock<std::mutex> lock(mMutex);
				mBusyArray[idx] = false;
				if (isIdle()) mBusy.notify_all();

				mCV.wait(lock, [&] () { return !mJobs.empty() || mCancelRequest; });
				if (mCancelRequest) break;
				pt = std::move(mJobs.front());
				mJobs.pop();

				mBusyArray[idx] = true;
			}

			//execute the job
			pt();
		}
	};

	//start threads
	for (size_t thr = 0; thr < numThreads; thr++) {
		mThreads.emplace_back(loopFunction, thr);
	}
}

bool ThreadPool::isIdle() const {
	return mJobs.empty() && std::none_of(mBusyArray.cbegin(), mBusyArray.cend(), [] (int a) { return a; });
}

bool ThreadPool::isBusy() const {
	return !mJobs.empty() || std::any_of(mBusyArray.cbegin(), mBusyArray.cend(), [] (int a) { return a; });
}

void ThreadPool::wait() const {
	std::unique_lock<std::mutex> lock(mMutex);
	if (isBusy()) mBusy.wait(lock);
}

ThreadPool::~ThreadPool() {
	shutdown();
}

std::future<void> ThreadPool::add(std::function<void()> job) const {
	std::unique_lock<std::mutex> lock(mMutex);
	mJobs.emplace(std::packaged_task<void()>(job));
	auto fut = mJobs.back().get_future();
	mCV.notify_one();
	return fut;
}

void ThreadPool::addAndWait(std::function<void(size_t)> job, size_t iterStart, size_t iterEnd) const {
	std::vector<std::future<void>> futures(iterEnd - iterStart);
	{
		//create jobs and queue up
		std::unique_lock<std::mutex> lock(mMutex);
		for (size_t i = iterStart; i < iterEnd; i++) {
			mJobs.emplace(std::packaged_task<void()>(std::bind(job, i)));
			futures[i] = mJobs.back().get_future();
		}
		mCV.notify_all();
	}
	//wait for jobs to complete
	//destructor of future only blocks when created through std::async ?!
	for (auto& f : futures) {
		f.wait();
	}
}

void ThreadPool::cancel() {
	std::unique_lock<std::mutex> lock(mMutex);
	mCancelRequest = true;
	mCV.notify_all();
}

void ThreadPool::shutdown() {
	cancel();
	for (std::thread& thr : mThreads) {
		thr.join();
	}
	mThreads.clear();
}

size_t ThreadPool::size() const {
	return mThreads.size();
}