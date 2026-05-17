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
#include "Util.hpp"
#include <numeric>

ThreadPool::ThreadPool(size_t numThreads) :
	mActive(numThreads)
{
	//define function to be executed by each thread
	for (size_t i = 0; i < numThreads; i++) {
		auto loopFunc = [&, i] {
			std::packaged_task<void()> pt;
			while (true) {
				std::unique_lock<std::mutex> lock(mMutexWork);
				mCVwork.wait(lock, [&] { return mHasSharedWork || mJobs.size() || mCancelRequest; });
				pt = std::packaged_task<void()>([] {});
				if (mCancelRequest) {
					return;

				} else if (mJobs.size()) {
					pt = std::move(mJobs.front());
					mJobs.pop();

				} else if (mHasSharedWork) {
					pt = std::packaged_task<void()>(mSharedJob);
				}
				mActive[i] = 1;
				lock.unlock();

				pt();

				std::unique_lock<std::mutex> lockDone(mMutexWork);
				mActive[i] = 0;
				if (activeWorkers() == 0) {
					mCVdone.notify_all();
				}
			}
		};
		mThreads.emplace_back(loopFunc);
	};
}

size_t ThreadPool::size() const {
	return mThreads.size();
}

int ThreadPool::activeWorkers() const {
	return std::accumulate(mActive.cbegin(), mActive.cend(), 0);
}

std::future<void> ThreadPool::add(std::function<void()> job) {
	std::unique_lock<std::mutex> lock(mMutexWork);
	auto& task = mJobs.emplace(std::packaged_task<void()>(job));
	auto fut = task.get_future();
	mCVwork.notify_one();
	return fut;
}

void ThreadPool::addAndWait(std::function<void(size_t)> job, size_t iterStart, size_t iterEnd) {
	auto func = [=] (FuncIndex workIndex) {
		for (size_t i = workIndex(); i < iterEnd; i = workIndex()) {
			job(i);
		}
	};
	workAndWait(func, iterStart, iterEnd);
}

void ThreadPool::workAndWait(FuncPool sharedJob, size_t iterStart, size_t iterEnd) {
	std::unique_lock<std::mutex> lock(mMutexWork);
	size_t idx = iterStart;
	std::mutex m;
	auto sharedIndex = [&] {
		std::unique_lock<std::mutex> lck(m);
		size_t temp = idx;
		idx++;
		mHasSharedWork = idx < iterEnd;
		return temp;
	};
	mSharedJob = std::bind(sharedJob, sharedIndex);
	mHasSharedWork = true;
	lock.unlock();
	mCVwork.notify_all();

	std::unique_lock<std::mutex> lockDone(mMutexWork);
	mCVdone.wait(lockDone, [&] { return activeWorkers() == 0 && mHasSharedWork == false; });
}

void ThreadPool::shutdown() {
	std::unique_lock<std::mutex> lock(mMutexWork);
	mCancelRequest = true;
	lock.unlock();
	mCVwork.notify_all();
	for (std::thread& thr : mThreads) {
		thr.join();
	}
	mThreads.clear();
}

ThreadPool::~ThreadPool() {
	shutdown();
}
