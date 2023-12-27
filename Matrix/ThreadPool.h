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
#include <future>
#include <mutex>
#include <vector>
#include <queue>
#include <cassert>

//base class directly executes job in the same thread
class ThreadPoolBase {

protected:

	std::vector<std::thread> mThreads;

public:

	virtual ~ThreadPoolBase() {}

	virtual void wait() {}
	virtual void cancel() {}
	virtual void shutdown() {}

	//execute one job
	virtual std::future<void> add(std::function<void()> job) {
		job();
		return {};
	}

	//iterate over job array
	virtual void add(std::function<void(size_t)> job, size_t iterStart, size_t iterEnd) {
		for (size_t i = iterStart; i < iterEnd; i++) {
			std::bind(job, i)(); //exectute job directly
		}
	}

	//number of threads
	size_t size() const {
		return mThreads.size();
	}
};


//actual thread pool
class ThreadPool : public ThreadPoolBase {

private:
	std::vector<int> mBusyArray; //do not use vector<bool> here
	std::queue<std::packaged_task<void()>> mJobs;
	std::vector<std::future<void>> futures;

	std::mutex mMutex;
	std::condition_variable mCV, mBusy;
	bool mCancelRequest = false;

	bool isIdle() const {
		return mJobs.empty() && std::none_of(mBusyArray.cbegin(), mBusyArray.cend(), [] (int a) { return a; });
	}

	bool isBusy() const {
		return !mJobs.empty() || std::any_of(mBusyArray.cbegin(), mBusyArray.cend(), [] (int a) { return a; }); 
	}

public:
	ThreadPool(size_t numThreads = 1) : 
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

	void wait() override {
		std::unique_lock<std::mutex> lock(mMutex);
		if (isBusy()) mBusy.wait(lock);
	}

	~ThreadPool() override {
		shutdown();
	}

	//add job to queue
	std::future<void> add(std::function<void()> job) override {
		std::unique_lock<std::mutex> lock(mMutex);
		mJobs.emplace(std::packaged_task<void()>(job));
		auto fut = mJobs.back().get_future();
		mCV.notify_one();
		return fut;
	}

	//create jobs and add to queue by iterating from a to b and calling func
	void add(std::function<void(size_t)> job, size_t iterStart, size_t iterEnd) override {
		assert(futures.empty() && "list of futures must be empty here");
		//create jobs and queue up
		for (size_t i = iterStart; i < iterEnd; i++) {
			futures.emplace_back(add(std::bind(job, i)));
		}
		//wait for jobs to complete
		for (auto& fut : futures) {
			fut.get();
		}
		futures.clear();
	}

	//cancel all jobs
	void cancel() override {
		std::unique_lock<std::mutex> lock(mMutex);
		mCancelRequest = true;
		mCV.notify_all();
	}

	//shutdown all threads
	void shutdown() override {
		cancel();
		for (std::thread& thr : mThreads) {
			thr.join();
		}
		mThreads.clear();
	}

};
