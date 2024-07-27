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

#include <chrono>
#include <vector>
#include <mutex>

struct ErrorEntry {
	std::chrono::time_point<std::chrono::system_clock> t;
	std::string msg;
};

class ErrorLogger {
	std::mutex mMutex;
	std::vector<ErrorEntry> errorList;

public:
	bool hasNoError();

	bool hasError();

	void logError(const std::string& msg);

	void logError(const char* title, const char* msg);

	void logError(const std::string& title, const std::string& msg);

	std::vector<ErrorEntry> getErrors();

	std::string getErrorMessage();

	void clearErrors();
};

inline ErrorLogger errorLogger = {};