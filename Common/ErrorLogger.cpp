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

#include "ErrorLogger.hpp"

bool ErrorLogger::hasNoError() {
	std::lock_guard<std::mutex> lock(mMutex);
	return errorList.empty();
}

bool ErrorLogger::hasError() {
	std::lock_guard<std::mutex> lock(mMutex);
	return errorList.size() > 0;
}

void ErrorLogger::logError(const std::string& msg, ErrorSource source) {
	std::lock_guard<std::mutex> lock(mMutex);
	errorList.emplace_back(std::chrono::system_clock::now(), msg, source);
}

void ErrorLogger::logError(const char* title, const char* msg, ErrorSource source) {
	logError(std::string(title) + std::string(msg));
}

void ErrorLogger::logError(const std::string& title, const std::string& msg, ErrorSource source) {
	logError(title + msg);
}

std::vector<ErrorEntry> ErrorLogger::getErrors() {
	std::lock_guard<std::mutex> lock(mMutex);
	return errorList;
}

std::string ErrorLogger::getErrorMessage() {
	std::lock_guard<std::mutex> lock(mMutex);
	return errorList.empty() ? "no error" : errorList[0].msg;
}

void ErrorLogger::clearErrors() {
	std::lock_guard<std::mutex> lock(mMutex);
	errorList.clear();
}

void ErrorLogger::clearErrors(ErrorSource source) {
	std::lock_guard<std::mutex> lock(mMutex);
	std::erase_if(errorList, [&] (const ErrorEntry& entry) { return entry.source == source; });
}