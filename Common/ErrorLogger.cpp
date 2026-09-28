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
#include <iostream>

std::ostream& printError(std::ostream& os, const std::string& msg) {
	//print in red ansi formatting
	return os << "\x1B[1;31m" << msg << "\x1B[0m" << std::endl;
}

ErrorLogger& errorLogger() {
	return *errorLoggerInstance;
}

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

std::vector<ErrorEntry> ErrorLogger::getErrors() {
	std::lock_guard<std::mutex> lock(mMutex);
	return std::vector<ErrorEntry>(errorList.cbegin(), errorList.cend());
}

std::vector<FFmpegLog> ErrorLogger::getLogs() {
	std::lock_guard<std::mutex> lock(mMutex);
	return std::vector<FFmpegLog>(ffmpegLog.cbegin(), ffmpegLog.cend());
}

std::string ErrorLogger::getErrorMessage() {
	std::lock_guard<std::mutex> lock(mMutex);
	return errorList.empty() ? "no error" : errorList.front().msg;
}

void ErrorLogger::logFFmpeg(int logLevel, std::string msg) {
	ffmpegLog.emplace_back(std::chrono::system_clock::now(), FFmpegLog::indexTotal, logLevel, msg);
	FFmpegLog::indexTotal++;
	while (ffmpegLog.size() > 5000) ffmpegLog.pop_front();
}

void ErrorLogger::printErrors(std::ostream& os) {
	//list of recorded errors
	std::vector<ErrorEntry> errorList = errorLogger().getErrors();
	if (errorList.size() > 0) {
		printError(os, "ERROR STACK:");
		for (int i = 0; i < errorList.size(); i++) {
			printError(os, std::format("[{}] {}", i, errorList[i].msg));
		}
	}

	//list of recorded ffmpeg logs
	std::vector<FFmpegLog> ffmpegErrors;
	std::vector<FFmpegLog> ffmpegLogs;
	for (auto iter = ffmpegLog.crbegin(); iter != ffmpegLog.crend(); iter++) {
		if (iter->logLevel <= 16) {
			ffmpegErrors.push_back(*iter);

		} else {
			ffmpegLogs.push_back(*iter);
		}
	}
	if (ffmpegErrors.size() > 0) {
		printError(os, "FFMPEG ERRORS:");
		for (int i = 0; i < ffmpegErrors.size(); i++) {
			printError(os, std::format("[{}] {}", i, ffmpegErrors[i].msg));
		}
	}

	if ((errorList.size() > 0 || ffmpegErrors.size() > 0) && ffmpegLogs.size() > 0) {
		printError(os, "LOGS:");
		for (int i = 0; i < ffmpegLogs.size(); i++) {
			printError(os, std::format("[{}] {}", i, ffmpegLogs[i].msg));
		}
	}
}

void ErrorLogger::clear() {
	std::lock_guard<std::mutex> lock(mMutex);
	errorList.clear();
	ffmpegLog.clear();
}

void ErrorLogger::clearErrors(ErrorSource source) {
	std::lock_guard<std::mutex> lock(mMutex);
	std::erase_if(errorList, [&] (const ErrorEntry& entry) { return entry.source == source; });
}