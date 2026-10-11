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

std::ostream& printWarning(std::ostream& os, const std::string& msg) {
	//print in yellow ansi formatting
	return os << "\x1B[1;33m" << msg << "\x1B[0m" << std::endl;
}

ErrorLogger& errorLogger() {
	return *errorLoggerInstance;
}

bool ErrorLogger::hasNoError() {
	std::lock_guard<std::mutex> lock(mMutex);
	return mErrorList.empty();
}

bool ErrorLogger::hasError() {
	std::lock_guard<std::mutex> lock(mMutex);
	return mErrorList.size() > 0;
}

void ErrorLogger::logError(const std::string& msg, ErrorSource source) {
	std::lock_guard<std::mutex> lock(mMutex);
	mErrorList.emplace_back(std::chrono::system_clock::now(), msg, source);
}

void ErrorLogger::logError(const char* title, const char* msg, ErrorSource source) {
	logError(std::string(title) + std::string(msg));
}

std::vector<ErrorEntry> ErrorLogger::getErrors() {
	std::lock_guard<std::mutex> lock(mMutex);
	return std::vector<ErrorEntry>(mErrorList.cbegin(), mErrorList.cend());
}

std::vector<FFmpegLog> ErrorLogger::getLogs() {
	std::lock_guard<std::mutex> lock(mMutex);
	return std::vector<FFmpegLog>(mFFmpegLog.cbegin(), mFFmpegLog.cend());
}

std::string ErrorLogger::getErrorMessage() {
	std::lock_guard<std::mutex> lock(mMutex);
	return mErrorList.empty() ? "no error" : mErrorList.front().msg;
}

void ErrorLogger::logFFmpeg(int logLevel, const std::string& msg) {
	std::lock_guard<std::mutex> lock(mMutex);
	mFFmpegLog.emplace_back(std::chrono::system_clock::now(), FFmpegLog::indexTotal, logLevel, msg);
	FFmpegLog::indexTotal++;
	while (mFFmpegLog.size() > 5000) mFFmpegLog.pop_front();
}

void ErrorLogger::logWarning(const std::string& msg) {
	std::lock_guard<std::mutex> lock(mMutex);
	mWarnings.push_back(msg);
}

void ErrorLogger::printErrors(std::ostream& os, bool showVerbose) {
	//list of recorded errors
	std::vector<ErrorEntry> errorList = errorLogger().getErrors();
	if (errorList.size() > 0) {
		printError(os, "ERROR STACK:");
		for (int i = 0; i < errorList.size(); i++) {
			printError(os, std::format("[{}] {}", i, errorList[i].msg));
		}
	}

	//list of other warnings
	if (showVerbose && mWarnings.size() > 0) {
		printWarning(os, "\n");
		printWarning(os, "WARNINGS:");
		int i = 0;
		for (const std::string& msg : mWarnings) {
			printWarning(os, std::format("[{}] {}", i++, msg));
		}
	}

	//list of recorded ffmpeg logs
	std::vector<FFmpegLog> ffmpegErrors;
	std::vector<FFmpegLog> ffmpegLogs;
	for (auto iter = mFFmpegLog.crbegin(); iter != mFFmpegLog.crend(); iter++) {
		if (iter->logLevel <= 16) {
			ffmpegErrors.push_back(*iter);

		} else {
			ffmpegLogs.push_back(*iter);
		}
	}
	if (showVerbose && ffmpegErrors.size() > 0) {
		printError(os, "\n");
		printError(os, "FFMPEG ERRORS:");
		for (int i = 0; i < ffmpegErrors.size(); i++) {
			printError(os, std::format("[{}] {}", i, ffmpegErrors[i].msg));
		}
	}

	if (showVerbose && ffmpegLogs.size() > 0) {
		printWarning(os, "\n");
		printWarning(os, "FFMPEG LOGS:");
		for (int i = 0; i < ffmpegLogs.size(); i++) {
			printWarning(os, std::format("[{}] {}", i, ffmpegLogs[i].msg));
		}
	}
}

void ErrorLogger::clear() {
	std::lock_guard<std::mutex> lock(mMutex);
	mErrorList.clear();
	mFFmpegLog.clear();
	mWarnings.clear();
}
