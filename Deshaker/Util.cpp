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

#include "Util.hpp"

void util::ConsoleTimer::interval(const std::string& name) {
    auto interval = std::chrono::high_resolution_clock::now();
    auto delta = std::chrono::duration_cast<std::chrono::microseconds>(interval - mInterval);
    mInterval = interval;
    std::cout << mName << ":" << name << "=" << delta.count() / 1000.0 << " ms" << std::endl;
}

util::ConsoleTimer::~ConsoleTimer() {
    auto stop = std::chrono::high_resolution_clock::now();
    auto delta = std::chrono::duration_cast<std::chrono::microseconds>(stop - mStart);
    std::cout << mName << "=" << delta.count() / 1000.0 << " ms" << std::endl;
}

std::string util::concatStrings(std::vector<std::string> strings, std::string_view delimiter, std::string_view prefix, std::string_view suffix) {
	std::string out = "";
	auto it = strings.begin();

	if (strings.size() > 0) {
		out += prefix;
		out += *it;
		it++;
	}

	while (it != strings.end()) {
		out += delimiter;
		out += *it;
		it++;
	}

	if (strings.size() > 0) {
		out += suffix;
	}
	return out;
}

std::string util::byteSizeToString(int64_t bytes) {
	if (bytes < 0)
		return "N/A";
	else if (bytes < 1024ull)
		return std::format("{} bytes", bytes);
	else if (bytes < 1024ull * 1024ull) 
		return std::format("{:.1f} kb", bytes / 1024.0);
	else 
		return std::format("{:.1f} Mb", bytes / 1048576.0);
}