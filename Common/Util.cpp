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

#include <iostream>
#include <format>
#include <fstream>
#include <cassert>
#include <numbers>
#include <cmath>
#include <regex>
#include "Util.hpp"

namespace util {

	void ConsoleTimer::interval(const std::string& name) {
		auto interval = std::chrono::steady_clock::now();
		auto delta = std::chrono::duration_cast<std::chrono::microseconds>(interval - mInterval);
		mInterval = interval;
		std::cout << mName << ":" << name << "=" << delta.count() / 1000.0 << " ms" << std::endl;
	}

	ConsoleTimer::~ConsoleTimer() {
		auto stop = std::chrono::steady_clock::now();
		auto delta = std::chrono::duration_cast<std::chrono::microseconds>(stop - mStart);
		std::cout << mName << "=" << delta.count() / 1000.0 << " ms" << std::endl;
	}

	std::string concatStrings(std::span<std::string_view> strings) {
		return concatStrings(strings, "", "", "");
	}

	std::vector<std::string> splitString(std::string_view str, std::string_view delimiter) {
		std::regex rd(delimiter.cbegin(), delimiter.cend());
		std::string rs(str.cbegin(), str.cend());
		return { std::sregex_token_iterator(rs.begin(), rs.end(), rd, -1), std::sregex_token_iterator() };
	}

	std::string concatStrings(std::span<std::string_view> strings, std::string_view delimiter, std::string_view prefix, std::string_view suffix) {
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

	std::string byteSizeToString(int64_t bytes) {
		if (bytes < 0)
			return "N/A";
		else if (bytes < 1024ull)
			return std::format("{} bytes", bytes);
		else if (bytes < 1024ull * 1024ull)
			return std::format("{:.1f} kb", bytes / 1024.0);
		else
			return std::format("{:.1f} Mb", bytes / 1048576.0);
	}


	std::chrono::time_point tp = std::chrono::steady_clock::now();

	void tickStart() {
		tp = std::chrono::steady_clock::now();
	}

	void tick(const std::string& message) {
		auto time = std::chrono::steady_clock::now();
		uint64_t ns = (time - tp).count();

		std::string str = "";
		if (ns > 1e9) str = std::format("{:.1f} s: {}", ns / 1e9, message);
		else if (ns > 1e6) str = std::format("{:.1f} ms: {}", ns / 1e6, message);
		else if (ns > 1e3) str = std::format("{:.1f} us: {}", ns / 1e3, message);
		else str = std::format("{} ns: {}", ns, message);

		std::cout << str << std::endl;
	}

	const char* baseString = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

	std::string base64_encode(std::span<unsigned char> data) {
		std::string out;
		out.reserve(4 * data.size() / 3);

		int val = 0, valb = -6;
		for (unsigned char c : data) {
			val = (val << 8) + c;
			valb += 8;
			while (valb >= 0) {
				int idx = (val >> valb) & 0x3F;
				out.push_back(baseString[idx]);
				valb -= 6;
			}
		}
		if (valb > -6) {
			int idx = ((val << 8) >> (valb + 8)) & 0x3F;
			out.push_back(baseString[idx]);
		}
		while (out.size() % 4) {
			out.push_back('=');
		}
		return out;
	}

	void base64_encode(std::span<unsigned char> data, const std::string& filename) {
		std::string encoded = base64_encode(data);
		std::ofstream file(filename);
		size_t pos = 0;

		file.put('"');
		while (pos < encoded.size()) {
			file.put(encoded[pos]);
			pos++;
			if (pos % 76 == 0) {
				file.put('"');
				file.put('\n');
				file.put('"');
			}
		}
		file.put('"');
	}

	std::vector<unsigned char> base64_decode(const std::string& base64string) {
		std::vector<unsigned char> out;
		out.reserve(3 * base64string.size() / 4);

		std::vector<int> v(256, -1);
		for (int i = 0; i < 64; i++) v[baseString[i]] = i;

		int val = 0, valb = -8;
		for (unsigned char c : base64string) {
			if (v[c] == -1) break;
			val = (val << 6) + v[c];
			valb += 6;
			if (valb >= 0) {
				int ch = (val >> valb) & 0xFF;
				out.push_back((unsigned char) ch);
				valb -= 8;
			}
		}
		return out;
	}

	CRC64::CRC64() {
		reset();
		for (int i = 0; i < 256; i++) {
			uint64_t part = i;
			for (int j = 0; j < 8; j++) {
				if (part & 1) part = (part >> 1) ^ poly;
				else part >>= 1;
			}
			table[i] = part;
		}
	}

	CRC64& CRC64::reset() {
		crc = 0xFFFFFFFFFFFFFFFFull;
		return *this;
	}

	CRC64& CRC64::addBytes(const unsigned char* data, size_t size) {
		for (size_t i = 0; i < size; i++) {
			int idx = (crc ^ data[i]) & 0xFF;
			crc = table[idx] ^ (crc >> 8);
		}
		return *this;
	}

	uint64_t CRC64::result() const {
		return crc;
	}

	std::ostream& operator << (std::ostream& os, const CRC64& crc) {
		os << std::hex << crc.crc;
		return os;
	}

	bool CRC64::operator == (const CRC64& other) const {
		return result() == other.result();
	}

	bool CRC64::operator == (uint64_t crc) const {
		return result() == crc;
	}

	std::string millisToTimeString(int64_t millis) {
		int64_t sign = millis < 0 ? -1 : 1;
		millis = std::abs(millis);
		int64_t sec = millis / 1000;
		int64_t min = sec / 60;
		int64_t hrs = min / 60;

		millis %= 1000;
		sec %= 60;
		min %= 60;
		hrs %= 60;

		std::string timeString = "";
		if (hrs > 0) timeString = std::format("{}:{:02}:{:02}.{:03}", hrs * sign, min, sec, millis);
		else timeString = std::format("{:02}:{:02}.{:03}", min * sign, sec, millis);
		return timeString;
	}


	//----------------------------------------
	//---------- MATH STUFF ------------------
	//----------------------------------------

	double sqr(double value) {
		return value * value;
	}

	int alignValue(int numToAlign, int alignment) {
		assert(alignment && "factor must not be 0");
		return numToAlign >= 0 ? ((numToAlign + alignment - 1) / alignment) * alignment : numToAlign / alignment * alignment;
	}

	constexpr double PI = std::numbers::pi;

	double cosd(double angleDegrees) {
		return std::cos(angleDegrees * PI / 180.0);
	}

	double sind(double angleDegrees) {
		return std::sin(angleDegrees * PI / 180.0);
	}

	double tand(double angleDegrees) {
		return std::tan(angleDegrees * PI / 180.0);
	}
}