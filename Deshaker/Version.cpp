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

#include "Version.hpp"
#include <regex>

CuvistaVersion CuvistaVersion::parse(const std::string& version) {
	std::regex pattern("^(\\d+)\\.(\\d+)\\.(\\d+)$");
	std::smatch matcher;
	CuvistaVersion out = {};

	if (std::regex_match(version, matcher, pattern)) {
		out.major = std::stoi(matcher[1]);
		out.minor = std::stoi(matcher[2]);
		out.patch = std::stoi(matcher[3]);
	}
	return out;
}

bool CuvistaVersion::operator < (const CuvistaVersion& other) {
	return std::tie(major, minor, patch) < std::tie(other.major, other.minor, other.patch);
}