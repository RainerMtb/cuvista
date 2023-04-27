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

#include <format>
#include "ProgressDisplay.hpp"
#include "Util.hpp"

double ProgressDisplay::progressPercent() {
	double percentage = std::numeric_limits<double>::quiet_NaN();
	if (data.frameCount != 0) {
		double p = std::abs((300.0 * data.status.frameInputIndex + 100.0 * data.status.frameWriteIndex) / 4.0 / data.frameCount);
		percentage = std::clamp(p, 0.0, 100.0);
	}
	return percentage;
}

bool ProgressDisplay::isDue(bool forceUpdate) {
	auto t = std::chrono::steady_clock::now();
	if (forceUpdate || t - timePoint > interval) {
		timePoint = t;
		return true;

	} else {
		return false;
	}
}

bool ProgressDisplay::isFinite() {
	return data.frameCount != 0;
}
