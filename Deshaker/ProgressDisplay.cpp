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
#include "MovieFrame.hpp"
#include "MovieReader.hpp"

double ProgressDisplay::progressPercent() {
	//progress is NaN when no frame count is available
	double percentage = std::numeric_limits<double>::quiet_NaN();
	int64_t frameCount = frame.mReader.frameCount;
	if (frameCount != 0) {
		//reading index is worth 3/4 towards progress and writing is worth 1/4
		double p = std::abs((300.0 * frame.mReader.frameIndex + 100.0 * frame.mWriter.frameIndex) / 4.0 / frameCount);
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
	return frame.mReader.frameCount != 0;
}
