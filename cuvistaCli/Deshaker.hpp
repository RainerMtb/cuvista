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

#include "CudaWriter.hpp"
#include "MovieReader.hpp"
#include "MovieWriter.hpp"
#include "MovieFrame.hpp"
#include "DummyFrame.hpp"
#include "ProgressDisplayConsole.hpp"

struct DeshakerResult {
	int statusCode = 0;
	std::string log;

	int64_t frameCount = 0;
	int64_t framesRead = 0;
	int64_t framesWritten = 0;
	int64_t framesEncoded = 0;
	int64_t bytesWritten = 0;
	int64_t bytesEncoded = 0;

	double secs = 0.0;
	std::string executorName;
	std::string executorNameShort;
	std::vector<TrajectoryItem> trajectory;
};

std::ostream& printError(std::ostream& os, const std::string& msg1);

DeshakerResult deshake(std::vector<std::string> argsInput, std::ostream* console, std::shared_ptr<MovieWriter> externalWriter);