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

//replace cuda stuff with empty shell when cuda is not included in build
#if defined(BUILD_CUDA) && BUILD_CUDA == 0

//dummy code to replace cuda stuff
#include "MovieFrame.hpp"
#include "Mat.hpp"
#include "cuDeshaker.cuh"

struct NvPacket {};

class NvEncoder {
public:
	NvEncoder(int cudaIndex) {}
	void probeEncoding(uint32_t* nvencVersionApi, uint32_t* nvencVersionDriver) {}
	void probeSupportedCodecs(DeviceInfoCuda& deviceInfoCuda) {}
};

#else

#include "NvEncoder.hpp"
#include "cuDeshaker.cuh"

#endif