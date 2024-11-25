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

#if defined(BUILD_CUDA) && BUILD_CUDA == 0

//replace cuda stuff with empty shell when cuda is not included in build
inline void encodeNvData(const std::vector<unsigned char>& nv12, unsigned char* nvencPtr) {};
inline void getNvData(std::vector<unsigned char>& nv12, unsigned char* cudaNv12ptr) {};

#else

//include cuda stuff
#include "cuDeshaker.cuh"

#endif