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

#include "DeviceInfoCpu.hpp"
#include <thread>
#include "cpu_features/cpuinfo_x86.h"
#include <format>

using namespace cpu_features;

static const X86Info cpu = GetX86Info();

 //CPU Device Info
std::string DeviceInfoCpu::getName() const {
	X86Microarchitecture arch = GetX86Microarchitecture(&cpu);
	return std::format("CPU: {}, {} threads", cpu.brand_string, std::to_string(std::thread::hardware_concurrency()));
}

std::string getCpuName() {
	return cpu.brand_string;
}