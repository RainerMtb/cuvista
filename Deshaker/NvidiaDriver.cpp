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

#include "NvidiaDriver.hpp"
#include "AVException.hpp"
#include <iostream>

//get nvidia driver version
#if defined(_WIN64)

//nvapi only works on windows
extern "C" {
#include "nvapi.h"
}

std::string probeNvidiaDriver() {
	NvAPI_Status status = NVAPI_OK;
	NvAPI_ShortString str;
	std::string retval = "";

	status = NvAPI_Initialize();
	if (status == NVAPI_LIBRARY_NOT_FOUND) {
		//in this case NvAPI_GetErrorMessage() will only return an empty string

	} else if (status != NVAPI_OK) {
		NvAPI_GetErrorMessage(status, str);
		throw AVException("error initializing nvapi: " + std::string(str));

	} else {
		NvU32 version = 0;
		NvAPI_ShortString branch;
		status = NvAPI_SYS_GetDriverAndBranchVersion(&version, branch);
		if (status != NVAPI_OK) {
			NvAPI_GetErrorMessage(status, str);
			throw AVException("error getting driver version: " + std::string(str));

		} else {
			//we have valid nvidia driver
			retval = std::to_string(version / 100) + "." + std::to_string(version % 100);
		}
	}
	return retval;
}

#else

//on linux use nvml.h, NVIDIA Management Library
extern "C" {
#include "nvml.h"
}

std::string probeNvidiaDriver() {
	//nvidia-smi --query-gpu=driver_version --format=csv,noheader
	nvmlReturn_t status;
	status = nvmlInit_v2();
	if (status != NVML_SUCCESS) throw AVException("error loading nvml");

	const int siz = NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE;
	char version[siz];
	status = nvmlSystemGetDriverVersion(version, siz);
	if (status != NVML_SUCCESS) throw AVException("error getting nvidia driver version");

	nvmlShutdown();
	return version;
}

#endif