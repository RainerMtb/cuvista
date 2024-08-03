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

#include <string>
#include "AVException.hpp"
#include "NvidiaDriver.hpp"

//nvapi only works on windows
#if defined(_WIN64)

extern "C" {
#include "nvapi/nvapi.h"
}

//get nvidia driver version
int probeNvidiaDriver() {
	NvAPI_Status status = NVAPI_OK;
	NvAPI_ShortString str;
	int retval = 0;

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
			retval = version;
		}
	}
	return retval;
}

#else
int probeNvidiaDriver() {
	//nvidia-smi --query-gpu=driver_version --format=csv,noheader
	return 0;
}

#endif