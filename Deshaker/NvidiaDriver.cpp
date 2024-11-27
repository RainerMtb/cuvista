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

extern "C" {
#include "nvml.h"
}

// get nvidia driver version
// dynamically load nvml https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html#nvml-api-reference

typedef nvmlReturn_t (*ptr_nvmlInit_v2)(void);
typedef nvmlReturn_t (*ptr_nvmlSystemGetDriverVersion)(char* version, unsigned int length);
typedef nvmlReturn_t (*ptr_nvmlShutdown)(void);


static NvidiaDriverInfo probe(ptr_nvmlInit_v2 nvInit, ptr_nvmlSystemGetDriverVersion nvVersion, ptr_nvmlShutdown nvShutdown) {
	if (nvInit == nullptr || nvVersion == nullptr || nvShutdown == nullptr) {
		return { "", "error getting nvml functions" };
	}
	std::string warning = "";
	nvmlReturn_t status;

	status = nvInit();
	if (status != NVML_SUCCESS) warning = "error initializing nvml";

	const int siz = NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE;
	char version[siz];
	status = nvVersion(version, siz);
	if (status != NVML_SUCCESS) warning = "error getting nvidia driver version";

	nvShutdown();
	return { version, warning };
}

//-------------- Windows -------------

#if defined(_WIN64)

#include <ShlObj.h>
#include <filesystem>

using fpath = std::filesystem::path;

NvidiaDriverInfo probeNvidiaDriver() {
	PWSTR folderPath;
	HMODULE nvml = NULL;
	HRESULT result;
	
	//first folder to search
	result = SHGetKnownFolderPath(FOLDERID_ProgramFilesX64, 0, NULL, &folderPath);
	if (result == S_OK) {
		fpath nvmlPath = fpath(folderPath) / "NVIDIA Corporation" / "NVSMI" / "nvml.dll";
		nvml = LoadLibraryA(nvmlPath.string().c_str());
	}
	CoTaskMemFree(folderPath);

	//second folder to search
	if (nvml == NULL) {
		result = SHGetKnownFolderPath(FOLDERID_System, 0, NULL, &folderPath);
		if (result == S_OK) {
			fpath nvmlPath = fpath(folderPath) / "nvml.dll";
			nvml = LoadLibraryA(nvmlPath.string().c_str());
		}
		CoTaskMemFree(folderPath);
	}

	NvidiaDriverInfo info;
	if (nvml != NULL) {
		ptr_nvmlInit_v2 nvInit = (ptr_nvmlInit_v2) GetProcAddress(nvml, "nvmlInit_v2");
		ptr_nvmlSystemGetDriverVersion nvVersion = (ptr_nvmlSystemGetDriverVersion) GetProcAddress(nvml, "nvmlSystemGetDriverVersion");
		ptr_nvmlShutdown nvShutdown = (ptr_nvmlShutdown) GetProcAddress(nvml, "nvmlShutdown");
		info = probe(nvInit, nvVersion, nvShutdown);
		FreeLibrary(nvml);
	}

	return info;
}

//-------------- Linux -------------

#elif defined(__linux__)

NvidiaDriverInfo probeNvidiaDriver() {
	//nvidia-smi --query-gpu=driver_version --format=csv,noheader
	dlopen();
	dlclose();
}

//-------------- Other Systems -------------

#else

NvidiaDriverInfo probeNvidiaDriver() {
	return { "", "" };
}

#endif