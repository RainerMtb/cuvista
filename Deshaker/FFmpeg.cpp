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

#include "FFmpeg.hpp"
#include "ErrorLogger.hpp"


//-----------------------------------------------------------------
//---------- loading ffmpeg dynamically ---------------------------
//-----------------------------------------------------------------

#if defined(_WIN64)

#include <Windows.h>

namespace ff {

    HMODULE ffmpegLib;

    template <typename F> void loadFunction(HMODULE lib, const char* funcName, F& funcPtr) {
        FARPROC f = GetProcAddress(lib, funcName);
        if (f == NULL) throw std::exception(std::format("error loading function '{}' code {}", funcName, GetLastError()).c_str());
        funcPtr = (F) f;
    }

    int loadFFmpegLibrary() {
        int retval = 0;
        try {
            ffmpegLib = LoadLibraryA("cuvistaFFmpeg.dll");
            if (!ffmpegLib) throw std::exception("cannot load ffmpeg library");

            loadFunction(ffmpegLib, "versionsCompiled", versionsCompiled);
            loadFunction(ffmpegLib, "versionsRuntime", versionsRuntime);
            loadFunction(ffmpegLib, "createReader", createReader);
            loadFunction(ffmpegLib, "createWriter", createWriter);

        } catch (std::exception e) {
            errorLogger().logError(e.what(), ErrorSource::FFMPEG);
            retval = 1;

        } catch (...) {
            retval = 10;
        }
        return retval;
    }

    int freeFFmpegLibrary() {
        int retval = 0;
        try {
            retval = FreeLibrary(ffmpegLib); //returns non-zero on success
            if (retval != 0) {
                retval = GetLastError();
            }

        } catch (...) {
            retval = 1;
        }

        return retval;
    }
}

#endif