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

#include "KeyboardInput.hpp"
#include <Windows.h>

#undef max
#undef min

std::optional<char> getKeyboardInput() {
    HANDLE stdIn = GetStdHandle(STD_INPUT_HANDLE);
    INPUT_RECORD irec = {};
    DWORD cc;
    BOOL retval = false;

    while (GetNumberOfConsoleInputEvents(stdIn, &cc), cc > 0) {
        retval = ReadConsoleInput(stdIn, &irec, 1, &cc);
        if (irec.EventType == KEY_EVENT && irec.Event.KeyEvent.bKeyDown) {
            wchar_t wchr = irec.Event.KeyEvent.uChar.UnicodeChar;
            return (char) wchr;
        }
    }
    return std::nullopt;
}
