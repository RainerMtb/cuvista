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

#include "SystemStuff.hpp"


#if defined(_WIN64)
#include <windows.h>

//get console width from system calls
int getSystemConsoleWidth() {
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
	return csbi.srWindow.Right - csbi.srWindow.Left + 1;
}

//enable ansi formatting, seems to be necessary on some windows 10 systems
void enableAnsiSupport() {
	HANDLE h;
	h = GetStdHandle(STD_OUTPUT_HANDLE);
	if (h != INVALID_HANDLE_VALUE) {
		DWORD dwMode = 0;
		if (GetConsoleMode(h, &dwMode)) {
			dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
			SetConsoleMode(h, dwMode);
		}
	}
	h = GetStdHandle(STD_ERROR_HANDLE);
	if (h != INVALID_HANDLE_VALUE) {
		DWORD dwMode = 0;
		if (GetConsoleMode(h, &dwMode)) {
			dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
			SetConsoleMode(h, dwMode);
		}
	}
}

#elif defined(__linux__)

extern "C" {
#include <sys/ioctl.h>
#include <unistd.h>
}

//get console width from system calls
int getSystemConsoleWidth() {
	winsize w;
	ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
	return w.ws_col;
}

//enable ansi formatting, seems to be necessary on some windows 10 systems
void enableAnsiSupport() {}

#else

//get console width
int getSystemConsoleWidth() {
	return 80;
}

//enable ansi formatting, seems to be necessary on some windows 10 systems
void enableAnsiSupport() {}

#endif
