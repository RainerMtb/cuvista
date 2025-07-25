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

 //handle keyboard input on console
#if defined(_WIN64)
//windows code
#include <Windows.h>

std::optional<char> getKeyboardInput() {
	HANDLE stdIn = GetStdHandle(STD_INPUT_HANDLE);
	INPUT_RECORD irec = {};
	DWORD cc;
	BOOL retval = false;

	std::optional<char> out = std::nullopt;
	while (GetNumberOfConsoleInputEvents(stdIn, &cc), cc > 0) {
		retval = ReadConsoleInput(stdIn, &irec, 1, &cc);
		if (irec.EventType == KEY_EVENT && irec.Event.KeyEvent.bKeyDown) {
			wchar_t wchr = irec.Event.KeyEvent.uChar.UnicodeChar;
			out = (char)wchr;
		}
	}
	return out;
}

#elif defined(__linux__)

//linux code
#include <stdio.h>
#include <sys/poll.h>
#include <termios.h>

std::optional<char> getKeyboardInput() {
	pollfd pfd = { 0, POLLIN };
	termios save;
	tcgetattr(0, &save);
	termios tc = save;
	tc.c_lflag &= ~(ICANON | ECHO);
	tc.c_cc[VMIN] = 0;
	tc.c_cc[VTIME] = 0;
	tcsetattr(0, TCSANOW, &tc);

	std::optional<char> out = std::nullopt;
	while (poll(&pfd, 1, 0) > 0) {
		int key = getchar();
		out = char(key);
	}
	tcsetattr(0, TCSAFLUSH, &save);
	return out;
}

#else
//no keyboard handling on unknown systems
std::optional<char> getKeyboardInput() {
	return std::nullopt;
}

#endif

//---------------------------------------------------------------

UserInputEnum UserInputDefault::checkState() { 
	return UserInputEnum::NONE; 
}

UserInputEnum UserInputConsole::checkState() {
	UserInputEnum state = UserInputEnum::NONE;
	std::optional<char> key = getKeyboardInput();
	if (key) {
		switch (*key) {
		case 'e':
		case 'E':
			state = UserInputEnum::END;
			//console << std::endl << "[e] command received. Stop reading input." << std::endl;
			break;
		case 'q':
		case 'Q':
			state = UserInputEnum::QUIT;
			//console << std::endl << "[q] command received. Stop writing output." << std::endl;
			break;
		case 'x':
		case 'X':
			state = UserInputEnum::HALT;
			//console << std::endl << "[x] command received. Terminating." << std::endl;
			break;
		}
	}
	return state;
}