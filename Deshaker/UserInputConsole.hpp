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

#include <vector>
#include <iostream>
#include "UserInput.hpp"
#include "KeyboardInput.hpp"

class UserInputConsole : public UserInput {

private:
	std::ostream& console;

public:
	UserInputConsole(std::ostream& console) : 
		console { console } {}

	void checkState() override {
		std::optional<char> key = getKeyboardInput();
		if (key) {
			switch (*key) {
			case 'e':
			case 'E':
				this->mCurrentInput = UserInputEnum::END;
				console << std::endl << "[e] command received. Stop reading input." << std::endl;
				break;
			case 'q':
			case 'Q':
				this->mCurrentInput = UserInputEnum::QUIT;
				console << std::endl << "[q] command received. Stop writing output." << std::endl;
				break;
			case 'x':
			case 'X':
				this->mCurrentInput = UserInputEnum::HALT;
				console << std::endl << "[x] command received. Terminating." << std::endl;
				break;
			}
		}
	}
};