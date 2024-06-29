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

enum class UserInputEnum {
	CONTINUE, END, QUIT, HALT,
};

class UserInput {

public:
	UserInputEnum mCurrentInput = UserInputEnum::CONTINUE;

	bool doContinue() const {
		return mCurrentInput == UserInputEnum::CONTINUE;
	}

	virtual void checkState() = 0;

	virtual void reset() {
		mCurrentInput = UserInputEnum::CONTINUE;
	}
};

class UserInputDefault : public UserInput {

public:
	void checkState() override {}
};