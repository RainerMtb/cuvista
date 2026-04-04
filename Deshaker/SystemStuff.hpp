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

#include <optional>
#include <format>
#include <memory>
#include <chrono>
#include <mutex>
#include <fstream>
#include "Util.hpp"

int getSystemConsoleWidth();

void enableAnsiSupport();

std::optional<char> getKeyboardInput();

void keepSystemAlive();


enum class UserInputEnum {
	NONE,
	CONTINUE,
	END,
	QUIT,
	HALT,
};


class UserInput {

public:
	virtual UserInputEnum checkState() = 0;
};

class UserInputDefault : public UserInput {

public:
	UserInputEnum checkState() override;
};


class UserInputConsole : public UserInput {

private:
	std::ostream& console;

public:
	UserInputConsole(std::ostream& console) :
		console{ console } {
	}

	UserInputEnum checkState() override;
};

namespace util {

	struct DebugLoggerConsole : public util::DebugLogger {
		void log(const std::string& msg) override;
		std::string str() override;
	};

	struct DebugLoggerString : public util::DebugLogger {
		std::stringstream ss;

		void log(const std::string& msg) override;
		std::string str() override;
	};

	struct DebugLoggerTcp : public util::DebugLogger {
		int mIsConnected = -1;

		DebugLoggerTcp(const char* ip, int port);
		~DebugLoggerTcp();

		void log(const std::string& msg) override;
		std::string str() override;
	};

	struct DebugLoggerFile : public util::DebugLogger {
		std::string filename;
		std::ofstream os;

		DebugLoggerFile(const std::string& filename);

		void log(const std::string& msg) override;
		std::string str() override;
	};
}
