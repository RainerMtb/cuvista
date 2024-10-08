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

#include <string>
#include <stdexcept>

class AVException : public std::runtime_error {

public:

    AVException() : std::runtime_error("") {}

    AVException(const std::string& msg) : std::runtime_error(msg.c_str()) {}

    AVException(const char* msg) : std::runtime_error(msg) {}
};


class CancelException : public AVException {

public:

    CancelException() : AVException("") {}

    CancelException(const std::string& msg) : AVException(msg) {}
};


class SilentQuitException : public AVException {

public:

    SilentQuitException() : AVException("") {}

    SilentQuitException(const std::string & msg) : AVException(msg) {}
};