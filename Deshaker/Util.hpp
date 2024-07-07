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

#include <chrono>
#include <string>
#include <vector>

//misc stuff
namespace util {

    //Timer Class to measure runtime and output to console
    class ConsoleTimer {

    private:
        std::chrono::time_point<std::chrono::steady_clock> mStart, mInterval;
        std::string mName;

    public:
        ConsoleTimer(std::string&& name) : 
            mName { name }, 
            mStart { std::chrono::steady_clock::now() },
            mInterval { mStart } {}

        void interval(const std::string& name);

        ~ConsoleTimer();
    };

    //concat givens strings by using delimiters
    std::string concatStrings(std::vector<std::string>& strings, std::string_view delimiter, std::string_view prefix, std::string_view suffix);

    //convert a number of bytes into more readable magnitude kb, Mb
    std::string byteSizeToString(int64_t bytes);

    //set timer start time
    void tickStart();

    //print elapsed time since start to console
    void tick(const std::string& message);
}