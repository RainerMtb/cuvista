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

#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <span>
#include <ostream>

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


    class NullOutstream : public std::ostream {

    public:
        NullOutstream() :
            std::ostream(nullptr) {}

        template <class T> NullOutstream& operator << (const T& value) { return *this; }
    };


    class MessagePrinter {

    public:
        virtual void print(const std::string& str) = 0;

        virtual void printNewLine() = 0;
    };


    class CRC64 {

    private:
        uint64_t poly = 0x95AC9329AC4BC9B5ull;
        uint64_t crc;
        uint64_t table[256];

    public:
        CRC64();
        template <class T> CRC64& add(T data) { return addBytes(reinterpret_cast<unsigned char*>(&data), sizeof(T)); }
        CRC64& reset();
        CRC64& addBytes(std::span<const unsigned char> data);
        CRC64& addBytes(const unsigned char* data, size_t size);
        uint64_t result() const;
    };


    //concat given strings by using delimiters
    std::string concatStrings(std::vector<std::string>& strings, std::string_view delimiter, std::string_view prefix, std::string_view suffix);

    //convert a number of bytes into more readable magnitude kb, Mb
    std::string byteSizeToString(int64_t bytes);

    //set timer start time
    void tickStart();

    //print elapsed time since start to console
    void tick(const std::string& message);

    //encode bytes to base64 string
    std::string base64_encode(std::span<unsigned char> data);

    //encode bytes to file in 76 column wide text
    void base64_encode(std::span<unsigned char> data, const std::string& filename);

    //decode base64 string to bytes
    std::vector<unsigned char> base64_decode(const std::string& base64string);
}