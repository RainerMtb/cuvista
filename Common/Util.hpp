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
#include <span>

#include "ImageData.hpp"

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
            mInterval { mStart } 
        {}

        ConsoleTimer() :
            ConsoleTimer("timer")
        {}

        void interval(const std::string& name);

        ~ConsoleTimer();
    };


    //output sent to this ostream will be suppressed
    class NullOutstream : public std::ostream {

    public:
        NullOutstream() :
            std::ostream(nullptr) {}

        template <class T> NullOutstream& operator << (const T& value) { return *this; }
    };


    //interface to display text messages
    class MessagePrinter {

    public:
        virtual void print(const std::string& str) = 0;

        virtual void printNewLine() = 0;
    };


    //implementation of crc in 64bit
    class CRC64 {

    private:
        uint64_t poly = 0x95AC9329AC4BC9B5ull;
        uint64_t crc;
        uint64_t table[256];

        CRC64& addBytes(const unsigned char* data, size_t size);

    public:
        CRC64();

        CRC64& reset();

        template <class T> CRC64& addDirect(T data) { 
            return addBytes(reinterpret_cast<const unsigned char*>(&data), sizeof(T)); 
        }

        CRC64& addDirect(std::span<const unsigned char> data) { 
            return addBytes(data.data(), data.size()); 
        }

        template <class T> CRC64& add(const ImageData<T>& image) {
            for (int z = 0; z < image.planes(); z++) {
                for (int r = 0; r < image.height(); r++) {
                    for (int c = 0; c < image.width(); c++) {
                        addBytes(reinterpret_cast<const unsigned char*>(image.addr(z, r, c)), sizeof(T));
                    }
                }
            }
            return *this;
        }
        
        uint64_t result() const;

        bool operator == (const CRC64& other) const;
        bool operator == (uint64_t crc) const;
        friend std::ostream& operator << (std::ostream& os, const CRC64& crc);
    };


    //concat given strings by using delimiters
    std::string concatStrings(std::span<std::string_view> strings, std::string_view delimiter, std::string_view prefix, std::string_view suffix);

    std::string concatStrings(std::span<std::string_view> strings);

    std::vector<std::string> splitString(std::string_view str, std::string_view delimiter);

    //convert a number of bytes into more readable values, bytes / kb / Mb
    std::string byteSizeToString(int64_t bytes);

    //convert millis into readable string hh:mm:ss.fff
    std::string millisToTimeString(int64_t millis);

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

    //print content of collection to string
    template <class T> std::string collectionToString(std::vector<T> items, size_t maxItems) {
        size_t i = 0;
        std::ostringstream ss;

        ss << "{ ";
        for (; i + 1 < items.size() && i + 1 < maxItems; i++) {
            ss << items[i];
            ss << ", ";
        }
        if (items.size() > 0) {
            ss << items[i];
        }
        if (items.size() > i + 1) {
            ss << ", ...";
        }
        ss << " }";
        return ss.str();
    }

    //print content of collection to string
    template <class T> std::string collectionToString(std::vector<T> items) {
        return collectionToString(items, items.size());
    }

    //math stuff
    double sqr(double value);

    int alignValue(int numToAlign, int alignment);

    double cosd(double angleDegrees);

    double sind(double angleDegrees);

    double tand(double angleDegrees);
}