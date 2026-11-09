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
#include <format>
#include <mutex>

namespace util {

    enum class DebugLoggerType {
        NONE,
        CONSOLE,
        STRING,
        TCP,
        FILE,
    };

    //logging misc messages
    struct DebugLogger {
        virtual ~DebugLogger() = default;

        std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

        static std::shared_ptr<DebugLogger> create(const std::string& logger);
        static void open(const std::string& logger);

        virtual void log(const std::string& msg) = 0;
        virtual std::string str() = 0;
        virtual DebugLoggerType type() = 0;

        std::string time() const;

        template <class... Args> void format(std::format_string<Args...> fmt, Args&&... args) {
            log(std::format(fmt, std::forward<Args>(args)...));
        }
    };

    struct DebugLoggerNull : public util::DebugLogger {
        void log(const std::string& msg) override;
        std::string str() override;
        DebugLoggerType type() override;
    };

    inline std::shared_ptr<DebugLogger> debugLoggerPtr = std::make_shared<DebugLoggerNull>();

}

//forward declaration for debugLogger();
util::DebugLogger& debugLogger();