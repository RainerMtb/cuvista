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

#include <filesystem>
#include "util.hpp"

namespace im {
	class Image8;
}

namespace ofx {

	using namespace util;

	class OfxGui {

	public:
		//must be called on the application thread, start the gui
		virtual void init() = 0;

		virtual bool checkNewWindow() = 0;

		//must be called on the application thread, show info and tests
		virtual void openInfo(const std::string& infoString, const std::string& hostName, const std::string& hostVersion) = 0;

		//send signal to append a text line to the info box
		virtual void updateInfo(const std::string& infoString) = 0;

		//must be called on the application thread, show progress window and start event loop
		virtual void openProgress() = 0;

		//send signal to update progress
		virtual void updateProgress(double progress, const im::Image8& image) = 0;

		//send signal to close the window
		virtual void close() = 0;

		//must be called on the application thread, terminate gui
		virtual void shutdown() = 0;

		//probe if cancelling was requested via the gui
		virtual bool isCancelled() = 0;

		virtual ~OfxGui() = default;
	};

	struct OfxGuiContext {
		std::filesystem::path pluginPath;
		std::shared_ptr<DebugLogger> debugLogger = {};
		std::shared_ptr<OfxGui> gui = {};
	};
}
