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

namespace ofx {

	using namespace util;

	class OfxGui {

	public:
		virtual void init() = 0;
		virtual void shutdown() = 0;
		virtual void showProgress() = 0;
		virtual void updateProgress(double progress) = 0;
		virtual void hideProgress() = 0;

		virtual ~OfxGui() = default;
	};

	struct OfxGuiContext {
		std::filesystem::path pluginPath;
		std::shared_ptr<DebugLogger> debugLogger = {};
		std::shared_ptr<OfxGui> gui = {};
	};
}
