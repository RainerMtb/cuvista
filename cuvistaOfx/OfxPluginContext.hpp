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

#include "ofxMain.hpp"
#include "ofxGuiInterface.hpp"

namespace ofx {

	using SptrGui = std::shared_ptr<OfxGui>;

	class InfoPrinter : public util::MessagePrinter {

	private:
		SptrGui gui;

	public:
		InfoPrinter(SptrGui gui);
		void print(const std::string& str) override;
		void printNewLine() override;
	};

	class PluginContext {

	public:
		int pluginIndex = 0;
		bool dirtyFlag = true;
		OfxImageClipHandle srcClip = nullptr;
		OfxImageClipHandle destClip = nullptr;

		OfxParamHandle paramRadius = nullptr;
		OfxParamHandle paramZoomMin = nullptr;
		OfxParamHandle paramZoomDynamic = nullptr;
		OfxParamHandle paramZoomMax = nullptr;

		int h = 0;
		int w = 0;
		OfxImageFloat banner;

		void render(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs);
		void stabilize(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs);
		void showInfo(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs);
	};
}