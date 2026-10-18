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

#include "ofxGuiInterface.hpp"
#include "ofxUtil.hpp"
#include "MainData.hpp"

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

	private:
		std::string contextInfo(OfxImageEffectHandle effect);

		std::string getString(OfxPropertySetHandle handle, const char* id, int index = 0);
		double getDouble(OfxPropertySetHandle handle, const char* id, int index = 0);
		int getInt(OfxPropertySetHandle handle, const char* id, int index = 0);

	public:
		int pluginIndex = 0;
		bool dirtyFlag = true;
		OfxImageClipHandle srcClip = nullptr;
		OfxImageClipHandle destClip = nullptr;

		OfxParamHandle paramRadius = nullptr;
		OfxParamHandle paramZoomMin = nullptr;
		OfxParamHandle paramZoomDynamic = nullptr;
		OfxParamHandle paramZoomMax = nullptr;
		OfxImageFloat banner;
		int clipH;
		int clipW;

		OfxImageFloat getBanner(int h, int w);
		OfxStatus getClipPreferences(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs);
		OfxStatus render(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs);
		OfxStatus stabilize(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs);
		void showInfo(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs);
	};
}