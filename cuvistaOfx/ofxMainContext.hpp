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

#include <map>
#include <mutex>
#include "ofxGuiInterface.hpp"
#include "ofxUtil.hpp"
#include "MainData.hpp"

namespace ofx {

	enum class PluginState {
		STARTED,
		LOADED,
		DESCRIBED,
		DESCRIBED_IN_CONTEXT,
		UNLOADED,
		UNKNOWN,
	};

	struct ImageSize {
		int h, w;

		bool operator < (const ImageSize & other) const;
	};

	struct MainContext {

		std::map<int, std::string> ofxStatsMap = {
			{0,  "kOfxStatOK"},
			{1,  "kOfxStatFailed"},
			{2,  "kOfxStatErrFatal"},
			{3,  "kOfxStatErrUnknown"},
			{4,  "kOfxStatErrMissingHostFeature"},
			{5,  "kOfxStatErrUnsupported"},
			{6,  "kOfxStatErrExists"},
			{7,  "kOfxStatErrFormat"},
			{8,  "kOfxStatErrMemory"},
			{9,  "kOfxStatErrBadHandle"},
			{10, "kOfxStatErrBadIndex"},
			{11, "kOfxStatErrValue"},
			{12, "kOfxStatReplyYes"},
			{13, "kOfxStatReplyNo"},
			{14, "kOfxStatReplyDefault"},
			{15, "kOfxStatUnlicensed"},
		};

		std::mutex bannerMutex;
		std::map<ImageSize, OfxImageFloat> bannerMap;

		const OfxHost* host = nullptr;
		int pluginIndex = 0;
		PluginState pluginState = PluginState::UNKNOWN;

		OfxPropertySuiteV1* propertySuite = nullptr;
		OfxImageEffectSuiteV1* imageEffectSuite = nullptr;
		OfxParameterSuiteV1* parameterSuite = nullptr;
		OfxTimeLineSuiteV1* timelineSuite = nullptr;
		OfxMessageSuiteV1* messageSuite = nullptr;
		OfxMultiThreadSuiteV1* multiThreadSuite = nullptr;

		OfxGuiContext guiContext;
		std::string hostName;
		std::string hostApiVersion;
		std::vector<int> hostApiVersionNumbers;
		unsigned int cpuCount = 0;

		OfxImageFloat mBannerElement;
		MainData mData;

		bool guiLoadLibrary(OfxGuiContext& guiContext);
		void guiFreeLibrary(OfxGuiContext& guiContext);
		void setupParameters(OfxParamSetHandle paramSet);

		std::string getString(OfxPropertySetHandle handle, const char* id, int index = 0, std::source_location location = std::source_location::current());
		double getDouble(OfxPropertySetHandle handle, const char* id, int index = 0, std::source_location location = std::source_location::current());
		int getInt(OfxPropertySetHandle handle, const char* id, int index = 0, std::source_location location = std::source_location::current());

		OfxStatus actionLoad(OfxImageEffectHandle effect);
		OfxStatus actionDescribe(OfxImageEffectHandle effect);
		OfxStatus actionDescribeInContext(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs);
		OfxStatus actionCreateInstance(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs);
		OfxStatus actionUnload();
	};
}
