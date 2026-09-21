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

#include <list>
#include <string>
#include <unordered_map>

#include "ofxGuiInterface.hpp"
#include "OfxUtil.hpp"
#include "MainData.hpp"

namespace ofx {

	class ProgressContext;

	enum class PluginState {
		STARTED,
		LOADED,
		DESCRIBED,
		DESCRIBED_IN_CONTEXT,
		UNLOADED,
		UNKNOWN,
	};

	class PluginContext;

	inline std::list<PluginContext*> pluginContextList;
	PluginContext* getPluginContext(OfxImageEffectHandle effect);

	struct MainContext {

		std::unordered_map<int, std::string> ofxStatsMap = {
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

		OfxPropertySuiteV1* propertySuite = nullptr;
		OfxImageEffectSuiteV1* imageEffectSuite = nullptr;
		OfxParameterSuiteV1* parameterSuite = nullptr;
		OfxTimeLineSuiteV1* timelineSuite = nullptr;
		OfxMessageSuiteV1* messageSuite = nullptr;

		OfxImageFloat mBannerElement;

		OfxGuiContext guiContext;
		std::string hostName;
		std::string hostApiVersion;
		MainData mData;

		bool guiLoadLibrary(OfxGuiContext& guiContext);
		void guiFreeLibrary(OfxGuiContext& guiContext);
		void setupParameters(OfxParamSetHandle paramSet);

		bool isLoaded() const;

		std::string getString(OfxPropertySetHandle handle, const char* id, int index);
		double getDouble(OfxPropertySetHandle handle, const char* id, int index);
		int getInt(OfxPropertySetHandle handle, const char* id, int index);
	};

	inline MainContext main;

	std::string getString(OfxPropertySetHandle handle, const char* id, int index = 0);
	double getDouble(OfxPropertySetHandle handle, const char* id, int index = 0);
	int getInt(OfxPropertySetHandle handle, const char* id, int index = 0);
}
