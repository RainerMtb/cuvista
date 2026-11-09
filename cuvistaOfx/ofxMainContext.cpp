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

#include "ofxMainContext.hpp"
#include "ofxMain.hpp"
#include "ofxPluginContext.hpp"
#include "CoreData.hpp"

using namespace ofx;

#if defined(_WIN64)

#include <Windows.h>

HMODULE guiLib;
typedef void (*LoadFunc)(OfxGuiContext& guiContext);

bool MainContext::guiLoadLibrary(OfxGuiContext& guiContext) {
	std::filesystem::current_path(guiContext.pluginPath);

	guiLib = LoadLibraryA("cuvistaOfxGui.dll");
	if (!guiLib) {
		return false;
	}
	FARPROC f = GetProcAddress(guiLib, "loadGui");
	if (!f) {
		return false;
	}
	LoadFunc loadFunc = (LoadFunc) f;
	loadFunc(guiContext);
	debugLogger().format("gui loaded");

	return true;
}

void MainContext::guiFreeLibrary(OfxGuiContext& guiContext) {
	guiContext.gui.reset();
	FreeLibrary(guiLib);
	debugLogger().log("gui unloaded");
	guiContext.debugLogger.reset();
}

#endif

bool ImageSize::operator < (const ImageSize& other) const {
	return std::tie(h, w) < std::tie(other.h, other.w);
}

std::string MainContext::getString(OfxPropertySetHandle handle, const char* id, int index, StatusHandling handling, std::source_location location) {
	char* cstr;
	handleStatus(propertySuite->propGetString(handle, id, index, &cstr), handling, "", location);
	return cstr;
}

double MainContext::getDouble(OfxPropertySetHandle handle, const char* id, int index, StatusHandling handling, std::source_location location) {
	double d;
	handleStatus(propertySuite->propGetDouble(handle, id, index, &d), handling, "", location);
	return d;
}

int MainContext::getInt(OfxPropertySetHandle handle, const char* id, int index, StatusHandling handling, std::source_location location) {
	int i;
	handleStatus(propertySuite->propGetInt(handle, id, index, &i), handling, "", location);
	return i;
}


//load plugin, first check message suite
OfxStatus MainContext::actionLoad(OfxImageEffectHandle effect) {
	OfxStatus status;
	messageSuite = (OfxMessageSuiteV1*) host->fetchSuite(host->host, kOfxMessageSuite, 1);
	if (messageSuite == nullptr) {
		status = kOfxStatFailed;

	} else {
		//fetch and store suites
		propertySuite = (OfxPropertySuiteV1*) host->fetchSuite(host->host, kOfxPropertySuite, 1);
		imageEffectSuite = (OfxImageEffectSuiteV1*) host->fetchSuite(host->host, kOfxImageEffectSuite, 1);
		parameterSuite = (OfxParameterSuiteV1*) host->fetchSuite(host->host, kOfxParameterSuite, 1);
		timelineSuite = (OfxTimeLineSuiteV1*) host->fetchSuite(host->host, kOfxTimeLineSuite, 1);
		multiThreadSuite = (OfxMultiThreadSuiteV1*) host->fetchSuite(host->host, kOfxMultiThreadSuite, 1);

		//host info
		try {
			hostName = getString(host->host, kOfxPropName);
			int dim = 0;
			handleStatus(propertySuite->propGetDimension(host->host, kOfxPropAPIVersion, &dim), StatusHandling::EXCEPTION);
			hostApiVersionNumbers.resize(dim);
			hostApiVersionNumbers[0] = getInt(host->host, kOfxPropAPIVersion, 0, StatusHandling::EXCEPTION);
			hostApiVersion = std::to_string(hostApiVersionNumbers[0]);
			for (int i = 1; i < dim; i++) {
				hostApiVersionNumbers[i] = getInt(host->host, kOfxPropAPIVersion, i, StatusHandling::EXCEPTION);
				hostApiVersion += ".";
				hostApiVersion += std::to_string(hostApiVersionNumbers[i]);
			}

			if (messageSuite && propertySuite && imageEffectSuite && parameterSuite && timelineSuite && multiThreadSuite) {
				pluginState = PluginState::LOADED;

			} else {
				throw OfxException("failed to load ofx suites");
			}

			multiThreadSuite->multiThreadNumCPUs(&cpuCount);
			//auto func = [] (unsigned int threadIndex, unsigned int threadMax, void* customArg) { debugLogger().format("thread index {}, pool size {}", threadIndex, threadMax); };
			//multiThreadSuite->multiThread(func, 4, nullptr);

			debugLogger().format(
				"-- action Load on thread {}, Host Name = {}, Api Version = {}, Cpu Count = {}", 
				threadId(), hostName, hostApiVersion, cpuCount
			);
			status = kOfxStatOK;

		} catch (OfxException e) {
			debugLogger().log(e.what());
			messageSuite->message(effect, kOfxMessageLog, "", e.what());
			status = kOfxStatFailed;
		}
	}
	return status;
}


//describe plugin to host, set global parameters for all clips
//called once on startup
OfxStatus MainContext::actionDescribe(OfxImageEffectHandle effect) {
	debugLogger().format("-- action Describe on thread {}", threadId());
	OfxStatus status;
	try {
		if (pluginState != PluginState::LOADED) {
			throw OfxException("plugin must be loaded here");
		}

		OfxPropertySetHandle effectProps;
		handleStatus(imageEffectSuite->getPropertySet(effect, &effectProps), StatusHandling::EXCEPTION);

		guiContext.pluginPath = std::filesystem::path(getString(effectProps, kOfxPluginPropFilePath)) / "Contents" / "Resources";
		guiContext.debugLogger = debugLoggerPtr;
		guiLoadLibrary(guiContext);
		if (!guiContext.gui) {
			throw OfxException("cannot load gui");
		}

		handleStatus(imageEffectSuite->getPropertySet(effect, &effectProps), StatusHandling::EXCEPTION);
		handleStatus(propertySuite->propSetString(effectProps, kOfxPropLabel, 0, "Cuvista"), StatusHandling::EXCEPTION);
		handleStatus(propertySuite->propSetString(effectProps, kOfxImageEffectPluginPropGrouping, 0, "Cuvista - Cuda Video Stabilizer"), StatusHandling::EXCEPTION);

		handleStatus(propertySuite->propSetString(effectProps, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextFilter), StatusHandling::EXCEPTION);
		//setting pixel depths seems to be ignored by host anyway, always sends float?????
		handleStatus(propertySuite->propSetString(effectProps, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthByte), StatusHandling::EXCEPTION);
		handleStatus(propertySuite->propSetString(effectProps, kOfxImageEffectPropSupportedPixelDepths, 1, kOfxBitDepthShort), StatusHandling::EXCEPTION);
		handleStatus(propertySuite->propSetString(effectProps, kOfxImageEffectPropSupportedPixelDepths, 2, kOfxBitDepthFloat), StatusHandling::EXCEPTION); //values can be outside [0..1]

		handleStatus(propertySuite->propSetString(effectProps, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA), StatusHandling::EXCEPTION);

		handleStatus(propertySuite->propSetInt(effectProps, kOfxImageEffectPropTemporalClipAccess, 0, 1), StatusHandling::EXCEPTION);
		handleStatus(propertySuite->propSetString(effectProps, kOfxImageEffectPluginRenderThreadSafety, 0, kOfxImageEffectRenderInstanceSafe), StatusHandling::EXCEPTION);
		handleStatus(propertySuite->propSetInt(effectProps, kOfxImageEffectPluginPropHostFrameThreading, 0, 0), StatusHandling::EXCEPTION); //work on one complete frame

		pluginState = PluginState::DESCRIBED;
		status = kOfxStatOK;

	} catch (OfxException e) {
		debugLogger().log(e.what());
		messageSuite->message(effect, kOfxMessageLog, "", e.what());
		status = kOfxStatFailed;
	}
	return status;
}


//describe plugin to host, set parameters for specific context
//called once on startup
OfxStatus MainContext::actionDescribeInContext(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
	debugLogger().format("-- action DescribeInContext on thread {}", threadId());
	OfxStatus status;
	try {
		if (pluginState != PluginState::DESCRIBED) {
			throw OfxException("plugin was not described before");
		}
		if (std::string str = getString(inArgs, kOfxImageEffectPropContext); str != kOfxImageEffectContextFilter) {
			debugLogger().format("unsupported context {}", str);
			return kOfxStatFailed;
		}

		OfxPropertySetHandle props;
		// define the mandated single source clip
		handleStatus(imageEffectSuite->clipDefine(effect, kOfxImageEffectSimpleSourceClipName, &props), StatusHandling::EXCEPTION);
		// set the component types we can handle for input
		handleStatus(propertySuite->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA), StatusHandling::EXCEPTION);
		// request temporal access
		handleStatus(propertySuite->propSetInt(props, kOfxImageEffectPropTemporalClipAccess, 0, 1), StatusHandling::EXCEPTION);

		handleStatus(imageEffectSuite->clipDefine(effect, kOfxImageEffectOutputClipName, &props), StatusHandling::EXCEPTION);
		// set the component types we can handle for output
		handleStatus(propertySuite->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA), StatusHandling::EXCEPTION);
		//propertySuite->propSetString(props, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentAlpha);
		//propertySuite->propSetString(props, kOfxImageEffectPropSupportedComponents, 2, kOfxImageComponentRGB);

		mData.probeOpenCl();
		mData.probeCuda();
		mData.collectDeviceInfo();

		//setup parameters
		OfxParamSetHandle paramSet;
		handleStatus(imageEffectSuite->getParamSet(effect, &paramSet), StatusHandling::EXCEPTION);
		setupParameters(paramSet);

		//load banner image
		mBannerElement = loadBannerElement();

		pluginState = PluginState::DESCRIBED_IN_CONTEXT;
		status = kOfxStatOK;

	} catch (OfxException e) {
		debugLogger().log(e.what());
		messageSuite->message(effect, kOfxMessageLog, "", e.what());
		status = kOfxStatFailed;
	}
	return status;
}


//init plugin instance, multiple instances are active at a time
//here we do not get the correct number of frames in a clip
OfxStatus MainContext::actionCreateInstance(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
	OfxStatus status;
	try {
		OfxPropertySetHandle effectProps;
		handleStatus(imageEffectSuite->getPropertySet(effect, &effectProps), StatusHandling::EXCEPTION);

		PluginContext* ctx = new PluginContext();
		OfxPropertySetHandle clipProperties;
		handleStatus(imageEffectSuite->clipGetHandle(effect, kOfxImageEffectSimpleSourceClipName, &ctx->mSrcClip, &clipProperties), StatusHandling::EXCEPTION);
		handleStatus(imageEffectSuite->clipGetHandle(effect, kOfxImageEffectOutputClipName, &ctx->mDestClip, &clipProperties), StatusHandling::EXCEPTION);

		//shortcut to parameters
		OfxParamSetHandle paramSet;
		handleStatus(imageEffectSuite->getParamSet(effect, &paramSet), StatusHandling::EXCEPTION);
		handleStatus(parameterSuite->paramGetHandle(paramSet, "radius", &ctx->mParamRadius, 0), StatusHandling::EXCEPTION);
		handleStatus(parameterSuite->paramGetHandle(paramSet, "zoom", &ctx->mParamZoom, 0), StatusHandling::EXCEPTION);

		//store context
		handleStatus(propertySuite->propSetPointer(effectProps, kOfxPropInstanceData, 0, ctx), StatusHandling::EXCEPTION);
		ctx->mPluginIndex = pluginIndex;
		pluginIndex++;
		pluginContextList.push_back(ctx);

		//here at this point clip parameters are useless, host reports fake values for frame count, frame size etc.
		debugLogger().format("-- action CreateInstance on thread {}, total instances = {}", threadId(), pluginContextList.size());
		status = kOfxStatOK;

	} catch (OfxException e) {
		debugLogger().log(e.what());
		messageSuite->message(effect, kOfxMessageLog, "", e.what());
		status = kOfxStatFailed;
	}
	return status;
}


OfxStatus MainContext::actionBeginSequenceRender(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
	PluginContext* ctx = getPluginContext(effect);
	return ctx->renderBegin(effect, inArgs, outArgs);
}


//render
OfxStatus MainContext::actionRender(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
	PluginContext* ctx = getPluginContext(effect);
	return ctx->render(effect, inArgs, outArgs);
}


OfxStatus MainContext::actionEndSequenceRender(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
	PluginContext* ctx = getPluginContext(effect);
	return ctx->renderEnd(effect, inArgs, outArgs);
}


//unload plugin
OfxStatus MainContext::actionUnload() {
	debugLogger().format("-- action Unload on thread {}, good bye", threadId());
	guiFreeLibrary(guiContext);
	pluginState = PluginState::UNLOADED;
	if (pluginContextList.size() != 0) {
		debugLogger().log("unloading while there are still effect instances!");
		return kOfxStatFailed;
	}
	return kOfxStatOK;
}


//set up effect parameters
void MainContext::setupParameters(OfxParamSetHandle paramSet) {
	OfxPropertySetHandle paramProps;

	//device selector
	handleStatus(parameterSuite->paramDefine(paramSet, kOfxParamTypeChoice, "device", &paramProps));
	handleStatus(propertySuite->propSetString(paramProps, kOfxPropLabel, 0, "Stabilization Device"));
	int deviceCount = (int) mData.deviceList.size();
	for (int i = 0; i < deviceCount; i++) {
		handleStatus(propertySuite->propSetString(paramProps, kOfxParamPropChoiceOption, i, mData.deviceList[i]->getNameShort().c_str()));
	}
	handleStatus(propertySuite->propSetInt(paramProps, kOfxParamPropDefault, 0, deviceCount - 1));
	handleStatus(propertySuite->propSetInt(paramProps, kOfxParamPropAnimates, 0, false));

	////group for parameters that require recalculation ------------------------
	//const char* paramGroup = "group";
	//parameterSuite->paramDefine(paramSet, kOfxParamTypeGroup, paramGroup, &paramProps);
	//propertySuite->propSetString(paramProps, kOfxPropLabel, 0, "Stabilization Parameters");

	//mode selector
	handleStatus(parameterSuite->paramDefine(paramSet, kOfxParamTypeChoice, "mode", &paramProps));
	handleStatus(propertySuite->propSetString(paramProps, kOfxPropLabel, 0, "Stabilization Mode"));
	handleStatus(propertySuite->propSetString(paramProps, kOfxParamPropChoiceOption, 0, "Single Pass Mode"));
	handleStatus(propertySuite->propSetString(paramProps, kOfxParamPropChoiceOption, 1, "Multi Pass - Anaylze 2x"));
	handleStatus(propertySuite->propSetString(paramProps, kOfxParamPropChoiceOption, 2, "Multi Pass - Anaylze 3x"));
	handleStatus(propertySuite->propSetString(paramProps, kOfxParamPropChoiceOption, 3, "Multi Pass - Anaylze 4x"));
	handleStatus(propertySuite->propSetInt(paramProps, kOfxParamPropAnimates, 0, false));
	//propertySuite->propSetString(paramProps, kOfxParamPropParent, 0, paramGroup);

	//temporal radius
	handleStatus(parameterSuite->paramDefine(paramSet, kOfxParamTypeDouble, "radius", &paramProps));
	handleStatus(propertySuite->propSetString(paramProps, kOfxPropLabel, 0, "Temporal Radius"));
	handleStatus(propertySuite->propSetInt(paramProps, kOfxParamPropDigits, 0, 2));
	handleStatus(propertySuite->propSetString(paramProps, kOfxParamPropDoubleType, 0, kOfxParamDoubleTypeX));
	handleStatus(propertySuite->propSetDouble(paramProps, kOfxParamPropDefault, 0, defaultParam.radsec));
	handleStatus(propertySuite->propSetDouble(paramProps, kOfxParamPropMin, 0, defaultParam.radsecMin));
	handleStatus(propertySuite->propSetDouble(paramProps, kOfxParamPropMax, 0, defaultParam.radsecMax));
	handleStatus(propertySuite->propSetInt(paramProps, kOfxParamPropAnimates, 0, false));
	handleStatus(propertySuite->propSetInt(paramProps, kOfxParamPropAnimates, 0, false));
	//propertySuite->propSetString(paramProps, kOfxParamPropParent, 0, paramGroup);

	//use dbscan
	handleStatus(parameterSuite->paramDefine(paramSet, kOfxParamTypeBoolean, "dbscan", &paramProps));
	handleStatus(propertySuite->propSetString(paramProps, kOfxPropLabel, 0, "Enable Dbscan"));
	handleStatus(propertySuite->propSetInt(paramProps, kOfxParamPropAnimates, 0, false));
	handleStatus(propertySuite->propSetInt(paramProps, kOfxParamPropDefault, 0, true));
	//propertySuite->propSetString(paramProps, kOfxParamPropParent, 0, paramGroup);

	//group end ----------------------------------------------------

	//zoom
	handleStatus(parameterSuite->paramDefine(paramSet, kOfxParamTypeDouble, "zoom", &paramProps));
	handleStatus(propertySuite->propSetString(paramProps, kOfxPropLabel, 0, "Image Zoom"));
	handleStatus(propertySuite->propSetString(paramProps, kOfxParamPropDoubleType, 0, kOfxParamDoubleTypeScale));
	handleStatus(propertySuite->propSetInt(paramProps, kOfxParamPropDigits, 0, 2));
	handleStatus(propertySuite->propSetDouble(paramProps, kOfxParamPropDefault, 0, 0.95));
	handleStatus(propertySuite->propSetDouble(paramProps, kOfxParamPropMin, 0, defaultParam.zoomMinRange));
	handleStatus(propertySuite->propSetDouble(paramProps, kOfxParamPropMax, 0, defaultParam.zoomMaxRange));
	handleStatus(propertySuite->propSetInt(paramProps, kOfxParamPropAnimates, 0, true));

	//button stabilize
	handleStatus(parameterSuite->paramDefine(paramSet, kOfxParamTypePushButton, "stabilize", &paramProps));
	handleStatus(propertySuite->propSetString(paramProps, kOfxPropLabel, 0, "   Stabilize   "));

	//button about
	handleStatus(parameterSuite->paramDefine(paramSet, kOfxParamTypePushButton, "info", &paramProps));
	handleStatus(propertySuite->propSetString(paramProps, kOfxPropLabel, 0, "   About   "));

	//parameterSuite->paramDefine(paramSet, kOfxParamTypeDouble, "zoomMin", &paramProps);
	//propertySuite->propSetString(paramProps, kOfxPropLabel, 0, "Image Zoom Min");
	//propertySuite->propSetString(paramProps, kOfxParamPropDoubleType, 0, kOfxParamDoubleTypeScale);
	//propertySuite->propSetDouble(paramProps, kOfxParamPropDefault, 0, defaultParam.zoomMin);
	//propertySuite->propSetDouble(paramProps, kOfxParamPropMin, 0, defaultParam.zoomMinRange);
	//propertySuite->propSetInt(paramProps, kOfxParamPropAnimates, 0, false);

	//parameterSuite->paramDefine(paramSet, kOfxParamTypeDouble, "zoomMax", &paramProps);
	//propertySuite->propSetString(paramProps, kOfxPropLabel, 0, "Image Zoom Max");
	//propertySuite->propSetString(paramProps, kOfxParamPropDoubleType, 0, kOfxParamDoubleTypeScale);
	//propertySuite->propSetDouble(paramProps, kOfxParamPropDefault, 0, defaultParam.zoomMax);
	//propertySuite->propSetDouble(paramProps, kOfxParamPropMaxn, 0, defaultParam.zoomMaxRange);
	//propertySuite->propSetInt(paramProps, kOfxParamPropAnimates, 0, false);
}
