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

#include "ofxMain.hpp"
#include "Version.hpp"
#include "util.hpp"
#include "ErrorLogger.hpp"
#include "OfxPluginContext.hpp"
#include <fstream>

#if defined(_WIN64)
#define LIBRARY_EXPORT extern "C" __declspec(dllexport)
#else
#define LIBRARY_EXPORT extern "C"
#endif

namespace ofx {

	bool MainContext::isLoaded() const {
		return propertySuite && imageEffectSuite && parameterSuite;
	}

	OfxPlugin plugin = {};
	const OfxHost* host = nullptr;

	int pluginIndex = 0;
	PluginState pluginState = PluginState::UNKNOWN;

	//forward declare functions
	void setHostFcn(OfxHost* host);
	OfxStatus mainEntryFcn(const char* action, const void* handle, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs);

	//mandatory OpenFX library function
	LIBRARY_EXPORT OfxStatus OfxSetHost(const OfxHost* host) {
		ofx::host = host;
		return kOfxStatOK;
	}

	//mandatory OpenFX library function
	LIBRARY_EXPORT int OfxGetNumberOfPlugins() {
		debugLogger().open("tcp://10.0.0.1:5555");
		debugLogger().log("get number of plugins");
		return 1;
	}

	//mandatory OpenFX library function
	LIBRARY_EXPORT OfxPlugin* OfxGetPlugin(int nth) {
		debugLogger().format("get plugin #{}", nth);
		if (nth == 0) {
			plugin = {
				.pluginApi = kOfxImageEffectPluginApi,
				.apiVersion = 1,
				.pluginIdentifier = "RainerMtb.cuvista",
				.pluginVersionMajor = 1,
				.pluginVersionMinor = (unsigned int) (cuvistaVersion.major * 10000 + cuvistaVersion.minor * 100 + cuvistaVersion.patch),
				.setHost = &ofx::setHostFcn,
				.mainEntry = &ofx::mainEntryFcn
			};
			return &plugin;

		} else {
			return nullptr;
		}
	}

	void setHostFcn(OfxHost* host) {
		debugLogger().open("tcp://10.0.0.1:5555"); //must reopen the logger, host resets the library ???
		debugLogger().format("set host on thread {}", threadId());
		ofx::host = host;
		pluginState = PluginState::STARTED;
	}

	OfxStatus mainEntryFcn(const char* action, const void* handle, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
		//debugLogger().format("-- action {} --", action);
		OfxImageEffectHandle effect = (OfxImageEffectHandle) handle;
		std::string actionString = action;
		OfxStatus status = kOfxStatReplyDefault;

		if (actionString == kOfxActionLoad) { //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
			//load plugin, fetch and store suites
			main.propertySuite = (OfxPropertySuiteV1*) host->fetchSuite(host->host, kOfxPropertySuite, 1);
			main.imageEffectSuite = (OfxImageEffectSuiteV1*) host->fetchSuite(host->host, kOfxImageEffectSuite, 1);
			main.parameterSuite = (OfxParameterSuiteV1*) host->fetchSuite(host->host, kOfxParameterSuite, 1);

			//host info
			main.hostName = getString(host->host, kOfxPropName);
			int dim = 0;
			main.propertySuite->propGetDimension(host->host, kOfxPropAPIVersion, &dim);
			main.hostApiVersion = std::to_string(getInt(host->host, kOfxPropAPIVersion, 0));
			for (int i = 1; i < dim; i++) {
				main.hostApiVersion += ".";
				main.hostApiVersion += std::to_string(getInt(host->host, kOfxPropAPIVersion, i));
			}

			if (main.isLoaded()) {
				pluginState = PluginState::LOADED;
				status = kOfxStatOK;
				debugLogger().format("-- action Load on thread {}, plugin loaded, Host Name = {}, Api Version = {}", threadId(), main.hostName, main.hostApiVersion);

			} else {
				status = kOfxStatFailed;
			}

		} else if (actionString == kOfxActionDescribe) { //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
			//describe plugin to host, set global parameters for all clips
			if (pluginState != PluginState::LOADED) {
				errorLogger().logError("plugin must be loaded here", ErrorSource::OFX);
				debugLogger().log("plugin must be loaded here");
				status = kOfxStatFailed;
			}

			//OfxPropertySetHandle effectProps;
			OfxPropertySetHandle effectProps;
			main.imageEffectSuite->getPropertySet(effect, &effectProps);
			main.guiContext.pluginPath = std::filesystem::path(getString(effectProps, kOfxPluginPropFilePath)) / "Contents" / "Resources";
			main.guiContext.debugLogger = debugLoggerPtr;
			main.guiLoadLibrary(main.guiContext);
			if (!main.guiContext.gui) {
				errorLogger().logError("cannot load gui");
				debugLogger().log("cannot load gui");
			}

			main.imageEffectSuite->getPropertySet(effect, &effectProps);
			main.propertySuite->propSetString(effectProps, kOfxPropLabel, 0, "Cuvista");
			main.propertySuite->propSetString(effectProps, kOfxImageEffectPluginPropGrouping, 0, "Cuvista - Cuda Video Stabilizer");
			
			main.propertySuite->propSetString(effectProps, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextFilter);
			//setting pixel depths seems to be ignored by host anyway, always sends float?????
			main.propertySuite->propSetString(effectProps, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthByte);
			main.propertySuite->propSetString(effectProps, kOfxImageEffectPropSupportedPixelDepths, 1, kOfxBitDepthShort);
			main.propertySuite->propSetString(effectProps, kOfxImageEffectPropSupportedPixelDepths, 2, kOfxBitDepthFloat); //values can be outside [0..1]
			
			main.propertySuite->propSetInt(effectProps, kOfxImageEffectPropTemporalClipAccess, 0, 1);
			main.propertySuite->propSetString(effectProps, kOfxImageEffectPluginRenderThreadSafety, 0, kOfxImageEffectRenderInstanceSafe);
			main.propertySuite->propSetInt(effectProps, kOfxImageEffectPluginPropHostFrameThreading, 0, 0); //work on one complete frame

			pluginState = PluginState::DESCRIBED;
			status = kOfxStatOK;

		} else if (actionString == kOfxImageEffectActionDescribeInContext) { //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
			//describe plugin to host, set parameters for specific context
			if (pluginState != PluginState::DESCRIBED) {
				errorLogger().logError("plugin must be described here", ErrorSource::OFX);
				debugLogger().log("plugin must be described here");
				return kOfxStatErrFatal;
			}
			std::string str = getString(inArgs, kOfxImageEffectPropContext);
			if (str != kOfxImageEffectContextFilter) {
				errorLogger().format(ErrorSource::OFX, "unsupported context {}", str);
				debugLogger().format("unsupported context {}", str);
				return kOfxStatFailed;
			}

			main.mData.deviceInfoOpenCl = main.mData.probeOpenCl();
			main.mData.deviceInfoCuda = main.mData.probeCuda();
			main.mData.collectDeviceInfo();

			OfxPropertySetHandle props;
			// define the mandated single source clip
			main.imageEffectSuite->clipDefine(effect, kOfxImageEffectSimpleSourceClipName, &props);
			// set the component types we can handle on our main input
			main.propertySuite->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
			//propertySuite->propSetString(props, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentAlpha);
			//propertySuite->propSetString(props, kOfxImageEffectPropSupportedComponents, 2, kOfxImageComponentRGB);
			// request temporal access
			main.propertySuite->propSetInt(props, kOfxImageEffectPropTemporalClipAccess, 0, 1);

			main.imageEffectSuite->clipDefine(effect, kOfxImageEffectOutputClipName, &props);
			// set the component types we can handle on out output
			main.propertySuite->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
			//propertySuite->propSetString(props, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentAlpha);
			//propertySuite->propSetString(props, kOfxImageEffectPropSupportedComponents, 2, kOfxImageComponentRGB);

			//setup parameters
			OfxParamSetHandle paramSet;
			main.imageEffectSuite->getParamSet(effect, &paramSet);
			main.setupParameters(paramSet);

			pluginState = PluginState::DESCRIBED_IN_CONTEXT;
			status = kOfxStatOK;

		} else if (actionString == kOfxActionCreateInstance) { //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
			//init plugin instance, multiple instances are active at a time
			//here we do not get the correct number of frames in a clip
			OfxPropertySetHandle effectProps;
			main.imageEffectSuite->getPropertySet(effect, &effectProps);

			PluginContext* ctx = new PluginContext();
			OfxPropertySetHandle clipProperties;
			main.imageEffectSuite->clipGetHandle(effect, kOfxImageEffectSimpleSourceClipName, &ctx->srcClip, &clipProperties);
			main.imageEffectSuite->clipGetHandle(effect, kOfxImageEffectOutputClipName, &ctx->destClip, &clipProperties);

			OfxParamSetHandle paramSet;
			main.imageEffectSuite->getParamSet(effect, &paramSet);
			main.parameterSuite->paramGetHandle(paramSet, "radius", &ctx->paramRadius, 0);
			main.parameterSuite->paramGetHandle(paramSet, "zoom", &ctx->paramZoomMin, 0);

			ctx->pluginIndex = pluginIndex;
			pluginIndex++;
			pluginContextList.push_back(ctx);
			main.propertySuite->propSetPointer(effectProps, kOfxPropInstanceData, 0, ctx);
			debugLogger().format("-- action CreateInstance on thread {}, total instances = {}", threadId(), pluginContextList.size());
			status = kOfxStatOK;

		} else if (actionString == kOfxActionBeginInstanceChanged) { 
			status = kOfxStatReplyDefault;

		} else if (actionString == kOfxActionInstanceChanged) { 
			//handle button push
			if (getString(inArgs, kOfxPropName) == "stabilize" && getString(inArgs, kOfxPropChangeReason) == kOfxChangeUserEdited) {
				PluginContext* ctx = getPluginContext(effect);
				ctx->stabilize(effect, inArgs, outArgs);

			} else if (getString(inArgs, kOfxPropName) == "info" && getString(inArgs, kOfxPropChangeReason) == kOfxChangeUserEdited) {
				PluginContext* ctx = getPluginContext(effect);
				ctx->showInfo(effect, inArgs, outArgs);
			}
			status = kOfxStatOK;

		} else if (actionString == kOfxActionEndInstanceChanged) { 
			status = kOfxStatReplyDefault;

		} else if (actionString == kOfxImageEffectActionGetFramesNeeded) { 
			PluginContext* ctx = getPluginContext(effect);
			double time = getDouble(inArgs, kOfxPropTime);
			main.propertySuite->propSetDouble(outArgs, kOfxImageEffectPropFrameRange, 0, time);
			main.propertySuite->propSetDouble(outArgs, kOfxImageEffectPropFrameRange, 1, time);
			debugLogger().format("frames needed {}:{}", time, time);
			status = kOfxStatOK;

		} else if (actionString == kOfxImageEffectActionGetRegionsOfInterest) {
			status = kOfxStatReplyDefault;

		} else if (actionString == kOfxImageEffectActionBeginSequenceRender) {
			status = kOfxStatReplyDefault;

		} else if (actionString == kOfxImageEffectActionRender) { //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
			PluginContext* ctx = getPluginContext(effect);
			ctx->render(effect, inArgs, outArgs);
			status = kOfxStatOK;

		} else if (actionString == kOfxImageEffectActionEndSequenceRender) {
			status = kOfxStatReplyDefault;

		} else if (actionString == kOfxActionDestroyInstance) { //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
			//destroy plugin instance
			PluginContext* ctx = getPluginContext(effect);
			pluginContextList.remove(ctx);
			delete ctx;
			debugLogger().format("action DestroyInstance, total instances = {}", pluginContextList.size());
			status = kOfxStatOK;

		} else if (actionString == kOfxActionUnload) {
			//unload plugin
			if (pluginContextList.size() != 0) {
				errorLogger().logError("unloading while there are still effenct instances!");
				debugLogger().log("unloading while there are still effenct instances!");
			}
			main.guiFreeLibrary(main.guiContext);
			pluginState = PluginState::UNLOADED;
			status = kOfxStatOK;
			debugLogger().log("plugin unloaded, good bye");

		} else {
			debugLogger().format(">> action unhandled {} ##", action);
		}

		return status;
	}

	std::string getString(OfxPropertySetHandle handle, const char* id, int index) { return main.getString(handle, id, index); }

	double getDouble(OfxPropertySetHandle handle, const char* id, int index) { return main.getDouble(handle, id, index); }

	int getInt(OfxPropertySetHandle handle, const char* id, int index) { return main.getInt(handle, id, index); }

	//get the user context
	PluginContext* getPluginContext(OfxImageEffectHandle effect) {
		PluginContext* ctx = nullptr;
		OfxPropertySetHandle effectProps;
		main.imageEffectSuite->getPropertySet(effect, &effectProps);
		main.propertySuite->propGetPointer(effectProps, kOfxPropInstanceData, 0, (void**) &ctx);
		return ctx;
	}
}
