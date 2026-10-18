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
#include "ofxPluginContext.hpp"
#include "Version.hpp"
#include "util.hpp"

#if defined(_WIN64)
#define LIBRARY_EXPORT extern "C" __declspec(dllexport)
#else
#define LIBRARY_EXPORT extern "C"
#endif

namespace ofx {

	OfxPlugin plugin = {};

	void openLogger() {
		const size_t siz = 256;
		char buf[siz];
		size_t len = 0;
		getenv_s(&len, buf, siz, "CUVISTA_LOG");
		if (len > 0 && buf) {
			debugLogger().open(buf);
		}
	}

	//forward declare functions
	void setHostFcn(OfxHost* host);
	OfxStatus mainEntryFcn(const char* action, const void* handle, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs);

	//mandatory OpenFX library function
	LIBRARY_EXPORT OfxStatus OfxSetHost(const OfxHost* host) {
		main.host = host;
		return kOfxStatOK;
	}

	//mandatory OpenFX library function
	LIBRARY_EXPORT int OfxGetNumberOfPlugins() {
		openLogger();
		debugLogger().format("get number of plugins on thread {}", threadId());
		return 1;
	}

	//mandatory OpenFX library function
	LIBRARY_EXPORT OfxPlugin* OfxGetPlugin(int nth) {
		debugLogger().format("get plugin #{} on thread {}", nth, threadId());
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

	//host function callback
	void setHostFcn(OfxHost* host) {
		openLogger();
		debugLogger().format("set host on thread {}", threadId());
		main.host = host;
		main.pluginState = PluginState::STARTED;
	}

	//get the user context
	PluginContext* getPluginContext(OfxImageEffectHandle effect) {
		PluginContext* ctx = nullptr;
		OfxPropertySetHandle effectProps;
		main.imageEffectSuite->getPropertySet(effect, &effectProps);
		main.propertySuite->propGetPointer(effectProps, kOfxPropInstanceData, 0, (void**) &ctx);
		return ctx;
	}

	//main function called by host
	OfxStatus mainEntryFcn(const char* action, const void* handle, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
		//debugLogger().format("-- action {} --", action);
		OfxImageEffectHandle effect = (OfxImageEffectHandle) handle;
		std::string actionString = action;

		if (actionString == kOfxActionLoad) {
			return main.actionLoad(effect);

		} else if (actionString == kOfxActionDescribe) {
			return main.actionDescribe(effect);

		} else if (actionString == kOfxImageEffectActionDescribeInContext) {
			return main.actionDescribeInContext(effect, inArgs, outArgs);

		} else if (actionString == kOfxActionCreateInstance) {
			return main.actionCreateInstance(effect, inArgs, outArgs);

		} else if (actionString == kOfxActionEndInstanceChanged) { 
			return kOfxStatReplyDefault;

		} else if (actionString == kOfxImageEffectActionGetClipPreferences) {
			PluginContext* ctx = getPluginContext(effect);
			return ctx->getClipPreferences(effect, inArgs, outArgs);

		} else if (actionString == kOfxImageEffectActionGetFramesNeeded) { 
			double time = main.getDouble(inArgs, kOfxPropTime);
			main.propertySuite->propSetDouble(outArgs, kOfxImageEffectPropFrameRange, 0, time);
			main.propertySuite->propSetDouble(outArgs, kOfxImageEffectPropFrameRange, 1, time);
			//debugLogger().format("frames needed {}:{}", time, time);
			return kOfxStatOK;

		} else if (actionString == kOfxImageEffectActionGetRegionsOfInterest) {
			return kOfxStatReplyDefault;

		} else if (actionString == kOfxImageEffectActionBeginSequenceRender) {
			return kOfxStatReplyDefault;

		} else if (actionString == kOfxImageEffectActionRender) {
			PluginContext* ctx = getPluginContext(effect);
			return ctx->render(effect, inArgs, outArgs);

		} else if (actionString == kOfxImageEffectActionEndSequenceRender) {
			return kOfxStatReplyDefault;

		} else if (actionString == kOfxActionBeginInstanceChanged) {
			return kOfxStatReplyDefault;

		} else if (actionString == kOfxActionInstanceChanged) {
			debugLogger().format("-- action InstanceChanged on thread {}", threadId());
			std::string name = main.getString(inArgs, kOfxPropName);
			std::string reason = main.getString(inArgs, kOfxPropChangeReason);

			//handle button clicks
			if (name == "stabilize" && reason == kOfxChangeUserEdited) {
				PluginContext* ctx = getPluginContext(effect);
				return ctx->stabilize(effect, inArgs, outArgs);

			} else if (name == "info" && reason == kOfxChangeUserEdited) {
				PluginContext* ctx = getPluginContext(effect);
				ctx->showInfo(effect, inArgs, outArgs);
				return kOfxStatOK;
			}
			return kOfxStatReplyDefault;

		} else if (actionString == kOfxActionDestroyInstance) {
			//destroy plugin instance
			PluginContext* ctx = getPluginContext(effect);
			pluginContextList.remove(ctx);
			delete ctx;
			debugLogger().format("-- action DestroyInstance, total instances = {}", pluginContextList.size());
			return kOfxStatOK;

		} else if (actionString == kOfxActionUnload) {
			return main.actionUnload();

		} else {
			debugLogger().format(">> action unhandled {} ##", action);
			return kOfxStatReplyDefault;
		}
	}
}
