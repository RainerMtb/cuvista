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
#include "OfxUtil.hpp"

#if defined(_WIN64)
#define LIBRARY_EXPORT extern "C" __declspec(dllexport)
#else
#define LIBRARY_EXPORT extern "C"
#endif

namespace ofx {

	static OfxPlugin plugin = {};

	void setHostFcn(OfxHost* host) {
		debugLogger().open("tcp://10.0.0.1:5555"); //must reopen the logger, host resets the library ???
		debugLogger().log("set host");
		ofx::host = host;
	}

	OfxStatus mainEntryFcn(const char* action, const void* handle, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
		debugLogger().format("action {}", action);
		OfxImageEffectHandle effect = (OfxImageEffectHandle) handle;
		OfxStatus status = kOfxStatReplyDefault;
		std::string actionString = action;

		if (actionString == kOfxActionLoad) {
			//load plugin, fetch and store suites
			propertySuite = (OfxPropertySuiteV1*) host->fetchSuite(host->host, kOfxPropertySuite, 1);
			imageEffectSuite = (OfxImageEffectSuiteV1*) host->fetchSuite(host->host, kOfxImageEffectSuite, 1);
			pluginState = PluginState::LOADED;
			status = kOfxStatOK;

		} else if (actionString == kOfxActionUnload) {
			//unload plugin
			pluginState = PluginState::UNLOADED;
			status = kOfxStatOK;

		} else if (actionString == kOfxActionDescribe) {
			//describe plugin to host, set global parameters for all clips
			OfxPropertySetHandle effectProps;
			imageEffectSuite->getPropertySet(effect, &effectProps);

			if (pluginState != PluginState::LOADED) {
				errorLogger().logError("plugin must be loaded here", ErrorSource::OFX);
				debugLogger().log("plugin must be loaded here");
			}

			propertySuite->propSetString(effectProps, kOfxPropLabel, 0, "Cuvista");
			propertySuite->propSetString(effectProps, kOfxImageEffectPluginPropGrouping, 0, "Cuvista - Cuda Video Stabilizer");

			propertySuite->propSetString(effectProps, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextFilter);
			propertySuite->propSetString(effectProps, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthByte); //seems to be ignored by host anyway?????
			propertySuite->propSetString(effectProps, kOfxImageEffectPropSupportedPixelDepths, 1, kOfxBitDepthShort);
			propertySuite->propSetString(effectProps, kOfxImageEffectPropSupportedPixelDepths, 2, kOfxBitDepthFloat); //values can be outside [0..1]
			propertySuite->propSetString(effectProps, kOfxImageEffectPluginRenderThreadSafety, 0, kOfxImageEffectRenderFullySafe);
			propertySuite->propSetInt(effectProps, kOfxImageEffectPluginPropHostFrameThreading, 0, 0); //work on one complete frame

			pluginState = PluginState::DESCRIBED;
			status = kOfxStatOK;

		} else if (actionString == kOfxImageEffectActionDescribeInContext) {
			//describe plugin to host, set parameters for specific context
			OfxPropertySetHandle props;
			imageEffectSuite->clipDefine(effect, "Output", &props);

			if (pluginState != PluginState::DESCRIBED) {
				errorLogger().logError("plugin must be described here", ErrorSource::OFX);
				debugLogger().log("plugin must be described here");
				return kOfxStatErrFatal;
			}
			std::string str = propGetString(inArgs, kOfxImageEffectPropContext);
			if (str != kOfxImageEffectContextFilter) {
				errorLogger().format(ErrorSource::OFX, "unsupported context {}", str);
				debugLogger().format("unsupported context {}", str);
				return kOfxStatFailed;
			}

			// set the component types we can handle on out output
			propertySuite->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
			//propertySuite->propSetString(props, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentAlpha);
			//propertySuite->propSetString(props, kOfxImageEffectPropSupportedComponents, 2, kOfxImageComponentRGB);

			// define the mandated single source clip
			imageEffectSuite->clipDefine(effect, "Source", &props);

			// set the component types we can handle on our main input
			propertySuite->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
			//propertySuite->propSetString(props, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentAlpha);
			//propertySuite->propSetString(props, kOfxImageEffectPropSupportedComponents, 2, kOfxImageComponentRGB);

			pluginState = PluginState::DESCRIBED_IN_CONTEXT;
			status = kOfxStatOK;

		} else if (actionString == kOfxActionCreateInstance) {
			//init plugin instance, multiple instances are active at a time
			status = kOfxStatOK;

		} else if (actionString == kOfxActionDestroyInstance) {
			//destroy plugin instance
			status = kOfxStatOK;

		} else if (actionString == kOfxImageEffectActionRender) {
			//render a frame, this action is a kOfxImageEffectAction
			status = render(effect, inArgs, outArgs);
		}

		return status;
	}
}

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
		ofx::plugin = {
			.pluginApi = kOfxImageEffectPluginApi,
			.apiVersion = 1,
			.pluginIdentifier = "RainerMtb.cuvista",
			.pluginVersionMajor = 1,
			.pluginVersionMinor = (unsigned int) (cuvistaVersion.major * 10000 + cuvistaVersion.minor * 100 + cuvistaVersion.patch),
			.setHost = &ofx::setHostFcn,
			.mainEntry = &ofx::mainEntryFcn
		};
		return &ofx::plugin;

	} else {
		return nullptr;
	}
}


//-------------------------------------------------------------------------

namespace ofx {

	//render a frame
	OfxStatus render(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
		OfxTime time;
		OfxRectI renderWindow;
		OfxStatus status = kOfxStatOK;
		propertySuite->propGetDouble(inArgs, kOfxPropTime, 0, &time);
		propertySuite->propGetIntN(inArgs, kOfxImageEffectPropRenderWindow, 4, &renderWindow.x1);
		debugLogger().format("render at {} window x {} to {}, y {} to {}", time, renderWindow.x1, renderWindow.x2, renderWindow.y1, renderWindow.y2);

		// fetch main input clip
		OfxImageClipHandle srcClip;
		imageEffectSuite->clipGetHandle(effect, "Source", &srcClip, NULL);
		OfxPropertySetHandle srcImg = nullptr;
		status = imageEffectSuite->clipGetImage(srcClip, time, NULL, &srcImg);
		if (status != kOfxStatOK) {
			debugLogger().log("error: no input image");
			return status;
		}

		// fetch output clip
		OfxImageClipHandle destClip;
		imageEffectSuite->clipGetHandle(effect, "Output", &destClip, NULL);
		OfxPropertySetHandle destImg = nullptr;
		status = imageEffectSuite->clipGetImage(destClip, time, NULL, &destImg);
		if (status != kOfxStatOK) {
			debugLogger().log("error: no output image");
			return status;
		}

		// read source image
		int srcRowBytes;
		OfxRectI srcBounds;
		void* srcPtr = nullptr;
		propertySuite->propGetInt(srcImg, kOfxImagePropRowBytes, 0, &srcRowBytes);
		propertySuite->propGetIntN(srcImg, kOfxImagePropBounds, 4, &srcBounds.x1);
		propertySuite->propGetPointer(srcImg, kOfxImagePropData, 0, &srcPtr);
		
		int h = srcBounds.y2 - srcBounds.y1;
		int w = srcBounds.x2 - srcBounds.x1;
		float* srcData = reinterpret_cast<float*>(srcPtr);
		std::string pixelDepth = propGetString(srcImg, kOfxImageEffectPropPixelDepth);
		OfxImageFloat srcImage(h, w, srcRowBytes / sizeof(float), srcData);
		//srcImage.saveBmpColor("f:/image.bmp");
		//std::ofstream file("f:/file.dat", std::ios::binary); file.write(reinterpret_cast<char*>(srcData), srcRowBytes * h);

		// write destination image
		int destRowBytes;
		OfxRectI destBounds;
		void* destPtr = nullptr;
		propertySuite->propGetInt(destImg, kOfxImagePropRowBytes, 0, &destRowBytes);
		propertySuite->propGetIntN(destImg, kOfxImagePropBounds, 4, &destBounds.x1);
		propertySuite->propGetPointer(destImg, kOfxImagePropData, 0, &destPtr);
		OfxImageFloat destImage(destBounds.y2 - destBounds.y1, destBounds.x2 - destBounds.x1, destRowBytes / sizeof(float), reinterpret_cast<float*>(destPtr));

		srcImage.copyTo(destImage);
		destImage.gray();

		// release images
		if (srcImg) imageEffectSuite->clipReleaseImage(srcImg);
		if (destImg) imageEffectSuite->clipReleaseImage(destImg);

		return kOfxStatOK;
	}


	std::string propGetString(OfxPropertySetHandle handle, const char* id) {
		char* cstr;
		propertySuite->propGetString(handle, id, 0, &cstr);
		return cstr;
	}
}
