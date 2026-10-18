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

#include "ofxPluginContext.hpp"
#include "ofxMain.hpp"
#include "ofxGuiInterface.hpp"
#include "util.hpp"
#include "ErrorLogger.hpp"

using namespace ofx;

//void PluginContext::init(int w, int h) {
//	banner = loadBannerInstance(h, w, main.mBannerElement);
//	mData.w = w;
//	mData.h = h;
//	mData.deviceInfoCuda = main.deviceInfoCuda;
//	mData.deviceInfoOpenCl = main.deviceInfoOpenCl;
//	mData.collectDeviceInfo();
//}

std::string PluginContext::getString(OfxPropertySetHandle handle, const char* id, int index) {
	char* cstr;
	main.propertySuite->propGetString(handle, id, index, &cstr);
	return cstr;
}

double PluginContext::getDouble(OfxPropertySetHandle handle, const char* id, int index) {
	double d;
	main.propertySuite->propGetDouble(handle, id, index, &d);
	return d;
}

int PluginContext::getInt(OfxPropertySetHandle handle, const char* id, int index) {
	int i;
	main.propertySuite->propGetInt(handle, id, index, &i);
	return i;
}

OfxImageFloat PluginContext::getBanner(int h, int w) {
	ImageSize key = { h, w };
	std::lock_guard<std::mutex> lock(main.bannerMutex);
	if (auto iter = main.bannerMap.find(key); iter == main.bannerMap.end()) {
		OfxImageFloat banner = loadBannerInstance(h, w, main.mBannerElement);
		main.bannerMap.insert({ key, banner });
		debugLogger().format("new banner {}:{}, count={}", w, h, main.bannerMap.size());
		return banner;

	} else {
		return iter->second;
	}
}


//run stabilization on the clip
OfxStatus PluginContext::stabilize(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
	SptrGui gui = main.guiContext.gui;
	OfxStatus status = kOfxStatOK;

	if (gui->checkNewWindow()) {
		debugLogger().format("stabilize start on thread {}", threadId());
		OfxPropertySetHandle clipProperties;
		status = main.imageEffectSuite->clipGetPropertySet(srcClip, &clipProperties);

		double frameRange[2];
		status = main.propertySuite->propGetDoubleN(clipProperties, kOfxImageEffectPropFrameRange, 2, frameRange);
		double frameRate = getDouble(clipProperties, kOfxImageEffectPropFrameRate);
		double timelineStart = 0.0;
		double timelineEnd = 0.0;
		status = main.timelineSuite->getTimeBounds(effect, &timelineStart, &timelineEnd);
		int tempAccess = getInt(clipProperties, kOfxImageEffectPropTemporalClipAccess);
		double par = getDouble(clipProperties, kOfxImagePropPixelAspectRatio);
		std::string pixelDepth = getString(clipProperties, kOfxImageEffectPropPixelDepth);
		std::string components = getString(clipProperties, kOfxImageEffectPropComponents);

		debugLogger().format(
			"clip size {}:{}, frames {}:{}, timeline {}:{}, framerate {:.3f}, par {:.3f}, temp access {}", 
			clipW, clipH, frameRange[0], frameRange[1], timelineStart, timelineEnd, frameRate, par, tempAccess
		);

		gui->init();
		ImageRGBA inputImage(clipH, clipW);
		auto func = [&] {
			for (double time = frameRange[0]; time <= frameRange[1] && gui->isCancelled() == false && errorLogger().hasNoError(); time += 1.0) {
				//debugLogger().format("frame {}", time);

				OfxPropertySetHandle srcImg = nullptr;
				status = main.imageEffectSuite->clipGetImage(srcClip, time, NULL, &srcImg);
				if (srcImg == nullptr || status != kOfxStatOK) throw OfxException("no image, " + status);

				std::string pixelDepth = getString(srcImg, kOfxImageEffectPropPixelDepth);
				std::string components = getString(srcImg, kOfxImageEffectPropComponents);

				int srcRowBytes = getInt(srcImg, kOfxImagePropRowBytes, 0);
				int srcBounds[4];
				status = main.propertySuite->propGetIntN(srcImg, kOfxImagePropBounds, 4, srcBounds);
				void* srcPtr;
				status = main.propertySuite->propGetPointer(srcImg, kOfxImagePropData, 0, &srcPtr);
				int h = srcBounds[3] - srcBounds[1];
				int w = srcBounds[2] - srcBounds[0];
				uint8_t* srcData = reinterpret_cast<uint8_t*>(srcPtr);
				//debugLogger().format("time {} image {}:{} stride {} depth {} comp {}", time, w, h, srcRowBytes, pixelDepth, components);

				OfxImageByte srcImage(h, w, srcRowBytes, srcData);
				srcImage.copyTo(inputImage);
				double progress = (time - frameRange[0]) / (frameRange[1] - frameRange[0]);
				gui->updateProgress(progress, inputImage);

				status = main.imageEffectSuite->clipReleaseImage(srcImg);
			}
			gui->close(); //send signal to break the event loop
		};
		std::thread thread(func);
		gui->openProgress(); //start the event loop in the gui, blocking call
		thread.join();
		gui->shutdown();

		debugLogger().log("stabilize done");
		if (errorLogger().hasError()) {
			main.messageSuite->message(effect, kOfxMessageWarning, "", "Cuvista Error: %s", errorLogger().getErrorMessage().c_str());
			errorLogger().clear();
		}
	}
	return status;
}


//check and set clip parameters
OfxStatus PluginContext::getClipPreferences(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
	OfxRectD rectStart;
	main.imageEffectSuite->clipGetRegionOfDefinition(srcClip, 0, &rectStart);
	clipW = (int) (rectStart.x2 - rectStart.x1);
	clipH = (int) (rectStart.y2 - rectStart.y1);

	OfxStatus status = kOfxStatOK;
	OfxPropertySetHandle clipProperties;
	main.imageEffectSuite->clipGetPropertySet(srcClip, &clipProperties);
	main.propertySuite->propSetString(clipProperties, kOfxImageEffectPropSupportedPixelDepths, 2, kOfxBitDepthFloat);
	main.propertySuite->propSetString(clipProperties, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
	return status;
}


//render a frame
OfxStatus PluginContext::render(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
	OfxRectI renderWindow;
	OfxStatus status = kOfxStatOK;
	double time = getDouble(inArgs, kOfxPropTime, 0);
	main.propertySuite->propGetIntN(inArgs, kOfxImageEffectPropRenderWindow, 4, &renderWindow.x1);

	OfxPropertySetHandle srcImg = nullptr;
	status = main.imageEffectSuite->clipGetImage(srcClip, time, NULL, &srcImg);
	if (status != kOfxStatOK) {
		debugLogger().log("error: no input image");
		return kOfxStatFailed;
	}

	OfxPropertySetHandle destImg = nullptr;
	status = main.imageEffectSuite->clipGetImage(destClip, time, NULL, &destImg);
	if (status != kOfxStatOK) {
		debugLogger().log("error: no output image");
		return kOfxStatFailed;
	}

	// read source image
	OfxRectI srcBounds;
	void* srcPtr = nullptr;
	int srcRowBytes = getInt(srcImg, kOfxImagePropRowBytes, 0);
	main.propertySuite->propGetIntN(srcImg, kOfxImagePropBounds, 4, &srcBounds.x1);
	main.propertySuite->propGetPointer(srcImg, kOfxImagePropData, 0, &srcPtr);
	int srcH = srcBounds.y2 - srcBounds.y1;
	int srcW = srcBounds.x2 - srcBounds.x1;
	float* srcData = reinterpret_cast<float*>(srcPtr);
	OfxImageFloat srcImage(srcH, srcW, srcRowBytes / sizeof(float), srcData);

	OfxRectD rectStart;
	main.imageEffectSuite->clipGetRegionOfDefinition(srcClip, 0, &rectStart);
	int clipW = (int) (rectStart.x2 - rectStart.x1);
	int clipH = (int) (rectStart.y2 - rectStart.y1);

	debugLogger().format(
		"render context={} clip={}:{} frame={} src={}:{} window x={}:{}, y={}:{}, thread={}", 
		pluginIndex, clipW, clipH, time, srcW, srcH, renderWindow.x1, renderWindow.x2, renderWindow.y1, renderWindow.y2, threadId()
	);

	OfxRectI destBounds;
	void* destPtr = nullptr;
	int destRowBytes = getInt(destImg, kOfxImagePropRowBytes, 0);
	main.propertySuite->propGetIntN(destImg, kOfxImagePropBounds, 4, &destBounds.x1);
	main.propertySuite->propGetPointer(destImg, kOfxImagePropData, 0, &destPtr);
	int destH = destBounds.y2 - destBounds.y1;
	int destW = destBounds.x2 - destBounds.x1;

	// write destination image
	if (srcW == destW && srcH == destH) {
		if (dirtyFlag) {
			//copy input frame and inset banner

			//create destination image
			OfxImageFloat destImage(destH, destW, destRowBytes / sizeof(float), reinterpret_cast<float*>(destPtr));

			//get banner
			banner = getBanner(srcH, srcW);
			int bannerOffset = int(time * 4) % srcW;
			int bannerY = (srcH - banner.h()) / 2;

			//copy input to output
			if (main.imageEffectSuite->abort(effect) == false) {
				srcImage.copyTo(destImage);
			}

			//insert banner
			if (main.imageEffectSuite->abort(effect) == false) {
				banner.copyTo(0, bannerOffset, banner.h(), srcW, destImage, bannerY, 0, 0.6f);
			}

		} else {
			//produce output frame
		}

	} else {
		debugLogger().format("invalid frame dimensions, input {}:{}, output {}:{}", srcW, srcH, destW, destH);
	}

	// release images
	main.imageEffectSuite->clipReleaseImage(destImg);
	main.imageEffectSuite->clipReleaseImage(srcImg);
	return status;
}


std::string PluginContext::contextInfo(OfxImageEffectHandle effect) {
	double frameRange[2];
	double timelineStart = 0.0;
	double timelineEnd = 0.0;
	OfxStatus status;

	OfxPropertySetHandle clipProperties;
	status = main.imageEffectSuite->clipGetPropertySet(srcClip, &clipProperties);
	status = main.propertySuite->propGetDoubleN(clipProperties, kOfxImageEffectPropFrameRange, 2, frameRange);
	double frameRate = getDouble(clipProperties, kOfxImageEffectPropFrameRate);
	status = main.timelineSuite->getTimeBounds(effect, &timelineStart, &timelineEnd);

	std::string context = std::format(
		"Plugin Host:\n{}, Api Version {}\n\nCurrent Clip:\nFrame Size: {} x {}, Frames: {}, Frame Rate: {:.3f}\nCurrent Timeline from {} to {}\n\n",
		main.hostName, main.hostApiVersion, clipW, clipH, frameRange[1] - frameRange[0], frameRate, timelineStart, timelineEnd
	);
	return context;
}


void PluginContext::showInfo(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
	SptrGui gui = main.guiContext.gui;
	if (gui->checkNewWindow()) {
		debugLogger().log("info show");

		std::string strInfo = contextInfo(effect) + main.mData.showDeviceInfo();
		gui->init();
		gui->openInfo(strInfo);
		gui->shutdown();
		debugLogger().log("info done");
	}
}


//---------------------------------------------------------------------------------

InfoPrinter::InfoPrinter(SptrGui gui) :
	gui { gui }
{}

void InfoPrinter::print(const std::string& str) {
	gui->updateInfo(str);
}

void InfoPrinter::printNewLine() {
	gui->updateInfo("\n");
}
