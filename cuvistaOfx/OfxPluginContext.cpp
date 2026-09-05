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

#include "OfxPluginContext.hpp"
#include "OfxUtil.hpp"
#include "util.hpp"
#include "ofxGuiInterface.hpp"


using namespace ofx;

//render a frame
void PluginContext::render(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
	OfxRectI renderWindow;
	OfxStatus status = kOfxStatOK;
	double time = getDouble(inArgs, kOfxPropTime, 0);
	main.propertySuite->propGetIntN(inArgs, kOfxImageEffectPropRenderWindow, 4, &renderWindow.x1);
	debugLogger().format("render frame {} window x={}:{}, y={}:{} on thread {}", time, renderWindow.x1, renderWindow.x2, renderWindow.y1, renderWindow.y2, threadId());

	OfxPropertySetHandle srcImg = nullptr;
	status = main.imageEffectSuite->clipGetImage(srcClip, time, NULL, &srcImg);
	if (status != kOfxStatOK) {
		debugLogger().log("error: no input image");
		return;
	}

	OfxPropertySetHandle destImg = nullptr;
	status = main.imageEffectSuite->clipGetImage(destClip, time, NULL, &destImg);
	if (status != kOfxStatOK) {
		debugLogger().log("error: no output image");
		return;
	}

	// read source image
	OfxRectI srcBounds;
	void* srcPtr = nullptr;
	int srcRowBytes = getInt(srcImg, kOfxImagePropRowBytes, 0);
	main.propertySuite->propGetIntN(srcImg, kOfxImagePropBounds, 4, &srcBounds.x1);
	main.propertySuite->propGetPointer(srcImg, kOfxImagePropData, 0, &srcPtr);
	int h = srcBounds.y2 - srcBounds.y1;
	int w = srcBounds.x2 - srcBounds.x1;
	float* srcData = reinterpret_cast<float*>(srcPtr);
	OfxImageFloat srcImage(h, w, srcRowBytes / sizeof(float), srcData);
	//std::string pixelDepth = propGetString(srcImg, kOfxImageEffectPropPixelDepth);

	// write destination image
	OfxRectI destBounds;
	void* destPtr = nullptr;
	int destRowBytes = getInt(destImg, kOfxImagePropRowBytes, 0);
	main.propertySuite->propGetIntN(destImg, kOfxImagePropBounds, 4, &destBounds.x1);
	main.propertySuite->propGetPointer(destImg, kOfxImagePropData, 0, &destPtr);
	OfxImageFloat destImage(destBounds.y2 - destBounds.y1, destBounds.x2 - destBounds.x1, destRowBytes / sizeof(float), reinterpret_cast<float*>(destPtr));

	srcImage.copyTo(destImage);
	destImage.gray();

	// release images
	if (srcImg) main.imageEffectSuite->clipReleaseImage(srcImg);
	if (destImg) main.imageEffectSuite->clipReleaseImage(destImg);
}


//run stabilization on the clip
void PluginContext::stabilize(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
	SptrGui gui = main.guiContext.gui;
	if (gui->checkNewWindow()) {
		OfxStatus status = kOfxStatOK;
		debugLogger().format("stabilize start on thread {}", threadId());
		OfxPropertySetHandle clipProperties;
		main.imageEffectSuite->clipGetPropertySet(srcClip, &clipProperties);
	
		double frameRange[2];
		main.propertySuite->propGetDoubleN(clipProperties, kOfxImageEffectPropFrameRange, 2, frameRange);
		OfxRectD rectStart;
		status = main.imageEffectSuite->clipGetRegionOfDefinition(srcClip, frameRange[0], &rectStart);
		OfxRectD rectEnd;
		status = main.imageEffectSuite->clipGetRegionOfDefinition(srcClip, frameRange[0], &rectEnd);
		this->w = (int) (rectStart.x2 - rectStart.x1);
		this->h = (int) (rectStart.y2 - rectStart.y1);
		debugLogger().format("clip frame size {}:{}, frame time {}:{}", w, h, frameRange[0], frameRange[1]);
	
		int tempAccess = getInt(clipProperties, kOfxImageEffectPropTemporalClipAccess);
		double par = getDouble(clipProperties, kOfxImagePropPixelAspectRatio);
		//std::string pixelDepth = getString(clipProperties, kOfxImageEffectPropPixelDepth);
		//std::string components = getString(clipProperties, kOfxImageEffectPropComponents);
		debugLogger().format("clip temporal access {} par {:.3f}", tempAccess, par);
	
		gui->init();
		ImageRGBA inputImage(h, w);
		auto func = [&] {
			for (double time = frameRange[0]; time <= frameRange[1] && gui->isCancelled() == false; time += 1.0) {
				//debugLogger().format("frame {}", time);
	
				try {
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
					debugLogger().format("time {} image {}:{} stride {} depth {} comp {}", time, w, h, srcRowBytes, pixelDepth, components);

					OfxImageByte srcImage(h, w, srcRowBytes, srcData);
					srcImage.copyTo(inputImage);
					double progress = (time - frameRange[0]) / (frameRange[1] - frameRange[0]);
					gui->updateProgress(progress, inputImage);

					main.imageEffectSuite->clipReleaseImage(srcImg);

				} catch (const OfxException e) {
					debugLogger().log(e.what());
	
				} catch (const std::exception e) {
					debugLogger().log(e.what());
	
				} catch (...) {
					debugLogger().log("unknown exception");
				}
			}
			gui->close(); //send signal to break the event loop
		};
		std::thread thread(func);
		gui->openProgress(); //start the event loop in the gui, blocking call
		thread.join();
		gui->shutdown();
		debugLogger().log("stabilize done");
	}
}


void PluginContext::showInfo(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
	SptrGui gui = main.guiContext.gui;
	if (gui->checkNewWindow()) {
		debugLogger().log("info show");

		std::stringstream ssInfo;
		main.mData.showDeviceInfo(ssInfo);

		gui->init();
		gui->openInfo(ssInfo.str(), main.hostName, main.hostApiVersion);
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
