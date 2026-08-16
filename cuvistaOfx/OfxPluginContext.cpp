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

 //-------------------------------------------------------------------------

using namespace ofx;

//render a frame
void PluginContext::render(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
	OfxRectI renderWindow;
	OfxStatus status = kOfxStatOK;
	double time = getDouble(inArgs, kOfxPropTime, 0);
	main.propertySuite->propGetIntN(inArgs, kOfxImageEffectPropRenderWindow, 4, &renderWindow.x1);
	debugLogger().format("render at {} window x={}:{}, y={}:{}", time, renderWindow.x1, renderWindow.x2, renderWindow.y1, renderWindow.y2);

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
	//srcImage.saveBmpColor("f:/image.bmp");
	//std::ofstream file("f:/file.dat", std::ios::binary); file.write(reinterpret_cast<char*>(srcData), srcRowBytes * h);

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

void PluginContext::stabilize(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
	debugLogger().format("stabilize start {}", threadId());
	OfxPropertySetHandle clipProperties;
	main.imageEffectSuite->clipGetHandle(effect, "Source", &srcClip, &clipProperties);

	double frameRange[2];
	main.propertySuite->propGetDoubleN(clipProperties, kOfxImageEffectPropFrameRange, 2, frameRange);
	debugLogger().format("clip frames {}:{}", frameRange[0], frameRange[1]);

	OfxStatus status = kOfxStatOK;
	main.guiContext.gui->showProgress();
	for (double time = frameRange[0]; time <= frameRange[1]; time += 1.0) {
		//debugLogger().format("frame {}", time);

		OfxPropertySetHandle srcImg = nullptr;
		status = main.imageEffectSuite->clipGetImage(srcClip, time, NULL, &srcImg);
		main.guiContext.gui->updateProgress((time - frameRange[0]) / (frameRange[1] - frameRange[0]));

		//if (time == 50.0) {
		//	// read source image
		//	OfxRectI srcBounds;
		//	void* srcPtr = nullptr;
		//	int srcRowBytes = getInt(srcImg, kOfxImagePropRowBytes, 0);
		//	main.propertySuite->propGetIntN(srcImg, kOfxImagePropBounds, 4, &srcBounds.x1);
		//	main.propertySuite->propGetPointer(srcImg, kOfxImagePropData, 0, &srcPtr);
		//	int h = srcBounds.y2 - srcBounds.y1;
		//	int w = srcBounds.x2 - srcBounds.x1;
		//	uint8_t* srcData = reinterpret_cast<uint8_t*>(srcPtr);
		//	debugLogger().format("image {}x{} stride {} depth {}", w, h, srcRowBytes, getString(srcImg, kOfxImageEffectPropPixelDepth));
		//	OfxImageByte srcImage(h, w, srcRowBytes, srcData);
		//	srcImage.saveBmpColor("f:/image.bmp");
		//}

		if (srcImg) main.imageEffectSuite->clipReleaseImage(srcImg);
	}
	main.guiContext.gui->hideProgress();
	debugLogger().log("stabilize done");
}
