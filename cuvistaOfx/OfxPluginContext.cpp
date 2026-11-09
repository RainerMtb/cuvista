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
#include "CpuFrame.hpp"

using namespace ofx;


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


//check and set clip parameters
OfxStatus PluginContext::getClipPreferences(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
	OfxRectD rectStart;
	main.imageEffectSuite->clipGetRegionOfDefinition(mSrcClip, 0, &rectStart);
	mClipWidth = (int) (rectStart.x2 - rectStart.x1);
	mClipHeight = (int) (rectStart.y2 - rectStart.y1);
	//debugLogger().format("clip preferences dim {}:{}", mClipW, mClipH);

	OfxPropertySetHandle clipProperties;
	handleStatus(main.imageEffectSuite->clipGetPropertySet(mSrcClip, &clipProperties));
	handleStatus(main.propertySuite->propSetString(clipProperties, kOfxImageEffectPropSupportedPixelDepths, 2, kOfxBitDepthFloat));
	handleStatus(main.propertySuite->propSetString(clipProperties, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA));
	return kOfxStatOK;;
}


//run stabilization on the clip
OfxStatus PluginContext::stabilize(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
	SptrGui gui = main.guiContext.gui;
	OfxStatus status = kOfxStatOK;

	if (gui->checkNewWindow()) {
		debugLogger().format("stabilize start on thread {}", threadId());
		OfxPropertySetHandle clipProperties;
		status = main.imageEffectSuite->clipGetPropertySet(mSrcClip, &clipProperties);

		double frameRange[2];
		status = main.propertySuite->propGetDoubleN(clipProperties, kOfxImageEffectPropFrameRange, 2, frameRange);
		double frameRate = main.getDouble(clipProperties, kOfxImageEffectPropFrameRate);
		double timelineStart = 0.0;
		double timelineEnd = 0.0;
		status = main.timelineSuite->getTimeBounds(effect, &timelineStart, &timelineEnd);
		int tempAccess = main.getInt(clipProperties, kOfxImageEffectPropTemporalClipAccess);
		double par = main.getDouble(clipProperties, kOfxImagePropPixelAspectRatio);
		std::string pixelDepth = main.getString(clipProperties, kOfxImageEffectPropPixelDepth);
		std::string components = main.getString(clipProperties, kOfxImageEffectPropComponents);

		double radsec;
		handleStatus(main.parameterSuite->paramGetValue(mParamRadius, &radsec));
		int radius = (int) std::round(radsec * frameRate);

		debugLogger().format(
			"clip dim {}:{}, frames {}:{}, timeline {}:{}, fps {:.3f}, par {:.3f}, temp {}, {}, {}", 
			mClipWidth, mClipHeight, frameRange[0], frameRange[1], timelineStart, timelineEnd, frameRate, par, tempAccess, components, pixelDepth
		);

		gui->init();
		ImageRGBA inputImage(mClipHeight, mClipWidth);
		mTrajectory.clear();
		bool isCancel = false;

		auto func = [&] {
			for (double time = timelineStart; time <= timelineEnd && isCancel == false && errorLogger().hasNoError(); time += 1.0) {
				//debugLogger().format("frame {}", time);
				OfxPropertySetHandle srcImg = nullptr;
				status = main.imageEffectSuite->clipGetImage(mSrcClip, time, NULL, &srcImg);
				if (srcImg == nullptr || status != kOfxStatOK) throw OfxException("no image, " + status);

				std::string pixelDepth = main.getString(srcImg, kOfxImageEffectPropPixelDepth);
				std::string components = main.getString(srcImg, kOfxImageEffectPropComponents);

				int srcRowBytes = main.getInt(srcImg, kOfxImagePropRowBytes);
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
				double progress = (time - timelineStart) / (timelineEnd - timelineStart);
				gui->updateProgress(progress, inputImage);

				status = main.imageEffectSuite->clipReleaseImage(srcImg);
				isCancel = gui->isCancelled() || main.imageEffectSuite->abort(effect);
			}

			for (int64_t idx = 0; idx < mTrajectory.size(); idx++) {
				mTrajectory.computeSmoothTransform(mClipHeight, mClipWidth, radius, 1.25, idx);
			}

			gui->close(); //send signal to break the event loop
		};
		std::thread thread(func);
		gui->openProgress(); //start the event loop in the gui, blocking call
		thread.join();
		gui->shutdown();

		if (errorLogger().hasError()) {
			main.messageSuite->message(effect, kOfxMessageWarning, "", "Cuvista Error: %s", errorLogger().getErrorMessage().c_str());
			errorLogger().clear();

		} else if (isCancel) {
			debugLogger().log("stabilize cancelled");

		} else {
			debugLogger().format("stabilize done, frames {}:{}", timelineStart, timelineEnd);
			mDirtyFlag = false;
			mTimelineStart = (int64_t) timelineStart;
			mTimelineEnd = (int64_t) timelineEnd;
		}
	}
	return status;
}


//setup buffers for rendering
OfxStatus PluginContext::renderBegin(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
	//set background matrix
	mBackground = OfxMatf4::allocate(mClipHeight, mClipWidth, mClipWidth * 4 * sizeof(float));
	mBackground.setValues(Float4(0, 0, 0, 255));
	return kOfxStatOK;
}


//clear buffers after for rendering
OfxStatus PluginContext::renderEnd(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
	//clear memory, reset to empty matrix
	mBackground = OfxMatf4();
	return kOfxStatOK;
}


//render a frame
OfxStatus PluginContext::render(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
	ConsoleTimer timer("render");
	OfxRectI renderWindow;
	OfxStatus status = kOfxStatOK;
	double time = main.getDouble(inArgs, kOfxPropTime);
	handleStatus(main.propertySuite->propGetIntN(inArgs, kOfxImageEffectPropRenderWindow, 4, &renderWindow.x1));
	double renderScale[2];
	handleStatus(main.propertySuite->propGetDoubleN(inArgs, kOfxImageEffectPropRenderScale, 2, renderScale));
	

	OfxPropertySetHandle srcImg = nullptr;
	status = main.imageEffectSuite->clipGetImage(mSrcClip, time, NULL, &srcImg);
	if (status != kOfxStatOK) {
		debugLogger().log("error: no input image");
		return kOfxStatFailed;
	}

	OfxPropertySetHandle destImg = nullptr;
	status = main.imageEffectSuite->clipGetImage(mDestClip, time, NULL, &destImg);
	if (status != kOfxStatOK) {
		debugLogger().log("error: no output image");
		return kOfxStatFailed;
	}

	// read source image
	OfxRectI srcBounds;
	void* srcPtr = nullptr;
	int srcRowBytes = main.getInt(srcImg, kOfxImagePropRowBytes);
	main.propertySuite->propGetIntN(srcImg, kOfxImagePropBounds, 4, &srcBounds.x1);
	main.propertySuite->propGetPointer(srcImg, kOfxImagePropData, 0, &srcPtr);
	int srcH = srcBounds.y2 - srcBounds.y1;
	int srcW = srcBounds.x2 - srcBounds.x1;
	float* srcData = reinterpret_cast<float*>(srcPtr);

	OfxRectD rectStart;
	main.imageEffectSuite->clipGetRegionOfDefinition(mSrcClip, 0, &rectStart);
	int clipW = (int) (rectStart.x2 - rectStart.x1);
	int clipH = (int) (rectStart.y2 - rectStart.y1);

	OfxRectI destBounds;
	void* destPtr = nullptr;
	int destRowBytes = main.getInt(destImg, kOfxImagePropRowBytes);
	main.propertySuite->propGetIntN(destImg, kOfxImagePropBounds, 4, &destBounds.x1);
	main.propertySuite->propGetPointer(destImg, kOfxImagePropData, 0, &destPtr);
	int destH = destBounds.y2 - destBounds.y1;
	int destW = destBounds.x2 - destBounds.x1;

	debugLogger().format(
		"render context={} clip={}:{} frame={} src={}:{} dest={}:{}, renderScale={:.3f}:{:.3f}, thread={}", 
		mPluginIndex, clipW, clipH, time, srcW, srcH, destW, destH, renderScale[0], renderScale[1], threadId()
	);

	// write destination image
	if (srcW == destW && srcH == destH) {
		if (mDirtyFlag) {
			//copy input frame and inset banner
			OfxImageFloat srcImage(srcH, srcW, srcRowBytes / sizeof(float), srcData);

			//create destination image
			float* destData = reinterpret_cast<float*>(destPtr);
			OfxImageFloat destImage(destH, destW, destRowBytes / sizeof(float), destData);

			//get banner
			mBanner = getBanner(srcH, srcW);
			int bannerOffset = int(time * 4) % srcW;
			int bannerY = (srcH - mBanner.h()) / 2;

			//copy input to output
			if (main.imageEffectSuite->abort(effect) == false) {
				srcImage.copyTo(destImage);
			}

			//insert banner
			if (main.imageEffectSuite->abort(effect) == false) {
				mBanner.copyTo(0, bannerOffset, mBanner.h(), srcW, destImage, bannerY, 0, 0.6f);
			}

		} else {
			//produce output frame
			double zoom;
			handleStatus(main.parameterSuite->paramGetValueAtTime(mParamZoom, time, &zoom));

			int64_t frameIndex = int64_t(time) - mTimelineStart;
			double dx = -40 * renderScale[0];
			double dy = -100 * renderScale[1];
			AffineTransform trf(1.05, -0.2, dx, dy);

			//source and destination frames
			Float4* srcData = reinterpret_cast<Float4*>(srcPtr);
			Matf4 srcImage = StridedMat<Float4>::fromData(srcData, srcH, srcW, srcRowBytes);
			float* destData = reinterpret_cast<float*>(destPtr);
			ImageFloat4 destImage(destH, destW, destRowBytes / sizeof(float), destData);

			//allocate buffer memory
			OfxMatf4 buffer1 = OfxMatf4::allocate(srcH, srcW, srcRowBytes);
			OfxMatf4 buffer2 = OfxMatf4::allocate(srcH, srcW, srcRowBytes);

			//transform context
			CpuFrame::TransformContext trc = { srcImage, destImage, mBackground, buffer1, buffer2, buffer1, main.filterKernel4, main.unsharp4 };
			CpuFrame::transform(trf, srcH, srcW, trc, main.mPool);
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
	status = main.imageEffectSuite->clipGetPropertySet(mSrcClip, &clipProperties);
	status = main.propertySuite->propGetDoubleN(clipProperties, kOfxImageEffectPropFrameRange, 2, frameRange);
	double frameRate = main.getDouble(clipProperties, kOfxImageEffectPropFrameRate);
	status = main.timelineSuite->getTimeBounds(effect, &timelineStart, &timelineEnd);

	std::string context = std::format(
		"Plugin Host:\n{}, Api Version {}\n\nCurrent Clip:\nFrame Size: {} x {}, Frames: {}, Frame Rate: {:.3f}\nCurrent Timeline from {} to {}\n\n",
		main.hostName, main.hostApiVersion, mClipWidth, mClipHeight, frameRange[1] - frameRange[0], frameRate, timelineStart, timelineEnd
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
