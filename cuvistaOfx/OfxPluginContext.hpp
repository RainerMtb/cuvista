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

#include "ofxGuiInterface.hpp"
#include "ofxUtil.hpp"
#include "Trajectory.hpp"

namespace ofx {

	using SptrGui = std::shared_ptr<OfxGui>;

	class InfoPrinter : public util::MessagePrinter {

	private:
		SptrGui gui;

	public:
		InfoPrinter(SptrGui gui);
		void print(const std::string& str) override;
		void printNewLine() override;
	};

	class PluginContext {

	private:
		std::string contextInfo(OfxImageEffectHandle effect);

	public:
		int64_t mTimelineStart;
		int64_t mTimelineEnd;
		Trajectory mTrajectory;
		bool mDirtyFlag = true;
		OfxImageFloat mBanner;

		int mPluginIndex = 0;
		int mClipHeight;
		int mClipWidth;
		OfxImageClipHandle mSrcClip = nullptr;
		OfxImageClipHandle mDestClip = nullptr;

		OfxParamHandle mParamRadius = nullptr;
		OfxParamHandle mParamZoom = nullptr;

		OfxMatf4 mBackground;

		OfxImageFloat getBanner(int h, int w);

		OfxStatus getClipPreferences(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs);
		OfxStatus stabilize(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs);

		OfxStatus renderBegin(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs);
		OfxStatus render(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs);
		OfxStatus renderEnd(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs);

		void showInfo(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs);
	};
}