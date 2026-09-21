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

extern "C" {
#include "ofxParam.h"
#include "ofxImageEffect.h"
#include "ofxTimeLine.h"
#include "ofxMessage.h"
}

#include "ImageClasses.hpp"

namespace ofx {

	class OfxImageFloat : public im::ImageBase<float> {

	public:
		OfxImageFloat(int h, int w, int stride, float* data);
		OfxImageFloat(int h, int w, int stride);
		OfxImageFloat(int h, int w);
		OfxImageFloat();

		constexpr im::ImageType imageType() const override { return im::ImageType::RGBA; }

		void saveBmpColor(const std::string& filename) const override;

		void copyTo(int y, int x, int h, int w, ImageBase<float>& dest, int destY, int destX, float alpha, ThreadPoolBase& pool = defaultPool) const;
		void copyTo(int y, int x, int h, int w, ImageBase<float>& dest, int destY, int destX) const override;
		void copyTo(ImageBase<float>& dest, int destY, int destX) const override;
		void copyTo(ImageBase<float>& dest) const override;
	};


	class OfxImageByte : public im::Image8 {

	public:
		OfxImageByte(int h, int w, int stride, uint8_t* data);
		OfxImageByte(int h, int w, int stride);

		constexpr im::ImageType imageType() const override { return im::ImageType::RGBA; }

		void saveBmpColor(const std::string& filename) const override;
	};


	class OfxException : public std::runtime_error {

	public:

		OfxException() : std::runtime_error("") {}

		OfxException(const std::string& msg) : std::runtime_error(msg.c_str()) {}

		OfxException(const char* msg) : std::runtime_error(msg) {}
	};


	void handleStatus(OfxStatus status, const std::string& message);

	OfxImageFloat loadBannerElement();
	OfxImageFloat loadBannerInstance(int targetHeight, int targetWidth, const OfxImageFloat& element);
}