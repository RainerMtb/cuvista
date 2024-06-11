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

#include "Image.hpp"
#include <format>

class ImagePPM;
class ImageBGR;
class ImageRGBA;


//image from CoreMat
class ImageYuvMat : public im::ImageMatShared<float> {

public:
	ImageYuvMat(int h, int w, int stride, float* y, float* u, float* v) :
		ImageMatShared(h, w, stride, y, u, v) {}

	ImagePPM& toPPM(ImagePPM& dest, ThreadPoolBase& pool = defaultPool) const;

	ImageRGBA& toRGBA(ImageRGBA& dest, ThreadPoolBase& pool = defaultPool) const;
};


//planar image in rgb
class ImageRGBplanar : public im::ImageBase<unsigned char> {

public:
	//allocate frame given height, width, and stride
	ImageRGBplanar(int h, int w) :
		ImageBase(h, w, w, 3) {}

	//default constructor produces invalid image
	ImageRGBplanar() :
		ImageRGBplanar(0, 0) {}
};


//packed rgba
class ImageRGBA : public im::ImagePacked<unsigned char> {

public:
	ImageRGBA(int h, int w, int stride, unsigned char* data) :
		ImagePacked(h, w, stride, 4, data, h* stride) {}

	ImageRGBA(int h, int w) :
		ImagePacked(h, w, 4 * w, 4, 4 * h * w) {}

	ImageRGBA() :
		ImageRGBA(0, 0) {}

	void copyTo(ImageRGBA& dest, size_t r0, size_t c0, ThreadPoolBase& pool = defaultPool) const;

	bool saveAsColorBMP(const std::string& filename) const;
};


//planar yuv 8bit image
class ImageYuv : public im::ImageBase<unsigned char> {

public:
	ImageYuv(int h, int w, int stride) :
		ImageBase<unsigned char>(h, w, stride, 3) {}

	ImageYuv(int h, int w, size_t stride) :
		ImageYuv(h, w, int(stride)) {}

	ImageYuv(int h, int w) :
		ImageYuv(h, w, w) {}

	ImageYuv() :
		ImageYuv(0, 0) {}

	virtual unsigned char* data();

	virtual const unsigned char* data() const;

	//downsample and copy pixeldata to given array in nv12 format
	void toNV12(std::vector<unsigned char>& nv12, size_t strideNV12, ThreadPoolBase& pool = defaultPool) const;

	//downsample and copy pixeldata
	std::vector<unsigned char> toNV12(size_t strideNV12) const;

	//convert NV12 array into YuvFrame
	ImageYuv& fromNV12(const std::vector<unsigned char>& nv12, size_t strideNV12);

	//convert to planar RGB format
	ImageRGBplanar& toRGB(ImageRGBplanar& dest, ThreadPoolBase& pool = defaultPool) const;

	ImageRGBplanar toRGB() const;

	//convert to BGR format for bmp files
	ImageBGR& toBGR(ImageBGR& dest, ThreadPoolBase& pool = defaultPool) const;

	ImageBGR toBGR() const;

	ImagePPM& toPPM(ImagePPM& dest, ThreadPoolBase& pool = defaultPool) const;

	ImagePPM toPPM() const;

	ImageRGBA& toRGBA(ImageRGBA& dest, ThreadPoolBase& pool = defaultPool) const;

	ImageRGBA toRGBA() const;

	void readFromPGM(const std::string& filename);

	//convert to ImageBGR and save to file, for repeated use BGR image should be preallocated
	bool saveAsColorBMP(const std::string& filename) const;
};


//packed 8bit in order BGR
class ImageBGR : public im::ImagePacked<unsigned char> {

public:
	ImageBGR(int h, int w) :
		ImagePacked(h, w, 3 * w, 3, 3 * h * w) {}

	ImageBGR() :
		ImageBGR(0, 0) {}

	bool saveAsColorBMP(const std::string& filename) const;
};


class ImagePPM : public im::ImagePacked<unsigned char> {

public:
	static inline int headerSize = 19;

	ImagePPM(int h, int w) :
		ImagePacked(h, w, 3 * w, 3, 3 * h * w + headerSize) 
	{
		//first 19 bytes are header for ppm format
		std::format_to_n(arrays.at(0).get(), headerSize, "P6 {:5} {:5} 255 ", w, h);
	}

	ImagePPM() :
		ImagePPM(0, 0) {}

	const unsigned char* data() const override {
		return arrays.at(0).get() + headerSize;
	}

	unsigned char* data() override {
		return arrays.at(0).get() + headerSize;
	}

	size_t size() const override {
		return 3ull * h * w;
	}

	size_t bytes() const override {
		return imageSize;
	}

	const unsigned char* header() const {
		return arrays.at(0).get();
	}

	bool saveAsPPM(const std::string& filename) const {
		std::ofstream os(filename, std::ios::binary);
		os.write(reinterpret_cast<const char*>(arrays.at(0).get()), size());
		return os.good();
	}

	bool saveAsColorBMP(const std::string& filename) const {
		ImageBGR bgr(h, w);
		copyTo(bgr, { 0, 1, 2 }, { 2, 1, 0 });
		return bgr.saveAsColorBMP(filename);
	}
};
