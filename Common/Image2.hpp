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

class ImagePPM;
class ImageBGR;
class ImageRGBA;


//image from CoreMat
class ImageYuvMatFloat : public im::ImageMatShared<float> {

public:
	ImageYuvMatFloat(int h, int w, int stride, float* y, float* u, float* v);

	ImagePPM& toPPM(ImagePPM& dest, ThreadPoolBase& pool = defaultPool) const;

	ImageRGBA& toRGBA(ImageRGBA& dest, ThreadPoolBase& pool = defaultPool) const;
};


//image from 8bit data
class ImageMatYuv8 : public im::ImageMatShared<uint8_t> {

public:
	ImageMatYuv8(int h, int w, int stride, uint8_t* y, uint8_t* u, uint8_t* v);
};


//planar image in rgb
class ImageRGBplanar : public im::ImageBase<unsigned char> {

public:
	//allocate frame given height, width, and stride
	ImageRGBplanar(int h, int w);

	//default constructor produces invalid image
	ImageRGBplanar();

	std::vector<unsigned char> getColorData(const Color& color) const override;
};


//packed rgba
class ImageRGBA : public im::ImagePacked<unsigned char> {

public:
	ImageRGBA(int h, int w, int stride, unsigned char* data);

	ImageRGBA(int h, int w, int stride);

	ImageRGBA(int h, int w);

	ImageRGBA();

	void copyTo(ImageRGBA& dest, size_t r0, size_t c0, ThreadPoolBase& pool = defaultPool) const;

	bool saveAsColorBMP(const std::string& filename) const;

	std::vector<unsigned char> getColorData(const Color& color) const override;
};


//planar yuv 8bit image
class ImageYuv : public im::ImageBase<unsigned char> {

public:
	ImageYuv(int h, int w, int stride);

	ImageYuv(int h, int w, size_t stride);

	ImageYuv(int h, int w);

	ImageYuv();

	virtual unsigned char* data();

	virtual const unsigned char* data() const;

	ImageYuv copy() const;

	//convert to NV12 format and copy pixeldata to provided array
	void toNV12(std::vector<unsigned char>& nv12, size_t strideNV12, ThreadPoolBase& pool = defaultPool) const;

	//convert to NV12 format and return pixeldata
	std::vector<unsigned char> toNV12(size_t strideNV12) const;

	//convert to NV12 format
	std::vector<unsigned char> toNV12() const;

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

	std::vector<unsigned char> getColorData(const Color& color) const override;
};


//packed 8bit in order BGR
class ImageBGR : public im::ImagePacked<unsigned char> {

public:
	ImageBGR(int h, int w);

	ImageBGR();

	bool saveAsColorBMP(const std::string& filename) const;

	ImageYuv toYUV() const;

	static ImageBGR readFromBMP(const std::string& filename);

	std::vector<unsigned char> getColorData(const Color& color) const override;
};


class ImagePPM : public im::ImagePacked<unsigned char> {

public:
	static inline int headerSize = 19;

	ImagePPM(int h, int w);

	ImagePPM();

	const unsigned char* data() const override;

	unsigned char* data() override;

	size_t stridedSize() const override;

	size_t stridedByteSize() const override;

	const unsigned char* header() const;

	bool saveAsPPM(const std::string& filename) const;

	bool saveAsColorBMP(const std::string& filename) const;
};
