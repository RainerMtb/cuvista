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

class ImageBGR;
class ImagePPM;

class ImageYuv3 : public ImagePlanar<float> {

public:

	ImageYuv3(int h, int w, CoreMat<float>& y, CoreMat<float>& u, CoreMat<float>& v) :
		ImagePlanar(h, w, y.data(), u.data(), v.data()) {}

	ImagePPM& toPPM(ImagePPM& dest, ThreadPoolBase& pool = defaultPool) const;

	ImagePPM toPPM() const;
};


class ImageRGB : public ImagePlanar<unsigned char> {

public:

	//allocate frame given height, width, and stride
	ImageRGB(int h, int w) :
		ImagePlanar(h, w, w, 3) {}

	//default constructor produces invalid image
	ImageRGB() :
		ImageRGB(0, 0) {}
};


class ImageYuv : public ImagePlanar<unsigned char> {

public:
	//allocate frame given height, width, and stride
	ImageYuv(int h, int w, int stride, int planes) :
		ImagePlanar(h, w, stride, planes) {}

	//allocate frame given height, width, and stride
	ImageYuv(int h, int w, int stride) :
		ImageYuv(h, w, stride, 3) {}

	//allocate frame given height, width, and stride
	ImageYuv(int h, int w, size_t stride) :
		ImageYuv(h, w, int(stride)) {}

	ImageYuv(int h, int w) :
		ImageYuv(h, w, w) {}

	ImageYuv() :
		ImageYuv(0, 0, 0, 1) {}

	//downsample and copy pixeldata to given array in nv12 format
	void toNV12(std::vector<unsigned char>& nv12, size_t strideNV12) const;

	//downsample and copy pixeldata
	std::vector<unsigned char> toNV12(size_t strideNV12) const;

	//convert NV12 array into YuvFrame
	ImageYuv& fromNV12(const std::vector<unsigned char>& nv12, size_t strideNV12);

	//convert to planar RGB format
	ImageRGB& toRGB(ImageRGB& dest) const;

	ImageRGB toRGB() const;

	//convert to BGR format for bmp files
	ImageBGR& toBGR(ImageBGR& dest, ThreadPoolBase& pool = defaultPool) const;

	ImageBGR toBGR() const;

	ImagePPM& toPPM(ImagePPM& dest, ThreadPoolBase& pool = defaultPool) const;

	ImagePPM toPPM() const;

	//write grayscale image to pgm file
	bool saveAsPGM(const std::string& filename) const;

	void readFromPGM(const std::string& filename);

	//write grayscale image in bmp format, Y-U-V planes one after the other, including stride
	bool saveAsBMP(const std::string& filename) const;

	//convert to ImageBGR and save to file
	//for repeated use consider preallocating BGR image
	bool saveAsColorBMP(const std::string& filename) const;
};


template <class T> class ImagePacked : public ImageBase<T> {

public:

	ImagePacked(int h, int w, int stride, int planes) :
		ImageBase<T>(h, w, stride, planes) {}

	ImagePacked(int h, int w, int stride, int planes, int arraySize) :
		ImageBase<T>(h, w, stride, planes, arraySize) {}

	unsigned char* addr(size_t idx, size_t r, size_t c) override;

	const unsigned char* addr(size_t idx, size_t r, size_t c) const override;
};


class ImageBGR : public ImagePacked<unsigned char> {

public:

	ImageBGR(int h, int w) :
		ImagePacked(h, w, w, 3) {}

	ImageBGR() :
		ImageBGR(0, 0) {}

	bool saveAsBMP(const std::string& filename) const;
};


class ImagePPM : public ImagePacked<unsigned char> {

private:
	static inline int headerSize = 19;

public:

	ImagePPM(int h, int w);

	ImagePPM() :
		ImagePPM(0, 0) {}

	const unsigned char* header() const;

	const unsigned char* data() const override;

	unsigned char* data() override;

	size_t size() const;

	size_t sizeTotal() const;

	bool saveAsPGM(const std::string& filename) const;

	bool saveAsBMP(const std::string& filename) const;
};