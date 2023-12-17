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

#include "ImageUtil.hpp"
#include "CoreMat.h"
#include "ThreadPool.h"

template <class T> class ImageMat : public CoreMat<T> {

public:

	ImageMat(T* data, int h, int w, int stride, int planeIdx);

	ImageMat() : ImageMat(nullptr, 0, 0, 0, 0) {}

	using CoreMat<T>::addr;
};


template <class T> class ImageBase {

public:

	int h, w, stride, numPlanes;
	int64_t index = -1;

protected:

	std::vector<T> array;
	static inline ThreadPoolBase defaultPool;

	int colorValue(T pixelValue) const;

	void plot(double x, double y, double a, ColorBase<T> color);

	void plot(int x, int y, double a, ColorBase<T> color);

	void plot4(double cx, double cy, double dx, double dy, double a, ColorBase<T> color);

	void convert8(ImageBase<unsigned char>& dest, int z0, int z1, int z2, ThreadPoolBase& pool = defaultPool) const;

public:

	virtual T* addr(size_t idx, size_t r, size_t c) = 0;

	virtual const T* addr(size_t idx, size_t r, size_t c) const = 0;

	ImageBase(int h, int w, int stride, int numPlanes, int arraySize);

	ImageBase(int h, int w, int stride, int numPlanes);

	ImageBase() : ImageBase(0, 0, 0, 0) {}

	//start of data array
	virtual T* data();

	//start of data array
	virtual const T* data() const;

	//access one pixel on plane idx (0..2) and row / col
	T& at(size_t idx, size_t r, size_t c);

	//read access one pixel on plane idx (0..2) and row / col
	const T& at(size_t idx, size_t r, size_t c) const;

	//set color value in one plane
	void setValues(int plane, T colorValue);

	//set color values per color plane
	void setValues(const ColorBase<T>& color);

	//equals operator
	virtual bool operator == (const ImageBase& other) const;

	//compute median value of differences
	double compareTo(const ImageBase& other) const;

	//size of data array in bytes
	size_t dataSizeInBytes() const;

	//imprint text
	void writeText(std::string_view text, int x0, int y0, int scaleX, int scaleY, ColorBase<T> fg, ColorBase<T> bg = { 0, 0, 0, 0.0 });

	void drawLine(double x0, double y0, double x1, double y1, ColorBase<T> color);

	void drawEllipse(double cx, double cy, double rx, double ry, ColorBase<T> color, bool fill = false);

	void drawCircle(double cx, double cy, double r, ColorBase<T> color, bool fill = false);

	void drawDot(double cx, double cy, double rx, double ry, ColorBase<T> color);
};


template <class T> class ImagePlanar : public ImageBase<T> {

protected:

	std::vector<ImageMat<T>> mats;

public:

	ImagePlanar(int h, int w, int stride, int numPlanes);

	ImagePlanar(int h, int w, T* y, T* u, T* v);

	ImagePlanar() : ImagePlanar(0, 0, 0, 0) {}

	//pointer to start of color plane
	T* plane(size_t idx);

	//pointer to start of color plane
	const T* plane(size_t idx) const;

	//access one pixel on plane idx (0..2) and row / col
	T* addr(size_t idx, size_t r, size_t c) override;

	//read access one pixel on plane idx (0..2) and row / col
	const T* addr(size_t idx, size_t r, size_t c) const override;

	T sample(size_t plane, double x, double y) const;

	//scale one plane
	void scaleTo(size_t srcPlane, ImageBase<T>& dest, size_t destPlane) const;

	//scale image
	void scaleTo(ImagePlanar<T>& dest) const;
};


class ImageBGR;
class ImagePPM;

class ImageYuvMat : public ImagePlanar<float> {

public:

	ImageYuvMat(int h, int w, CoreMat<float>& y, CoreMat<float>& u, CoreMat<float>& v) : ImagePlanar(h, w, y.data(), u.data(), v.data()) {}

	ImagePPM& toPPM(ImagePPM& dest, ThreadPoolBase& pool = defaultPool) const;

	ImagePPM toPPM() const;
};


class ImageRGB : public ImagePlanar<unsigned char> {

public:

	//allocate frame given height, width, and stride
	ImageRGB(int h, int w) : ImagePlanar(h, w, w, 3) {}

	//default constructor produces invalid image
	ImageRGB() : ImageRGB(0, 0) {}
};


class ImageYuv : public ImagePlanar<unsigned char> {

public:
	//allocate frame given height, width, and stride
	ImageYuv(int h, int w, int stride, int planes) : ImagePlanar(h, w, stride, planes) {}

	//allocate frame given height, width, and stride
	ImageYuv(int h, int w, int stride) : ImageYuv(h, w, stride, 3) {}

	//allocate frame given height, width, and stride
	ImageYuv(int h, int w, size_t stride) : ImageYuv(h, w, int(stride)) {}

	ImageYuv(int h, int w) : ImageYuv(h, w, w) {}

	ImageYuv() : ImageYuv(0, 0, 0) {}

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

	ImagePacked(int h, int w, int stride, int planes) : ImageBase<T>(h, w, stride, planes) {}

	ImagePacked(int h, int w, int stride, int planes, int arraySize) : ImageBase<T>(h, w, stride, planes, arraySize) {}

	unsigned char* addr(size_t idx, size_t r, size_t c) override;

	const unsigned char* addr(size_t idx, size_t r, size_t c) const override;
};


class ImageBGR : public ImagePacked<unsigned char> {

public:

	ImageBGR(int h, int w) : ImagePacked(h, w, w, 3) {}

	ImageBGR() : ImageBGR(0, 0) {}

	bool saveAsBMP(const std::string& filename) const;
};


class ImagePPM : public ImagePacked<unsigned char> {

private:
	static inline int headerSize = 19;

public:

	ImagePPM(int h, int w);

	ImagePPM() : ImagePPM(0, 0) {}

	const unsigned char* header() const;

	const unsigned char* data() const override;

	unsigned char* data() override;

	size_t size() const;

	size_t sizeTotal() const;

	bool saveAsPGM(const std::string& filename) const;
};