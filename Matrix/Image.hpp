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
#include "ThreadPoolBase.h"

class ImageGray;

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

	void yuvToRgb(ImageBase<unsigned char>& dest, int z0, int z1, int z2, ThreadPoolBase& pool = defaultPool) const;

	void shufflePlanes(ImageBase<T>& dest, int z0, int z1, int z2, ThreadPoolBase& pool = defaultPool) const;

	void copy2D(const std::vector<T>& src, std::vector<T>& dest, int rows, int cols, int srcStride, int destStride) const;

public:

	ImageBase(int h, int w, int stride, int numPlanes, int arraysize);

	ImageBase(int h, int w, int stride, int numPlanes) :
		ImageBase(h, w, stride, numPlanes, 1ull * h * stride * numPlanes) {}

	ImageBase() : 
		ImageBase(0, 0, 0, 0) {}

	virtual T* addr(size_t idx, size_t r, size_t c) = 0;

	virtual const T* addr(size_t idx, size_t r, size_t c) const = 0;

	//size of data array in bytes
	virtual size_t dataSizeInBytes() const;

	//access one pixel on plane idx (0..2) and row / col
	T& at(size_t idx, size_t r, size_t c);

	//read access one pixel on plane idx (0..2) and row / col
	const T& at(size_t idx, size_t r, size_t c) const;

	//set color value for all pixels in one plane
	void setValues(int plane, T colorValue);

	//set color values per color plane
	void setValues(const ColorBase<T>& color);

	//copy area from source image into this image
	void setArea(size_t r0, size_t c0, const ImageBase<T>& src, const ImageGray& mask);

	//set color values for one pixel
	void setPixel(size_t row, size_t col, std::vector<T> colors);

	//equals operator
	virtual bool operator == (const ImageBase& other) const;

	//compute median value of differences
	double compareTo(const ImageBase& other) const;

	//imprint text
	void writeText(std::string_view text, int x0, int y0, int scaleX, int scaleY, ColorBase<T> fg, ColorBase<T> bg = { 0, 0, 0, 0.0 });

	void drawLine(double x0, double y0, double x1, double y1, ColorBase<T> color, double alpha = 1.0);

	void drawEllipse(double cx, double cy, double rx, double ry, ColorBase<T> color, bool fill = false);

	void drawCircle(double cx, double cy, double r, ColorBase<T> color, bool fill = false);

	void drawDot(double cx, double cy, double rx, double ry, ColorBase<T> color);

	virtual void copyTo(ImageBase<T>& other) const;

	virtual void copyFrom(const ImageBase<T>& other);

private:

	double fpart(double d);

	double rfpart(double d);
};



template <class T> class ImageMat : public CoreMat<T> {

public:

	ImageMat(T* array, int h, int stride, int planeIdx) :
		CoreMat<T>(array, h, stride, false) {}

	ImageMat(int h, int stride, int planeIdx) :
		CoreMat<T>(h, stride) {}

	ImageMat() :
		ImageMat(0, 0, 0) {}

	ImageMat(CoreMat<T>& mat) :
		CoreMat<T>(mat.data(), mat.rows(), mat.cols(), false) {}

	using CoreMat<T>::addr;
};


template <class T> class ImagePlanar : public ImageBase<T> {

protected:

	std::vector<ImageMat<T>> mats;

public:

	ImagePlanar(int h, int w, int stride, int numPlanes);

	ImagePlanar(CoreMat<T>& y, CoreMat<T>& u, CoreMat<T>& v);

	ImagePlanar(CoreMat<T>& mat);

	ImagePlanar(CoreMat<T>* mat);

	ImagePlanar() : 
		ImagePlanar(0, 0, 0, 0) {}

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

	bool saveAsBMP(const std::string& filename, T scale = 1) const;
};


class ImageGray : public ImagePlanar<unsigned char> {

public:

	ImageGray(int h, int w) :
		ImagePlanar(h, w, w, 1) {}

	ImageGray(CoreMat<unsigned char>& mat) :
		ImagePlanar(mat) {}

	ImageGray() :
		ImageGray(0, 0) {}
};


template <class T> class ImagePacked : public ImageBase<T> {

public:

	ImagePacked(int h, int w, int stride, int planes, int arraysize) :
		ImageBase<T>(h, w, stride, planes, arraysize) {}

	ImagePacked(int h, int w, int stride, int planes) :
		ImagePacked(h, w, stride, planes, h * stride * planes) {}

	virtual T* data();

	virtual const T* data() const;

	T* addr(size_t idx, size_t r, size_t c) override;

	const T* addr(size_t idx, size_t r, size_t c) const override;
};
