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
#include "ThreadPoolBase.h"

namespace im {

	template <class T> class ImageBase {

	public:
		int64_t index = -1;
		int h, w, stride, numPlanes;

	protected:
		size_t imageSize;
		std::vector<std::shared_ptr<T[]>> arrays;

		static inline ThreadPoolBase defaultPool;

		int colorValue(T pixelValue) const;

		void plot(double x, double y, double a, ColorBase<T> color);

		void plot(int x, int y, double a, ColorBase<T> color);

		void plot4(double cx, double cy, double dx, double dy, double a, ColorBase<T> color);

		void yuvToRgb(ImageBase<unsigned char>& dest, std::vector<int> planes, ThreadPoolBase& pool = defaultPool) const;

		void copyTo(ImageBase<T>& dest, size_t r0, size_t c0, std::vector<int> srcPlanes, std::vector<int> destPlanes, ThreadPoolBase& pool = defaultPool) const;

	public:
		ImageBase(int h, int w, int stride, int numPlanes);

		ImageBase(int h, int w, int stride, int numPlanes, std::vector<std::shared_ptr<T[]>> arrays, size_t imageSize);

		ImageBase() :
			ImageBase(0, 0, 0, 0, {}, 0) {}

		virtual ~ImageBase() = default;

		virtual T* addr(size_t idx, size_t r, size_t c);

		virtual const T* addr(size_t idx, size_t r, size_t c) const;

		virtual size_t size() const;

		virtual size_t bytes() const;

		virtual uint64_t crc() const;

		T* plane(size_t idx);

		const T* plane(size_t idx) const;

		//access one pixel on plane idx and row / col
		T& at(size_t idx, size_t r, size_t c);

		//read access one pixel on plane idx and row / col
		const T& at(size_t idx, size_t r, size_t c) const;

		//set color value for all pixels in one plane
		void setValues(int plane, T colorValue);

		//set color values per color plane
		void setValues(const ColorBase<T>& color);

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

		void copyTo(ImageBase<T>& dest, std::vector<int> srcPlanes, std::vector<int> destPlanes, ThreadPoolBase& pool = defaultPool) const;

		void copyTo(ImageBase<T>& dest, ThreadPoolBase& pool = defaultPool) const;

		T sample(size_t plane, double x, double y) const;

		void scaleByTwo(size_t srcPlane, ImageBase<T>& dest, size_t destPlane) const;

		void scaleByTwo(ImageBase<T>& dest) const;

		bool saveAsBMP(const std::string& filename, T scale = 1) const;

		bool saveAsPPM(const std::string& filename, T scale = 1) const;

	private:
		double fpart(double d);

		double rfpart(double d);
	};


	template <class T> class ImagePacked : public ImageBase<T> {

	public:
		ImagePacked(int h, int w, int stride, int numPlanes, int arraysize);

		ImagePacked(int h, int w, int stride, int numPlanes, T* data, int arraysize);

		T* addr(size_t idx, size_t r, size_t c) override;

		const T* addr(size_t idx, size_t r, size_t c) const override;

		virtual T* data();

		virtual const T* data() const;

		uint64_t crc() const override;

		void copyTo(ImageBase<T>& dest) const;

		void copyTo(ImageBase<T>& dest, std::vector<int> srcPlanes, std::vector<int> destPlanes) const;
	};


	template <class T> class ImageMatShared : public ImageBase<T> {

	public:
		ImageMatShared(int h, int w, int stride, T* mat) :
			ImageBase<T>(h, w, stride, 1, { { mat, NullDeleter<T>() } }, h * stride) {}

		ImageMatShared(int h, int w, int stride, T* y, T* u, T* v) :
			ImageBase<T>(h, w, stride, 3, { { y, NullDeleter<T>() }, { u, NullDeleter<T>() }, { v, NullDeleter<T>() } }, 3 * h * stride) {}

		T* addr(size_t idx, size_t r, size_t c) override;

		const T* addr(size_t idx, size_t r, size_t c) const;
	};
}