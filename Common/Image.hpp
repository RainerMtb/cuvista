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

#include "ThreadPoolBase.h"
#include "ImageData.hpp"
#include "Color.hpp"

struct Size {
	int h, w;
};

namespace im {

	template <class T> struct LocalColor {
		std::array<T, 4> colorData;
		double alpha;
	};

	enum class TextAlign {
		TOP_LEFT,
		TOP_CENTER,
		TOP_RIGHT,
		MIDDLE_LEFT,
		MIDDLE_CENTER,
		MIDDLE_RIGHT,
		BOTTOM_LEFT,
		BOTTOM_CENTER,
		BOTTOM_RIGHT,
	};

	enum class MarkerType {
		DOT,
		BOX,
		DIAMOND,
	};

	template <class T> class ImageBase : public ImageData<T> {

	public:
		int64_t index = -1;
		int h, w, stride, numPlanes;

	protected:
		//size_t imageSize;
		std::vector<int> arraySizes;
		std::vector<std::shared_ptr<T[]>> arrays;

		int colorValue(T pixelValue) const;

		void plot4(double cx, double cy, double dx, double dy, double a, const LocalColor<T>& localColor);
		void plot(double x, double y, double a, const LocalColor<T>& localColor);
		void plot(int x, int y, double a, const LocalColor<T>& localColor);
		void plot(double x, double y, double a, const Color& color);
		void plot(int x, int y, double a, const Color& color);

		void yuvToRgb(ImageData<unsigned char>& dest, const std::vector<int>& planes, ThreadPoolBase& pool = defaultPool) const;

		void copyTo(ImageData<T>& dest, size_t r0, size_t c0, std::vector<int> srcPlanes, std::vector<int> destPlanes, 
			ThreadPoolBase& pool = defaultPool) const;

		virtual LocalColor<T> getLocalColor(const Color& color) const;

	public:
		ImageBase(int h, int w, int stride, int numPlanes, std::vector<std::shared_ptr<T[]>> arrays, std::vector<int> arraySizes);

		ImageBase(int h, int w, int stride, int numPlanes);

		ImageBase();

		ImageType type() const override;

		virtual ~ImageBase() = default;

		T* addr(size_t idx, size_t r, size_t c) override;

		const T* addr(size_t idx, size_t r, size_t c) const override;

		T* plane(size_t idx);

		const T* plane(size_t idx) const;

		virtual uint64_t crc() const;

		int planes() const override;

		int height() const override;

		int width() const override;

		int strideInBytes() const override;

		void setIndex(int64_t frameIndex) override;

		//access one pixel on plane idx and row / col
		virtual T& at(size_t idx, size_t r, size_t c);

		//read access one pixel on plane idx and row / col
		virtual const T& at(size_t idx, size_t r, size_t c) const;

		//set color value for all pixels in one plane
		virtual void setColorPlane(int plane, T colorValue);

		//set color values per color plane
		virtual void setColor(const Color& color);

		//set color values for one pixel
		void setPixel(size_t row, size_t col, std::vector<T> colors);

		//equals operator
		virtual bool operator == (const ImageBase& other) const;

		//compute median value of differences
		double compareTo(const ImageBase& other) const;

		//write text into image
		Size writeText(std::string_view text, int x, int y);

		//write text into image
		Size writeText(std::string_view text, int x, int y, int sx, int sy, TextAlign alignment);

		//write text into image
		Size writeText(std::string_view text, int x, int y, int sx, int sy, TextAlign alignment, const Color& fg, const Color& bg);

		void drawLine(double x0, double y0, double x1, double y1, const Color& color, double alpha = 1.0);

		void drawEllipse(double cx, double cy, double rx, double ry, const Color& color, bool fill = false);

		void drawCircle(double cx, double cy, double r, const Color& color, bool fill = false);

		void drawMarker(double cx, double cy, const Color& color, double rx, double ry, MarkerType type);

		void drawMarker(double cx, double cy, const Color& color, double radius = 1.5, MarkerType = MarkerType::DOT);

		void copyTo(ImageData<T>& dest, std::vector<int> srcPlanes, std::vector<int> destPlanes, ThreadPoolBase& pool = defaultPool) const;

		void copyTo(ImageData<T>& dest, size_t r0, size_t c0, ThreadPoolBase& pool = defaultPool) const;

		void copyTo(ImageData<T>& dest, ThreadPoolBase& pool = defaultPool) const;

		int sizeInBytes() const override;

		std::vector<T> rawBytes() const override;

		//sample clamped to area
		T sample(size_t plane, float x, float y, float x0, float x1, float y0, float y1) const;

		//sample clamped to image bounds
		T sample(size_t plane, float x, float y) const;

		//sample from image, return defaultValue when outside
		T sample(size_t plane, float x, float y, T defaultValue) const;

		bool saveAsBMP(const std::string& filename, T scale = 1) const;

		bool saveAsPGM(const std::string& filename, T scale = 1) const;

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

		std::vector<T> rawBytes() const override;
	};

	
	template <class T> class ImageMatShared : public ImageBase<T> {

	public:
		ImageMatShared(int h, int w, int stride, T* mat);

		ImageMatShared(int h, int w, int stride, T* y, T* u, T* v);

		ImageType type() const override;

		T* addr(size_t idx, size_t r, size_t c) override;

		const T* addr(size_t idx, size_t r, size_t c) const;
	};
}

using Image8 = im::ImageBase<unsigned char>;