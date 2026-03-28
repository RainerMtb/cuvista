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

#include <cstdint>
#include <vector>
#include <array>
#include <iostream>
#include "ThreadPoolBase.h"

#undef min
#undef max

struct Size {
	int h, w;
};

namespace util {
	class CRC64;
}

namespace im {

	using uchar = unsigned char;
	using ushort = unsigned short;

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

	enum class ImageType {
		BGR,
		RGB,
		RGBA,
		BGRA,
		YUV,
		Y,
		VUYX,
		NV12,
		NONE,
	};

	enum class ColorBase {
		YUV,
		RGB,
		UNKNOWN,
	};

	template <class T> struct LocalColor {
		std::array<T, 4> colorData;
		double alpha;
	};

	//Interface to Image Classes
	template <class T> class IImage {

	public:
		int64_t index = -1;

		virtual constexpr ImageType imageType() const = 0;

		void setIndex(int64_t index) { this->index = index; }

		virtual T* addr(size_t idx, size_t r, size_t c) = 0;
		virtual const T* addr(size_t idx, size_t r, size_t c) const = 0;

		virtual T* row(size_t r) = 0;
		virtual const T* row(size_t r) const = 0;

		virtual T* plane(size_t idx) = 0;
		virtual const T* plane(size_t idx) const = 0;

		virtual T& at(size_t idx, size_t r, size_t c) = 0;
		virtual const T& at(size_t idx, size_t r, size_t c) const = 0;

		virtual int h() const = 0;
		virtual int rows() const = 0;
		virtual int w() const = 0;
		virtual int cols() const = 0;
		virtual int stride() const = 0;
		virtual int strideInBytes() const = 0;
		virtual int planes() const = 0;
		virtual size_t sizeInBytes() const = 0;
		virtual std::vector<T> bytes() const = 0;
		virtual void write(std::ostream& os) const = 0;

		virtual void saveBmpColor(const std::string& filename) const = 0;
		virtual void saveBmpPlanes(const std::string& filename) const = 0;
		virtual void savePgm(const std::string& filename) const = 0;

		virtual uint64_t crc() const = 0;
		virtual void crc(util::CRC64& base) const = 0;

		virtual ~IImage() = default;
	};
}