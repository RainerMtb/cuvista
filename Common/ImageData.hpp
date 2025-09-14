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

#include <vector>
#include <string>

enum class ImageType {
	BASE,
	YUV444,
	NV12,
	BGRA,
	RGBA,
	SHARED,
	UNKNOWN,
};

template <class T> class ImageData {
public:
	virtual T* addr(size_t idx, size_t r, size_t c) = 0;
	virtual const T* addr(size_t idx, size_t r, size_t c) const = 0;
	virtual int planes() const = 0;
	virtual int height() const = 0;
	virtual int width() const = 0;
	virtual int strideInBytes() const = 0;
	virtual int sizeInBytes() const = 0;
	virtual std::vector<T> rawBytes() const = 0;
	virtual ImageType type() const = 0;

	virtual void setIndex(int64_t frameIndex) = 0;
	virtual bool saveAsBMP(const std::string& filename, T scale = 1) const = 0;
};
