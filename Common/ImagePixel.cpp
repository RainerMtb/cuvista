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

#include "ImageColor.hpp"

using namespace im;

void ImagePixelRgb<uchar>::writeTo(ImageColorBase<uchar>* dest, size_t r, size_t c) const {
	if (dest->getColorBase() == ColorBase::RGB) {
		dest->atColor(0, r, c) = x;
		dest->atColor(1, r, c) = y;
		dest->atColor(2, r, c) = z;

	} else if (dest->getColorBase() == ColorBase::YUV) {
		rgb_to_yuv(x, y, z, dest->addrColor(0, r, c), dest->addrColor(1, r, c), dest->addrColor(2, r, c));
	}
}

void ImagePixelRgb<uchar>::writeTo(ImageColorBase<float>* dest, size_t r, size_t c) const {
	if (dest->getColorBase() == ColorBase::RGB) {
		float f = 1.0f / 255.0f;
		dest->atColor(0, r, c) = x * f;
		dest->atColor(1, r, c) = y * f;
		dest->atColor(2, r, c) = z * f;

	} else if (dest->getColorBase() == ColorBase::YUV) {
		rgb_to_yuv(x, y, z, dest->addrColor(0, r, c), dest->addrColor(1, r, c), dest->addrColor(2, r, c));
	}
}


void ImagePixelRgb<float>::writeTo(ImageColorBase<uchar>* dest, size_t r, size_t c) const {
	if (dest->getColorBase() == ColorBase::RGB) {
		dest->atColor(0, r, c) = (uchar) (x * 255.0f);
		dest->atColor(1, r, c) = (uchar) (y * 255.0f);
		dest->atColor(2, r, c) = (uchar) (z * 255.0f);

	} else if (dest->getColorBase() == ColorBase::YUV) {
		rgb_to_yuv(x, y, z, dest->addrColor(0, r, c), dest->addrColor(1, r, c), dest->addrColor(2, r, c));
	}
}

void ImagePixelRgb<float>::writeTo(ImageColorBase<float>* dest, size_t r, size_t c) const {
	assert(false && "not implemented");
}

//------------------------------------------------------------------------------------------------------

void ImagePixelYuv<uchar>::writeTo(ImageColorBase<uchar>* dest, size_t r, size_t c) const {
	if (dest->getColorBase() == ColorBase::RGB) {
		yuv_to_rgb(x, y, z, dest->addrColor(0, r, c), dest->addrColor(1, r, c), dest->addrColor(2, r, c));

	} else if (dest->getColorBase() == ColorBase::YUV) {
		dest->atColor(0, r, c) = x;
		dest->atColor(1, r, c) = y;
		dest->atColor(2, r, c) = z;
	}
}

void ImagePixelYuv<uchar>::writeTo(ImageColorBase<float>* dest, size_t r, size_t c) const {
	if (dest->getColorBase() == ColorBase::RGB) {
		yuv_to_rgb(x, y, z, dest->addrColor(0, r, c), dest->addrColor(1, r, c), dest->addrColor(2, r, c));

	} else if (dest->getColorBase() == ColorBase::YUV) {
		float f = 1.0f / 255.0f;
		dest->atColor(0, r, c) = x * f;
		dest->atColor(1, r, c) = y * f;
		dest->atColor(2, r, c) = z * f;
	}
}


void ImagePixelYuv<float>::writeTo(ImageColorBase<uchar>* dest, size_t r, size_t c) const {
	if (dest->getColorBase() == ColorBase::RGB) {
		yuv_to_rgb(x, y, z, dest->addrColor(0, r, c), dest->addrColor(1, r, c), dest->addrColor(2, r, c));

	} else if (dest->getColorBase() == ColorBase::YUV) {
		dest->atColor(0, r, c) = (uchar) (x * 255.0f);
		dest->atColor(1, r, c) = (uchar) (y * 255.0f);
		dest->atColor(2, r, c) = (uchar) (z * 255.0f);
	}
}

void ImagePixelYuv<float>::writeTo(ImageColorBase<float>* dest, size_t r, size_t c) const {
	assert(false && "not implemented");
}
