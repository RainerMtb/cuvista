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

//uchar to uchar
template <> void ImagePixel<uchar>::writeTo(ColorBase srcColor, ColorBase destColor, ImagePixel<uchar>& dest) const {
	if (srcColor == destColor) {
		*dest.x = *x;
		*dest.y = *y;
		*dest.z = *z;

	} else if (srcColor == ColorBase::RGB && destColor == ColorBase::YUV) {
		rgb_to_yuv(*x, *y, *z, dest.x, dest.y, dest.z);

	} else if (srcColor == ColorBase::YUV && destColor == ColorBase::RGB) {
		yuv_to_rgb(*x, *y, *z, dest.x, dest.y, dest.z);
	}

	if (dest.w != nullptr) *dest.w = 255;
}

//uchar to float
template <> void ImagePixel<uchar>::writeTo(ColorBase srcColor, ColorBase destColor, ImagePixel<float>& dest) const {
	if (srcColor == destColor) {
		float f = 1.0f / 255.0f;
		*dest.x = *x * f;
		*dest.y = *y * f;
		*dest.z = *z * f;

	} else if (srcColor == ColorBase::RGB && destColor == ColorBase::YUV) {
		rgb_to_yuv(*x, *y, *z, dest.x, dest.y, dest.z);

	} else if (srcColor == ColorBase::YUV && destColor == ColorBase::RGB) {
		yuv_to_rgb(*x, *y, *z, dest.x, dest.y, dest.z);
	}

	if (dest.w != nullptr) *dest.w = 1.0f;
}

//float to uchar
template <> void ImagePixel<float>::writeTo(ColorBase srcColor, ColorBase destColor, ImagePixel<uchar>& dest) const {
	if (srcColor == destColor) {
		*dest.x = (uchar) std::rint(*x * 255.0f);
		*dest.y = (uchar) std::rint(*y * 255.0f);
		*dest.z = (uchar) std::rint(*z * 255.0f);

	} else if (srcColor == ColorBase::RGB && destColor == ColorBase::YUV) {
		rgb_to_yuv(*x, *y, *z, dest.x, dest.y, dest.z);

	} else if (srcColor == ColorBase::YUV && destColor == ColorBase::RGB) {
		yuv_to_rgb(*x, *y, *z, dest.x, dest.y, dest.z);
	}

	if (dest.w != nullptr) *dest.w = 255;
}
