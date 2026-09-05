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
#include "ImagePixel.hpp"

using namespace im;

template <> void ImageColorBase<uchar>::convertValue(uchar value, uchar* dest) const {
	*dest = value;
}

template <> void ImageColorBase<float>::convertValue(float value, uchar* dest) const {
	*dest = (uchar) std::rint(value * 255.0f);
}

template <> void ImageColorBase<uchar>::convertValue(uchar value, float* dest) const {
	float f = 1.0f / 255.0f;
	*dest = value * f;
}

template <> void ImageColorBase<float>::convertValue(float value, float* dest) const {
	*dest = value;
}

//uchar to uchar
template <> void ImagePixel<uchar>::writeTo(ColorBase srcColor, ColorBase destColor, ImagePixel<uchar>& dest) const {
	if (srcColor == destColor) {
		*dest[0] = *pix[0];
		*dest[1] = *pix[1];
		*dest[2] = *pix[2];

	} else if (srcColor == ColorBase::RGB && destColor == ColorBase::YUV) {
		rgb_to_yuv(*pix[0], *pix[1], *pix[2], dest[0], dest[1], dest[2]);

	} else if (srcColor == ColorBase::YUV && destColor == ColorBase::RGB) {
		yuv_to_rgb(*pix[0], *pix[1], *pix[2], dest[0], dest[1], dest[2]);
	}

	if (dest.size == 4) {
		*dest[3] = 255;
	}
}

//uchar to float
template <> void ImagePixel<uchar>::writeTo(ColorBase srcColor, ColorBase destColor, ImagePixel<float>& dest) const {
	if (srcColor == destColor) {
		float f = 1.0f / 255.0f;
		*dest[0] = *pix[0] * f;
		*dest[1] = *pix[1] * f;
		*dest[2] = *pix[2] * f;

	} else if (srcColor == ColorBase::RGB && destColor == ColorBase::YUV) {
		rgb_to_yuv(*pix[0], *pix[1], *pix[2], dest[0], dest[1], dest[2]);

	} else if (srcColor == ColorBase::YUV && destColor == ColorBase::RGB) {
		yuv_to_rgb(*pix[0], *pix[1], *pix[2], dest[0], dest[1], dest[2]);
	}

	if (dest.size == 4) {
		*dest[3] = 1.0f;
	}
}

//float to uchar
template <> void ImagePixel<float>::writeTo(ColorBase srcColor, ColorBase destColor, ImagePixel<uchar>& dest) const {
	if (srcColor == destColor) {
		*dest[0] = (uchar) std::rint(*pix[0] * 255.0f);
		*dest[1] = (uchar) std::rint(*pix[1] * 255.0f);
		*dest[2] = (uchar) std::rint(*pix[2] * 255.0f);

	} else if (srcColor == ColorBase::RGB && destColor == ColorBase::YUV) {
		rgb_to_yuv(*pix[0], *pix[1], *pix[2], dest[0], dest[1], dest[2]);

	} else if (srcColor == ColorBase::YUV && destColor == ColorBase::RGB) {
		yuv_to_rgb(*pix[0], *pix[1], *pix[2], dest[0], dest[1], dest[2]);
	}

	if (dest.size == 4) {
		*dest[3] = 255;
	}
}
