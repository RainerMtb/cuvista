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


#include "OfxUtil.hpp"

namespace ofx {

	OfxImageFloat::OfxImageFloat(int h, int w, int stride, float* data) {
		std::span<float> span(data, h * stride);
		storePtr = std::make_shared<im::ImageStoreSharedSingle<float>>(span);
		typePtr = std::make_shared<im::ImageTypePacked<float>>(storePtr, h, w, stride, 4);
		colorPtr = std::make_shared<im::ImageColorRgb<float>>(typePtr, std::array<int, 4>{ 0, 1, 2, 3 }, 1.0f);
	}

	OfxImageFloat::OfxImageFloat(int h, int w, int stride) {
		storePtr = std::make_shared<im::ImageStoreLocal<float>>(h * stride);
		typePtr = std::make_shared<im::ImageTypePacked<float>>(storePtr, h, w, stride, 4);
		colorPtr = std::make_shared<im::ImageColorRgb<float>>(typePtr, std::array<int, 4>{ 0, 1, 2, 3 }, 1.0f);
	}

	OfxImageFloat::OfxImageFloat(int h, int w) :
		OfxImageFloat(h, w, util::alignValue(w * 4, 16))
	{}

	OfxImageFloat::OfxImageFloat() :
		OfxImageFloat(0, 0)
	{}

	void OfxImageFloat::saveBmpColor(const std::string& filename) const {
		std::ofstream os(filename, std::ios::binary);
		im::BmpColorHeader(w(), h()).writeHeader(os);
		std::vector<unsigned char> imageRow(util::alignValue(w() * 3, 4));

		for (int r = 0; r < h(); r++) {
			const float* src = data() + r * stride();
			unsigned char* dest = imageRow.data();
			for (int c = 0; c < w(); c++) {
				*dest++ = (unsigned char) std::clamp(src[2] * 255.0f, 0.0f, 255.0f);
				*dest++ = (unsigned char) std::clamp(src[1] * 255.0f, 0.0f, 255.0f);
				*dest++ = (unsigned char) std::clamp(src[0] * 255.0f, 0.0f, 255.0f);
				src += 4;
			}
			os.write(reinterpret_cast<char*>(imageRow.data()), imageRow.size());
		}
	}


	OfxImageByte::OfxImageByte(int h, int w, int stride, uint8_t* data) {
		std::span<uint8_t> span(data, h * stride);
		storePtr = std::make_shared<im::ImageStoreSharedSingle<uint8_t>>(span);
		typePtr = std::make_shared<im::ImageTypePacked<uint8_t>>(storePtr, h, w, stride, 4);
		colorPtr = std::make_shared<im::ImageColorRgb<uint8_t>>(typePtr, std::array<int, 4>{ 0, 1, 2, 3 }, 255);
	}

	OfxImageByte::OfxImageByte(int h, int w, int stride) {
		storePtr = std::make_shared<im::ImageStoreLocal<uint8_t>>(h * stride);
		typePtr = std::make_shared<im::ImageTypePacked<uint8_t>>(storePtr, h, w, stride, 4);
		colorPtr = std::make_shared<im::ImageColorRgb<uint8_t>>(typePtr, std::array<int, 4>{ 0, 1, 2, 3 }, 255);
	}

	OfxImageByte::OfxImageByte(int h, int w) :
		OfxImageByte(h, w, util::alignValue(w * 4, 16))
	{}

	OfxImageByte::OfxImageByte() :
		OfxImageByte(0, 0)
	{}

	void OfxImageByte::saveBmpColor(const std::string& filename) const {
		std::ofstream os(filename, std::ios::binary);
		im::BmpColorHeader(w(), h()).writeHeader(os);
		std::vector<unsigned char> imageRow(util::alignValue(w() * 3, 4));

		for (int r = 0; r < h(); r++) {
			const uint8_t* src = data() + r * stride();
			unsigned char* dest = imageRow.data();
			for (int c = 0; c < w(); c++) {
				*dest++ = (unsigned char) src[2];
				*dest++ = (unsigned char) src[1];
				*dest++ = (unsigned char) src[0];
				src += 4;
			}
			os.write(reinterpret_cast<char*>(imageRow.data()), imageRow.size());
		}
	}
}
