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

#include <iostream>
#include "ImageClasses.hpp"

namespace im {

	ImageBgrV2::ImageBgrV2(int h, int w, int stride) 
	{
		this->store = std::make_shared<ImageStoreLocal<uchar>>(h * stride, 3, 0);
		this->type = std::make_shared<ImageTypePacked<uchar>>(this->store, h, w, stride);
		this->color = std::make_shared<ImageColorRgb<uchar>>(this->type, std::vector<int>({ 2, 1, 0 }), 255);
	}

	ImageBgrV2::ImageBgrV2(int h, int w) :
		ImageBgrV2(h, w, util::alignValue(w * 3, 4))
	{}

	ImageBgrV2::ImageBgrV2() :
		ImageBgrV2(0, 0)
	{}

	void ImageBgrV2::saveBmpColor(const std::string& filename) const {
		std::ofstream os(filename, std::ios::binary);
		BmpColorHeader(width(), height()).writeHeader(os);

		for (int r = height() - 1; r >= 0; r--) {
			os.write(reinterpret_cast<const char*>(addr(0, r, 0)), strideInBytes());
		}
	}

	ImageBgrV2 ImageBgrV2::ImageBgrV2::readBmpFile(const std::string& filename) {
		ImageBgrV2 image;

		auto readBytes = [] (const char* ptr, int byteCount) {
			int out = 0;
			for (int i = 0; i < byteCount; i++, ptr++) {
				out |= uint8_t(*ptr) << i * 8;
			}
			return out;
		};

		try {
			//read all bytes from file
			std::ifstream is(filename, std::ios::binary);
			std::vector<char> data((std::istreambuf_iterator<char>(is)), (std::istreambuf_iterator<char>()));
			is.close();
			int fileSize = (int) data.size();

			//analyse header
			if (fileSize < 54) throw std::runtime_error("invalid file");
			if (data[0] != 'B' || data[1] != 'M') throw std::runtime_error("not a bmp file");
			int siz = readBytes(&data[2], 4);
			if (siz != fileSize) throw std::runtime_error("invalid file size");

			int dataOffset = readBytes(&data[10], 4);
			int infoHeaderSize = readBytes(&data[14], 4);
			if (infoHeaderSize != 40) throw std::runtime_error("only BITMAPINFOHEADER is supported");

			int w = readBytes(&data[18], 4);
			int height = readBytes(&data[22], 4);

			int planes = readBytes(&data[26], 2);
			if (planes != 1) throw std::runtime_error("number of planes must be 1");
			int bits = readBytes(&data[28], 2);
			if (bits != 24) throw std::runtime_error("only 24 bit images are supported");

			int compression = readBytes(&data[30], 4);
			if (compression != 0) throw std::runtime_error("only uncompressed images are supported");

			//copy bytes to Image
			int h = std::abs(height);
			int stride = (siz - dataOffset) / h;
			image = ImageBgrV2(h, w);

			//when height is negative, rows are stored from top to bottom
			const char* src = data.data() + dataOffset;
			if (height < 0) {
				uchar* dest = image.addr(0, 0, 0);
				for (int r = 0; r < h; r++) {
					std::copy_n(src, 3ull * w, dest);
					src += stride;
					dest += image.stride();
				}

			} else {
				uchar* dest = image.addr(0, h - 1ull, 0);
				for (int r = 0; r < h; r++) {
					std::copy_n(src, 3ull * w, dest);
					src += stride;
					dest -= image.stride();
				}
			}

		} catch (const std::runtime_error& err) {
			std::cerr << err.what() << std::endl;
		}

		return image;
	}
}