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

	ImageBgr::ImageBgr(int h, int w) {
		int stride = util::alignValue(w * 3, 4);
		storePtr = std::make_shared<ImageStoreLocal<uchar>>(h * stride);
		typePtr = std::make_shared<ImageTypePacked<uchar>>(storePtr, h, w, stride, 3);
		colorPtr = std::make_shared<ImageColorRgb<uchar>>(typePtr, std::array<int, 4>{ 2, 1, 0 }, 255);
	}

	ImageBgr::ImageBgr() :
		ImageBgr(0, 0)
	{}

	void ImageBgr::saveBmpColor(const std::string& filename) const {
		assert(strideInBytes() % 4 == 0 && "invalid stride");
		std::ofstream os(filename, std::ios::binary);
		BmpColorHeader(w(), h()).writeHeader(os);

		for (int r = h() - 1; r >= 0; r--) {
			os.write(reinterpret_cast<const char*>(addr(0, r, 0)), strideInBytes());
		}
		assert(os.good() && "error writing file");
	}

	ImageBgr ImageBgr::ImageBgr::readBmpFile(const std::string& filename) {
		ImageBgr image;

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
			image = ImageBgr(h, w);

			//when height is negative, rows are stored from top to bottom
			const char* src = data.data() + dataOffset;
			if (height < 0) {
				uchar* dest = image.typePtr->row(0);
				for (int r = 0; r < h; r++) {
					std::copy_n(src, 3ull * w, dest);
					src += stride;
					dest += image.stride();
				}

			} else {
				uchar* dest = image.typePtr->row(h - 1ull);
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


	//-----------------------------------------------------------------------

	ImageVuyxFloat::ImageVuyxFloat(int h, int w, int stride, float* data) {
		std::span<float> span(data, h * stride);
		storePtr = std::make_shared<ImageStoreSharedSingle<float>>(span);
		typePtr = std::make_shared<ImageTypePacked<float>>(storePtr, h, w, stride, 4);
		colorPtr = std::make_shared<ImageColorYuv<float>>(typePtr, std::array<int, 4>{ 2, 1, 0, 3 }, 1.0f);
	}

	ImageVuyxFloat::ImageVuyxFloat(int h, int w, int stride) {
		storePtr = std::make_shared<ImageStoreLocal<float>>(h * stride * 4);
		typePtr = std::make_shared<ImageTypePacked<float>>(storePtr, h, w, stride, 4);
		colorPtr = std::make_shared<ImageColorYuv<float>>(typePtr, std::array<int, 4>{ 2, 1, 0, 3 }, 1.0f);
	}

	ImageVuyxFloat::ImageVuyxFloat(int h, int w) :
		ImageVuyxFloat(h, w, w * 4) 
	{}

	ImageVuyxFloat::ImageVuyxFloat() :
		ImageVuyxFloat(0, 0)
	{}


	//-----------------------------------------------------------------------

	ImageYuv::ImageYuv(int h, int w, int stride) {
		storePtr = std::make_shared<ImageStoreLocal<uchar>>(h * stride * 3);
		typePtr = std::make_shared<ImageTypePlanar<uchar>>(storePtr, h, w, stride, 3);
		colorPtr = std::make_shared<ImageColorYuv<uchar>>(typePtr, std::array<int, 4>{ 0, 1, 2 }, 255);
	}

	ImageYuv::ImageYuv(int h, int w, size_t stride) :
		ImageYuv(h, w, (int) stride)
	{}

	ImageYuv::ImageYuv(int h, int w) :
		ImageYuv(h, w, util::alignValue(w, 64))
	{}

	ImageYuv::ImageYuv() :
		ImageYuv(0, 0)
	{}

	ImageYuv ImageYuv::readPgmFile(const std::string& filename) {
		ImageYuv yuv;
		std::ifstream file(filename, std::ios::binary);
		try {
			std::string p5;
			int w, h, maxVal;
			file >> p5 >> w >> h >> maxVal;
			if (p5 != "P5") throw std::runtime_error("file does not start with 'P5'");
			if (maxVal != 255) throw std::runtime_error("max value must be 255");
			file.get(); //read delimiter
			yuv = ImageYuv(h, w);
			file.read(reinterpret_cast<char*>(yuv.data()), 1ull * w * h);

		} catch (const std::exception& e) {
			std::cerr << "error reading from file: " << e.what() << std::endl;

		} catch (...) {
			std::cerr << "error reading from file" << std::endl;
		}
		return yuv;
	}

	ImageYuv ImageYuv::readBmpFile(const std::string& filename) {
		ImageBgr bgr = ImageBgr::readBmpFile(filename);
		ImageYuv yuv(bgr.h(), bgr.w());
		bgr.convertTo(yuv);
		return yuv;
	}

	double ImageYuv::lumaRms() const {
		int64_t sum = 0;
		for (int r = 0; r < h(); r++) {
			const unsigned char* ptr = typePtr->row(0);
			for (int c = 0; c < w(); c++) {
				int64_t val = ptr[c];
				sum += val * val;
			}
		}
		double s = w() * h();
		return std::sqrt(sum / s);
	}

	void ImageYuv::adjustGamma(float value) {
		float g = 1.0f / value;
		for (int r = 0; r < h(); r++) {
			unsigned char* ptr = typePtr->row(0);
			for (int c = 0; c < w(); c++) {
				unsigned char& p = ptr[c];
				float x = p / 255.0f;
				p = (unsigned char) std::rint(std::pow(x, g) * 255.0f);
			}
		}
	}


	//-----------------------------------------------------------------------

	ImageVuyx::ImageVuyx(int h, int w, int stride) {
		storePtr = std::make_shared<ImageStoreLocal<uchar>>(h * stride);
		typePtr = std::make_shared<ImageTypePacked<uchar>>(storePtr, h, w, stride, 4);
		colorPtr = std::make_shared<ImageColorYuv<uchar>>(typePtr, std::array<int, 4>{ 2, 1, 0, 3 }, 255);
	}

	ImageVuyx::ImageVuyx(int h, int w, size_t stride) :
		ImageVuyx(h, w, (int) stride)
	{}

	ImageVuyx::ImageVuyx(int h, int w) :
		ImageVuyx(h, w, util::alignValue(w * 4, 64))
	{}

	ImageVuyx::ImageVuyx() :
		ImageVuyx(0, 0)
	{}

	ImageVuyx ImageVuyx::readPgmFile(const std::string& filename) {
		ImageYuv yuv = ImageYuv::readPgmFile(filename);
		ImageVuyx vuyx(yuv.h(), yuv.w());
		yuv.convertTo(vuyx);
		return vuyx;
	}

	ImageVuyx ImageVuyx::readBmpFile(const std::string& filename) {
		ImageBgr bgr = ImageBgr::readBmpFile(filename);
		ImageVuyx vuyx(bgr.h(), bgr.w());
		bgr.convertTo(vuyx);
		return vuyx;
	}

	uchar* ImageVuyx::addr(size_t idx, size_t r, size_t c) { 
		return storePtr->data() + r * typePtr->stride + c * typePtr->planes + idx; 
	}

	const uchar* ImageVuyx::addr(size_t idx, size_t r, size_t c) const {
		return storePtr->data() + r * typePtr->stride + c * typePtr->planes + idx;
	}

	uchar& ImageVuyx::at(size_t idx, size_t r, size_t c) { 
		return *addr(idx, r, c); 
	}

	const uchar& ImageVuyx::at(size_t idx, size_t r, size_t c) const { 
		return *addr(idx, r, c); 
	}

	uchar* ImageVuyx::row(size_t r) {
		return storePtr->data() + r * typePtr->stride;
	}

	const uchar* ImageVuyx::row(size_t r) const {
		return storePtr->data() + r * typePtr->stride;
	}



	//-----------------------------------------------------------------------

	ImageNV12::ImageNV12(int h, int w, int stride) {
		storePtr = std::make_shared<ImageStoreLocal<uchar>>(h * stride * 3 / 2);
		typePtr = std::make_shared<ImageTypeNV12<uchar>>(storePtr, h, w, stride, 3);
		colorPtr = std::make_shared<ImageColorYuv<uchar>>(typePtr, std::array<int, 4>{ 0, 1, 2 }, 255);
	}

	ImageNV12::ImageNV12(int h, int w) :
		ImageNV12(h, w, w)
	{}

	ImageNV12::ImageNV12() :
		ImageNV12(0, 0)
	{}


	//-----------------------------------------------------------------------

	ImageBGRA::ImageBGRA(int h, int w, int stride, uchar* data) {
		std::span<uchar> span(data, h * stride);
		storePtr = std::make_shared<ImageStoreSharedSingle<uchar>>(span);
		typePtr = std::make_shared<ImageTypePacked<uchar>>(storePtr, h, w, stride, 4);
		colorPtr = std::make_shared<ImageColorRgb<uchar>>(typePtr, std::array<int, 4>{ 2, 1, 0, 3 }, 255);
	}

	ImageBGRA::ImageBGRA(int h, int w, int stride) {
		storePtr = std::make_shared<ImageStoreLocal<uchar>>(h * stride);
		typePtr = std::make_shared<ImageTypePacked<uchar>>(storePtr, h, w, stride, 4);
		colorPtr = std::make_shared<ImageColorRgb<uchar>>(typePtr, std::array<int, 4>{ 2, 1, 0, 3 }, 255);
	}

	ImageBGRA::ImageBGRA(int h, int w) :
		ImageBGRA(h, w, util::alignValue(w * 4, 64))
	{}

	ImageBGRA::ImageBGRA() :
		ImageBGRA(0, 0)
	{}

	void ImageBGRA::saveBmpColor(const std::string& filename) const {
		std::ofstream os(filename, std::ios::binary);
		BmpColorHeader(w(), h()).writeHeader(os);
		std::vector<uchar> line(util::alignValue(w() * 3, 4));

		for (int r = h() - 1; r >= 0; r--) {
			uchar* dest = line.data();
			const uchar* src = row(r);
			for (int c = 0; c < w(); c++) {
				*dest++ = *src++;
				*dest++ = *src++;
				*dest++ = *src++;
				src++;
			}
			os.write(reinterpret_cast<char*>(line.data()), line.size());
		}
		assert(os.good() && "error writing file");
	}


	//-----------------------------------------------------------------------

	ImageRGBA::ImageRGBA(int h, int w, int stride, uchar* data) {
		std::span<uchar> span(data, h * stride);
		storePtr = std::make_shared<ImageStoreSharedSingle<uchar>>(span);
		typePtr = std::make_shared<ImageTypePacked<uchar>>(storePtr, h, w, stride, 4);
		colorPtr = std::make_shared<ImageColorRgb<uchar>>(typePtr, std::array<int, 4>{ 0, 1, 2, 3 }, 255);
	}

	ImageRGBA::ImageRGBA(int h, int w, int stride) {
		storePtr = std::make_shared<ImageStoreLocal<uchar>>(h * stride);
		typePtr = std::make_shared<ImageTypePacked<uchar>>(storePtr, h, w, stride, 4);
		colorPtr = std::make_shared<ImageColorRgb<uchar>>(typePtr, std::array<int, 4>{ 0, 1, 2, 3 }, 255);
	}

	ImageRGBA::ImageRGBA(int h, int w) :
		ImageRGBA(h, w, util::alignValue(w * 4, 64))
	{}

	ImageRGBA::ImageRGBA() :
		ImageRGBA(0, 0)
	{}
}