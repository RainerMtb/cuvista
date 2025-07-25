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

#include <format>
#include <stdexcept>
#include <cstring>
#include <iostream>

#include "Image2.hpp"
#include "Util.hpp"
#include "Color.hpp"


ImageYuvMatFloat::ImageYuvMatFloat(int h, int w, int stride, float* y, float* u, float* v) :
	ImageMatShared(h, w, stride, y, u, v) {}

ImagePPM& ImageYuvMatFloat::toPPM(ImagePPM& dest, ThreadPoolBase& pool) const {
	yuvToRgb(dest, { 0, 1, 2 }, pool);
	return dest;
}

ImageRGBA& ImageYuvMatFloat::toRGBA(ImageRGBA& dest, ThreadPoolBase& pool) const {
	dest.setColorPlane(3, 0xFF);
	yuvToRgb(dest, { 0, 1, 2 }, pool);
	return dest;
}

ImageMatYuv8::ImageMatYuv8(int h, int w, int stride, uint8_t* y, uint8_t* u, uint8_t* v) :
	ImageMatShared(h, w, stride, y, u, v) {}


//------------------------
// RGB planar image stuff
//------------------------

ImageRGBplanar::ImageRGBplanar(int h, int w) :
	ImageBase(h, w, w, 3) {}

ImageRGBplanar::ImageRGBplanar() :
	ImageRGBplanar(0, 0) {}

std::vector<unsigned char> ImageRGBplanar::getColorData(const Color& color) const {
	return color.getRGB();
}


//------------------------
// YUV image stuff
//------------------------

ImageYuv::ImageYuv(int h, int w, int stride) :
	ImageBase<unsigned char>(h, w, stride, 3) {}

ImageYuv::ImageYuv(int h, int w, size_t stride) :
	ImageYuv(h, w, int(stride)) {}

ImageYuv::ImageYuv(int h, int w) :
	ImageYuv(h, w, w) {}

ImageYuv::ImageYuv() :
	ImageYuv(0, 0, 0) {}

unsigned char* ImageYuv::data() {
	return arrays.at(0).get();
}

const unsigned char* ImageYuv::data() const {
	return arrays.at(0).get();
}

ImageYuv ImageYuv::copy() const {
	ImageYuv copyImage(h, w, stride);
	std::copy(arrays[0].get(), arrays[0].get() + imageSize, copyImage.arrays[0].get());
	copyImage.index = index;
	return copyImage;
}

bool ImageYuv::saveAsColorBMP(const std::string& filename) const {
	return toBGR().saveAsColorBMP(filename);
}

void ImageYuv::readFromPGM(const std::string& filename) {
	std::ifstream file(filename, std::ios::binary);
	try {
		std::string p5;
		int inw, inh, maxVal;
		file >> p5 >> inw >> inh >> maxVal;
		if (p5 != "P5") throw std::runtime_error("file does not start with 'P5'");
		if (maxVal != 255) throw std::runtime_error("max value must be 255");
		if (inw > stride) throw std::runtime_error("invalid width");
		if (inh != h * 3) throw std::runtime_error("invalid height");
		file.get(); //read delimiter

		for (int i = 0; i < 3; i++) {
			for (size_t r = 0; r < h; r++) {
				file.read(reinterpret_cast<char*>(addr(i, r, 0)), w);
			}
		}

	} catch (const std::exception& e) {
		printf("error reading from file: %s\n", e.what());

	} catch (...) {
		printf("error reading from file\n");
	}
}

ImageYuv& ImageYuv::fromNV12(const std::vector<unsigned char>& nv12, size_t strideNV12) {
	//read U and V into temporary image
	int h2 = h / 2;
	int w2 = w / 2;
	ImageBase<unsigned char> uv(h2, w2, w2, 2);
	for (int z = 0; z < 2; z++) {
		for (int r = 0; r < h2; r++) {
			for (int c = 0; c < w2; c++) {
				uv.at(z, r, c) = nv12[h * strideNV12 + r * strideNV12 + c * 2ull + z];
			}
		}
	}

	//upscale temporary uv into U and V planes here
	uv.scaleByTwo(0, *this, 1);
	uv.scaleByTwo(1, *this, 2);

	//copy Y plane over
	for (int r = 0; r < h; r++) {
		for (int c = 0; c < w; c++) {
			at(0, r, c) = nv12[r * strideNV12 + c];
		}
	}
	return *this;
}

void ImageYuv::toNV12(std::vector<unsigned char>& nv12, size_t strideNV12, ThreadPoolBase& pool) const {
	//copy Y plane
	unsigned char* dest = nv12.data();
	for (int r = 0; r < h; r++) {
		std::copy(addr(0, r, 0), addr(0, r, 0) + w, dest);
		dest += strideNV12;
	}

	//interleave U and V plane, simple bilinear downsampling
	unsigned char* outptr = nv12.data() + h * strideNV12;
	for (size_t z = 0; z < 2; z++) {
		const unsigned char* inptr = plane(z + 1); //U and V plane of input data
		auto fcn = [&] (size_t r) {
			for (size_t c = 0; c < w / 2; c++) {
				size_t idx = r * 2 * stride + c * 2;
				int sum = (int) inptr[idx] + inptr[idx + 1] + inptr[idx + stride] + inptr[idx + stride + 1];
				outptr[r * strideNV12 + c * 2 + z] = sum / 4;
			}
		};
		pool.addAndWait(fcn, 0, h / 2);
	}
}

std::vector<unsigned char> ImageYuv::toNV12(size_t strideNV12) const {
	std::vector<unsigned char> data(strideNV12 * h * 3 / 2);
	toNV12(data, strideNV12);
	return data;
}

std::vector<unsigned char> ImageYuv::toNV12() const {
	return toNV12(w);
}

ImageRGBplanar& ImageYuv::toRGB(ImageRGBplanar& dest, ThreadPoolBase& pool) const {
	yuvToRgb(dest, { 0, 1, 2 }, pool);
	return dest;
}

ImageRGBplanar ImageYuv::toRGB() const {
	ImageRGBplanar out(h, w);
	return toRGB(out);
}

ImageBGR& ImageYuv::toBGR(ImageBGR& dest, ThreadPoolBase& pool) const {
	yuvToRgb(dest, { 2, 1, 0 }, pool);
	return dest;
}

ImageBGR ImageYuv::toBGR() const {
	ImageBGR out(h, w);
	return toBGR(out);
}

ImagePPM& ImageYuv::toPPM(ImagePPM& dest, ThreadPoolBase& pool) const {
	assert(h == dest.h && w == dest.w && "dimensions mismatch");
	yuvToRgb(dest, { 0, 1, 2 }, pool);
	return dest;
}

ImagePPM ImageYuv::toPPM() const {
	ImagePPM out(h, w);
	return toPPM(out);
}

ImageRGBA& ImageYuv::toRGBA(ImageRGBA& dest, ThreadPoolBase& pool) const {
	dest.setColorPlane(3, 0xFF);
	yuvToRgb(dest, { 0, 1, 2 }, pool);
	return dest;
}

ImageRGBA ImageYuv::toRGBA() const {
	ImageRGBA out(h, w);
	toRGBA(out);
	return out;
}

std::vector<unsigned char> ImageYuv::getColorData(const Color& color) const {
	return color.getYUV();
}


//------------------------
// BGR image stuff
//------------------------

ImageBGR::ImageBGR(int h, int w) :
	ImagePacked(h, w, 3 * w, 3, 3 * h * w) {}

ImageBGR::ImageBGR() :
	ImageBGR(0, 0) {}

bool ImageBGR::saveAsColorBMP(const std::string& filename) const {
	std::ofstream os(filename, std::ios::binary);
	im::BmpColorHeader(w, h).writeHeader(os);
	std::vector<char> data(util::alignValue(w * 3, 4));

	for (int r = h - 1; r >= 0; r--) {
		const unsigned char* ptr = addr(0, r, 0);
		std::copy(ptr, ptr + 3 * w, data.data());
		os.write(data.data(), data.size());
	}
	return os.good();
}

static int readBytes(const char* ptr, int byteCount) {
	int out = 0;
	for (int i = 0; i < byteCount; i++, ptr++) {
		out |= uint8_t(*ptr) << i * 8;
	}
	return out;
}

ImageBGR ImageBGR::readFromBMP(const std::string& filename) {
	ImageBGR image;
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
		image = ImageBGR(h, w);

		//when height is negative, rows are stored from top to bottom
		char* ptr = data.data() + dataOffset;
		if (height < 0) {
			for (int r = 0; r < h; r++) std::memcpy(image.addr(0, r, 0), ptr + r * stride, 3ull * w);

		} else {
			for (int r = 0; r < h; r++) std::memcpy(image.addr(0, h - 1ull - r, 0), ptr + r * stride, 3ull * w);
		}

	} catch (const std::runtime_error& err) {
		std::cout << err.what() << std::endl;
	}

	return image;
}

ImageYuv ImageBGR::toYUV() const {
	ImageYuv out(h, w);
	for (int r = 0; r < h; r++) {
		for (int c = 0; c < w; c++) {
			im::rgb_to_yuv(at(2, r, c), at(1, r, c), at(0, r, c), out.addr(0, r, c), out.addr(1, r, c), out.addr(2, r, c));
		}
	}
	return out;
}

std::vector<unsigned char> ImageBGR::getColorData(const Color& color) const {
	auto rgb = color.getRGB();
	return { rgb[2], rgb[1], rgb[0] };
}


//------------------------
// RGBA image stuff
//------------------------


ImageRGBA::ImageRGBA(int h, int w, int stride, unsigned char* data) :
	ImagePacked(h, w, stride, 4, data, h* stride) {}

ImageRGBA::ImageRGBA(int h, int w, int stride) :
	ImagePacked(h, w, stride, 4, h* stride) {}

ImageRGBA::ImageRGBA(int h, int w) :
	ImageRGBA(h, w, 4 * w) {}

ImageRGBA::ImageRGBA() :
	ImageRGBA(0, 0) {}

void ImageRGBA::copyTo(ImageRGBA& dest, size_t r0, size_t c0, ThreadPoolBase& pool) const {
	assert(c0 + w <= dest.w && r0 + h <= dest.h && "dimensions mismatch");
	for (int c = 0; c < w; c++) {
		for (int r = 0; r < h; r++) {
			if (at(3, r, c) > 0) {
				dest.at(0, r + r0, c + c0) = at(0, r, c);
				dest.at(1, r + r0, c + c0) = at(1, r, c);
				dest.at(2, r + r0, c + c0) = at(2, r, c);
			}
		}
	}
}

bool ImageRGBA::saveAsColorBMP(const std::string& filename) const {
	std::ofstream os(filename, std::ios::binary);
	im::BmpColorHeader(w, h).writeHeader(os);
	int stridedWidth = util::alignValue(w * 3, 4);
	std::vector<char> imageRow(stridedWidth);

	for (int r = h - 1; r >= 0; r--) {
		unsigned char* ptr = (unsigned char*) imageRow.data();
		for (int c = 0; c < w; c++) {
			*ptr++ = at(2, r, c);
			*ptr++ = at(1, r, c);
			*ptr++ = at(0, r, c);
		}
		os.write(imageRow.data(), imageRow.size());
	}
	return os.good();
}

std::vector<unsigned char> ImageRGBA::getColorData(const Color& color) const {
	return color.getRGB();
}


//------------------------
// PPM image stuff
//------------------------

ImagePPM::ImagePPM(int h, int w) :
	ImagePacked(h, w, 3 * w, 3, 3 * h * w + headerSize) 
{
	//first 19 bytes are header for ppm format
	std::string header = std::format("P6 {:5} {:5} 255 ", w, h);
	std::copy_n(header.begin(), headerSize, arrays.at(0).get());
	//std::format_to_n(arrays.at(0).get(), headerSize, "P6 {:5} {:5} 255 ", w, h);
}

ImagePPM::ImagePPM() :
	ImagePPM(0, 0) {}

const unsigned char* ImagePPM::data() const {
	return arrays.at(0).get() + headerSize;
}

unsigned char* ImagePPM::data() {
	return arrays.at(0).get() + headerSize;
}

size_t ImagePPM::stridedSize() const {
	return 3ull * h * w;
}

size_t ImagePPM::stridedByteSize() const {
	return imageSize;
}

const unsigned char* ImagePPM::header() const {
	return arrays.at(0).get();
}

bool ImagePPM::saveAsPPM(const std::string& filename) const {
	std::ofstream os(filename, std::ios::binary);
	os.write(reinterpret_cast<const char*>(arrays.at(0).get()), stridedSize());
	return os.good();
}

bool ImagePPM::saveAsColorBMP(const std::string& filename) const {
	ImageBGR bgr(h, w);
	copyTo(bgr, { 0, 1, 2 }, { 2, 1, 0 });
	return bgr.saveAsColorBMP(filename);
}