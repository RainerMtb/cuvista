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
#include "Image2.hpp"


ImagePPM& ImageYuvFloat::toPPM(ImagePPM& dest, ThreadPoolBase& pool) const {
	yuvToRgb(dest, 0, 1, 2, pool);
	return dest;
}

ImagePPM ImageYuvFloat::toPPM() const {
	ImagePPM out(h, w);
	return toPPM(out);
}


//------------------------
// YUV image stuff
//------------------------

unsigned char* ImageYuv::data() {
	return array.data();
}

const unsigned char* ImageYuv::data() const {
	return array.data();
}

bool ImageYuv::saveAsColorBMP(const std::string& filename) const {
	return toBGR().saveAsBMP(filename);
}

bool ImageYuv::saveAsPGM(const std::string& filename) const {
	std::ofstream os(filename, std::ios::binary);
	PgmHeader(stride, h).writeHeader(os);
	for (int i = 0; i < 3; i++) {
		os.write(reinterpret_cast<const char*>(mats[i].data()), mats[i].numel());
	}
	return os.good();
}

void ImageYuv::readFromPGM(const std::string& filename) {
	std::ifstream file(filename, std::ios::binary);
	try {
		std::string p5;
		int inw, inh, maxVal;
		file >> p5 >> inw >> inh >> maxVal;
		if (p5 != "P5") throw std::exception("file does not start with 'P5'");
		if (maxVal != 255) throw std::exception("max value must be 255");
		if (inw > stride) throw std::exception("invalid width");
		if (inh != h * 3) throw std::exception("invalid height");
		file.get(); //read delimiter

		for (int i = 0; i < 3; i++) {
			for (size_t row = 0; row < h; row++) {
				file.read(reinterpret_cast<char*>(mats[i].data() + row * stride), w);
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
	ImageYuv uv(h2, w2, w2, 2);
	for (int z = 0; z < 2; z++) {
		for (int r = 0; r < h2; r++) {
			for (int c = 0; c < w2; c++) {
				uv.at(z, r, c) = nv12[h * strideNV12 + r * strideNV12 + c * 2ull + z];
			}
		}
	}

	//upscale temporary uv into U and V planes here
	uv.scaleTo(0, *this, 1);
	uv.scaleTo(1, *this, 2);

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
		std::copy(addr(0, r, 0), addr(0, r, w), dest);
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

ImageRGB& ImageYuv::toRGB(ImageRGB& dest, ThreadPoolBase& pool) const {
	yuvToRgb(dest, 0, 1, 2, pool);
	return dest;
}

ImageRGB ImageYuv::toRGB() const {
	ImageRGB out(h, w);
	return toRGB(out);
}

ImageBGR& ImageYuv::toBGR(ImageBGR& dest, ThreadPoolBase& pool) const {
	yuvToRgb(dest, 2, 1, 0, pool);
	return dest;
}

ImageBGR ImageYuv::toBGR() const {
	ImageBGR out(h, w);
	return toBGR(out);
}

ImagePPM& ImageYuv::toPPM(ImagePPM& dest, ThreadPoolBase& pool) const {
	assert(dest.size() == 3ull * h * w && "dimensions mismatch");
	yuvToRgb(dest, 0, 1, 2, pool);
	return dest;
}

ImagePPM ImageYuv::toPPM() const {
	ImagePPM out(h, w);
	return toPPM(out);
}


//-----------------------------
// ImageBGR
//-----------------------------

bool ImageBGR::saveAsBMP(const std::string& filename) const {
	assert(stride * 3 % 4 == 0 && "one image row must be multiple of 4 bytes");
	std::ofstream os(filename, std::ios::binary);
	BmpColorHeader(stride, h).writeHeader(os);

	for (int r = h - 1; r >= 0; r--) {
		os.write(reinterpret_cast<const char*>(addr(0, r, 0)), stride * 3ull);
	}
	return os.good();
}


//-----------------------------------
// ImagePPM
//-----------------------------------

ImagePPM::ImagePPM(int h, int w) : ImagePacked(h, w, w, 3, h * w * 3 + headerSize) {
	//first 19 bytes are header for ppm format
	std::format_to_n(array.data(), headerSize, "P6 {:5} {:5} 255 ", w, h);
}

const unsigned char* ImagePPM::header() const {
	return array.data();
}

const unsigned char* ImagePPM::data() const {
	return array.data() + headerSize;
}

unsigned char* ImagePPM::data() {
	return array.data() + headerSize;
}

size_t ImagePPM::size() const {
	return array.size() - headerSize;
}

size_t ImagePPM::sizeTotal() const {
	return array.size();
}

bool ImagePPM::saveAsPGM(const std::string& filename) const {
	std::ofstream os(filename, std::ios::binary);
	os.write(reinterpret_cast<const char*>(header()), sizeTotal());
	return os.good();
}

bool ImagePPM::saveAsBMP(const std::string& filename) const {
	ImageBGR bgr(h, w);
	shufflePlanes(bgr, 2, 1, 0);
	return bgr.saveAsBMP(filename);
}