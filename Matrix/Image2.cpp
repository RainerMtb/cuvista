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


#include "Image2.hpp"


ImagePPM& ImageYuvMat::toPPM(ImagePPM& dest, ThreadPoolBase& pool) const {
	yuvToRgb(dest, { 0, 1, 2 }, pool);
	return dest;
}

ImagePPM ImageYuvMat::toPPM() const {
	ImagePPM out(h, w);
	return toPPM(out);
}


//------------------------
// YUV image stuff
//------------------------

unsigned char* ImageYuv::data() {
	return arrays.at(0).get();
}

const unsigned char* ImageYuv::data() const {
	return arrays.at(0).get();
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
		if (p5 != "P5") throw std::exception("file does not start with 'P5'");
		if (maxVal != 255) throw std::exception("max value must be 255");
		if (inw > stride) throw std::exception("invalid width");
		if (inh != h * 3) throw std::exception("invalid height");
		file.get(); //read delimiter

		for (int i = 0; i < 3; i++) {
			for (size_t row = 0; row < h; row++) {
				file.read(reinterpret_cast<char*>(arrays[i].get() + row * stride), w);
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

ImageRGB& ImageYuv::toRGB(ImageRGB& dest, ThreadPoolBase& pool) const {
	yuvToRgb(dest, { 0, 1, 2 }, pool);
	return dest;
}

ImageRGB ImageYuv::toRGB() const {
	ImageRGB out(h, w);
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

ImageARGB& ImageYuv::toARGB(ImageARGB& dest, ThreadPoolBase& pool) const {
	dest.setValues(0, 0xFF);
	yuvToRgb(dest, { 1, 2, 3 }, pool);
	return dest;
}

