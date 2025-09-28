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
#include <cmath>
#include <iostream>

#include "ImageHeaders.hpp"
#include "Image2.hpp"
#include "Util.hpp"
#include "Color.hpp"


ImageYuvMatFloat::ImageYuvMatFloat(int h, int w, int stride, float* y, float* u, float* v) :
	ImageMatShared(h, w, stride, y, u, v) {}

void ImageYuvMatFloat::toBaseRgb(ImageBaseRgb& dest, ThreadPoolBase& pool) const {
	yuvToRgb(dest, dest.indexRgba(), pool);
}

ImageMatYuv8::ImageMatYuv8(int h, int w, int stride, uint8_t* y, uint8_t* u, uint8_t* v) :
	ImageMatShared(h, w, stride, y, u, v) {}


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

ImageType ImageYuv::type() const {
	return ImageType::YUV444;
}

unsigned char* ImageYuv::data() {
	return arrays.at(0).get();
}

const unsigned char* ImageYuv::data() const {
	return arrays.at(0).get();
}

ImageYuv ImageYuv::copy() const {
	ImageYuv copyImage(h, w, stride);
	std::copy(arrays[0].get(), arrays[0].get() + arraySizes[0], copyImage.arrays[0].get());
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

void ImageYuv::toNV12(ImageNV12& dest, ThreadPoolBase& pool) const {
	//copy Y plane
	for (int r = 0; r < h; r++) {
		std::copy(addr(0, r, 0), addr(0, r, 0) + w, dest.addr(0, r, 0));
	}

	//interleave U and V plane, simple bilinear downsampling
	for (size_t z = 1; z < 3; z++) {
		const unsigned char* inptr = plane(z); //U and V plane of input data
		auto fcn = [&] (size_t r) {
			for (size_t c = 0; c < w / 2; c++) {
				size_t idx = r * 2 * stride + c * 2;
				int sum = int(inptr[idx]) + int(inptr[idx + 1]) + int(inptr[idx + stride]) + int(inptr[idx + stride + 1]);
				dest.at(z, r, c) = sum / 4;
			}
		};
		pool.addAndWait(fcn, 0, h / 2);
	}
}

ImageBaseRgb& ImageYuv::toBaseRgb(ImageBaseRgb& dest, ThreadPoolBase& pool) const {
	yuvToRgb(dest, dest.indexRgba(), pool);
	return dest;
}

ImageBGR ImageYuv::toBGR() const {
	ImageBGR out(h, w);
	toBaseRgb(out);
	return out;
}

ImageRGBA ImageYuv::toRGBA() const {
	ImageRGBA out(h, w);
	toBaseRgb(out);
	return out;
}

ImageBGRA ImageYuv::toBGRA() const {
	ImageBGRA out(h, w);
	toBaseRgb(out);
	return out;
}

im::LocalColor<unsigned char> ImageYuv::getLocalColor(const Color& color) const {
	im::LocalColor<unsigned char> out = {};
	im::rgb_to_yuv(color.getChannel(0), color.getChannel(1), color.getChannel(2), &out.colorData[0], &out.colorData[1], &out.colorData[2]);
	out.alpha = color.getAlpha();
	return out;
}


//------------------------
// NV12 class
//------------------------


ImageNV12::ImageNV12(int h, int w, int stride) :
	ImageBase<unsigned char>(h, w, stride, 3, { std::make_shared<unsigned char[]>(h * stride * 3 / 2) }, { h * stride * 3 / 2 })
{}

ImageNV12::ImageNV12() :
	ImageNV12(0, 0, 0)
{}

ImageNV12 ImageYuv::toNV12(int strideNV12, ThreadPoolBase& pool) const {
	ImageNV12 out(h, w, strideNV12);
	toNV12(out, pool);
	return out;
}

ImageNV12 ImageYuv::toNV12() const {
	return toNV12(w, defaultPool);
}

void ImageNV12::toYuv(ImageYuv& dest, ThreadPoolBase& pool) const {
	//copy Y plane over
	for (int r = 0; r < h; r++) {
		std::copy(addr(0, r, 0), addr(0, r, 0) + w, dest.addr(0, r, 0));
	}

	//upscale temporary uv into U and V planes here
	int h2 = h / 2;
	int w2 = w / 2;
	for (size_t z = 1; z < 3; z++) {
		for (int r = 0; r < h; r++) {
			for (int c = 0; c < w; c++) {
				double py = (0.5 + r) * h2 / h - 0.5;
				double px = (0.5 + c) * w2 / w - 0.5;
				dest.at(z, r, c) = sample(z, px, py, 0.0, w2 - 1.0, 0.0, h2 - 1.0);
			}
		}
	}
}

ImageYuv ImageNV12::toYuv() const {
	ImageYuv out(h, w, stride);
	toYuv(out);
	return out;
}

ImageType ImageNV12::type() const {
	return ImageType::NV12;
}

size_t ImageNV12::addrOffset(size_t idx, size_t r, size_t c) const {
	assert((idx == 0 && r < h && c < w) || (idx > 0 && idx < numPlanes && r < h / 2 && c < w / 2) && "invalid pixel address");
	return idx == 0 ? r * stride + c : (h + r) * stride + c * 2 + idx - 1;
}

unsigned char* ImageNV12::addr(size_t idx, size_t r, size_t c) {
	return arrays[0].get() + addrOffset(idx, r, c);
}

const unsigned char* ImageNV12::addr(size_t idx, size_t r, size_t c) const {
	return arrays[0].get() + addrOffset(idx, r, c);
}

unsigned char* ImageNV12::data() {
	return arrays[0].get();
}

const unsigned char* ImageNV12::data() const {
	return arrays[0].get();
}


//------------------------
// YUV float stuff
//------------------------


ImageYuvFloat::ImageYuvFloat(int h, int w, int stride) :
	ImageBase(h, w, stride, 3)
{}

ImageYuvFloat::ImageYuvFloat() :
	ImageYuvFloat(0, 0, 0)
{}

im::LocalColor<float> ImageYuvFloat::getLocalColor(const Color& color) const {
	float y, u, v;
	color.toYUVfloat(&y, &u, &v);
	return { { y, u, v, }, color.getAlpha() };
}

ImageYuv& ImageYuvFloat::toYuv(ImageYuv& dest, ThreadPoolBase& pool) const {
	for (size_t z = 0; z < 3; z++) {
		auto func = [&] (size_t r) {
			for (size_t c = 0; c < w; c++) {
				dest.at(z, r, c) = (unsigned char) std::rint(at(z, r, c) * 255.0f);
			}
		};
		pool.addAndWait(func, 0, h);
	}
	return dest;
}

ImageBaseRgb& ImageYuvFloat::toBaseRgb(ImageBaseRgb& dest, ThreadPoolBase& pool) const {
	yuvToRgb(dest, dest.indexRgba(), pool);
	return dest;
}

void ImageYuvFloat::toNV12(ImageNV12& dest, ThreadPoolBase& pool) const {
	//copy Y plane
	auto fcn = [&] (size_t r) {
		for (size_t c = 0; c < w; c++) {
			dest.at(0, r, c) = (unsigned char) std::rint(at(0, r, c) * 255.0f);
		}
	};
	pool.addAndWait(fcn, 0, h);

	//interleave U and V plane, simple bilinear downsampling
	for (size_t z = 1; z < 3; z++) {
		const float* inptr = addr(z, 0, 0); //U and V plane of input data
		auto fcn = [&] (size_t r) {
			for (size_t c = 0; c < w / 2; c++) {
				size_t idx = r * 2 * stride + c * 2;
				float sum = inptr[idx] + inptr[idx + 1] + inptr[idx + stride] + inptr[idx + stride + 1];
				dest.at(z, r, c) = (unsigned char) std::rint(sum / 4.0f * 255.0f);
			}
		};
		pool.addAndWait(fcn, 0, h / 2);
	}
}


//------------------------
// Base RGB implementation
//------------------------


ImageBaseRgb::ImageBaseRgb(int h, int w, int stride, int numPlanes, int arraysize) :
	ImagePacked<unsigned char>(h, w, stride, numPlanes, arraysize)
{}

ImageBaseRgb::ImageBaseRgb(int h, int w, int stride, int numPlanes, unsigned char* data, int arraysize) :
	ImagePacked<unsigned char>(h, w, stride, numPlanes, data, arraysize)
{}

im::LocalColor<unsigned char> ImageBaseRgb::getLocalColor(const Color& color) const {
	im::LocalColor<unsigned char> local = {};
	for (size_t idx = 0; idx < indexRgba().size(); idx++) {
		int channelIndex = indexRgba().at(idx);
		local.colorData[idx] = color.getChannel(channelIndex);
	}
	local.alpha = color.getAlpha();
	return local;
}

void ImageBaseRgb::copyTo(ImageBaseRgb& dest, size_t r0, size_t c0, ThreadPoolBase& pool) const {
	assert(c0 + w <= dest.w && r0 + h <= dest.h && "dimensions mismatch");
	for (int c = 0; c < w; c++) {
		for (int r = 0; r < h; r++) {
			int srcAlpha = numPlanes < 4 ? 255 : at(indexRgba().at(3), r, c);
			int destAlpha = 255 - srcAlpha;
			for (int idx = 0; idx < 3; idx++) {
				int srcidx = indexRgba().at(dest.indexRgba().at(idx));
				uint8_t& ptr = dest.at(idx, r0 + r, c0 + c);
				ptr = (at(srcidx, r, c) * srcAlpha + ptr * destAlpha) / 255;
			}
		}
	}
}

bool ImageBaseRgb::saveAsColorBMP(const std::string& filename) const {
	std::ofstream os(filename, std::ios::binary);
	im::BmpColorHeader(w, h).writeHeader(os);
	int stridedWidth = util::alignValue(w * 3, 4);
	std::vector<char> imageRow(stridedWidth);

	int ir = indexRgba().at(0);
	int ig = indexRgba().at(1);
	int ib = indexRgba().at(2);
	for (int r = h - 1; r >= 0; r--) {
		unsigned char* ptr = (unsigned char*) imageRow.data();
		for (int c = 0; c < w; c++) {
			*ptr++ = at(ib, r, c);
			*ptr++ = at(ig, r, c);
			*ptr++ = at(ir, r, c);
		}
		os.write(imageRow.data(), imageRow.size());
	}
	return os.good();
}

bool ImageBaseRgb::saveAsPPM(const std::string& filename) const {
	std::ofstream os(filename, std::ios::binary);
	im::PpmHeader(w, h).writeHeader(os);
	std::vector<unsigned char> row(w * 3);
	for (int r = 0; r < h; r++) {
		int pos = 0;
		for (int c = 0; c < w; c++) {
			for (int idx = 0; idx < 3; idx++) {
				int colorIdx = indexRgba().at(idx);
				row[pos++] = at(colorIdx, r, c);
			}
		}
		os.write(reinterpret_cast<char*>(row.data()), row.size());
	}
	return os.good();
}


//------------------------
// BGR image stuff
//------------------------


ImageBGR::ImageBGR(int h, int w, int stride, unsigned char* data) :
	ImageBaseRgb(h, w, stride, 3, data, h * stride)
{}

ImageBGR::ImageBGR(int h, int w, int stride) :
	ImageBaseRgb(h, w, stride, 3, h * stride)
{}

ImageBGR::ImageBGR(int h, int w) :
	ImageBGR(h, w, 3 * w) 
{}

ImageBGR::ImageBGR() :
	ImageBGR(0, 0) 
{}

ImageBGR ImageBGR::readFromBMP(const std::string& filename) {
	ImageBGR image;

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


//------------------------
// RGBA image stuff
//------------------------


ImageRGBA::ImageRGBA(int h, int w, int stride, unsigned char* data) :
	ImageBaseRgb(h, w, stride, 4, data, h * stride) 
{}

ImageRGBA::ImageRGBA(int h, int w, int stride) :
	ImageBaseRgb(h, w, stride, 4, h * stride)
{}

ImageRGBA::ImageRGBA(int h, int w) :
	ImageRGBA(h, w, 4 * w) 
{}

ImageRGBA::ImageRGBA() :
	ImageRGBA(0, 0) 
{}

ImageType ImageRGBA::type() const {
	return ImageType::RGBA;
}


//------------------------
// BGRA image stuff
//------------------------


ImageBGRA::ImageBGRA(int h, int w, int stride, unsigned char* data) :
	ImageBaseRgb(h, w, stride, 4, data, h * stride) 
{}

ImageBGRA::ImageBGRA(int h, int w, int stride) :
	ImageBaseRgb(h, w, stride, 4, h * stride) 
{}

ImageBGRA::ImageBGRA(int h, int w) :
	ImageBGRA(h, w, 4 * w) 
{}

ImageBGRA::ImageBGRA() :
	ImageBGRA(0, 0) 
{}

ImageType ImageBGRA::type() const {
	return ImageType::BGRA;
}
