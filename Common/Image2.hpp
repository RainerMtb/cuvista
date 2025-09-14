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

#pragma once

#include "Image.hpp"

class ImageBaseRgb;
class ImageBGR;
class ImageRGBA;
class ImageBGRA;
class ImageNV12;


//image from CoreMat
class ImageYuvMatFloat : public im::ImageMatShared<float> {

public:
	ImageYuvMatFloat(int h, int w, int stride, float* y, float* u, float* v);

	void toBaseRgb(ImageBaseRgb& dest, ThreadPoolBase& pool = defaultPool) const;
};


//image from 8bit data
class ImageMatYuv8 : public im::ImageMatShared<uint8_t> {

public:
	ImageMatYuv8(int h, int w, int stride, uint8_t* y, uint8_t* u, uint8_t* v);
};


//planar YUV 8bit image
class ImageYuv : public im::ImageBase<unsigned char> {

protected:
	im::LocalColor<unsigned char> getLocalColor(const Color& color) const override;

public:
	ImageYuv(int h, int w, int stride);

	ImageYuv(int h, int w, size_t stride);

	ImageYuv(int h, int w);

	ImageYuv();

	ImageType type() const override;

	virtual unsigned char* data();

	virtual const unsigned char* data() const;

	ImageYuv copy() const;

	//convert to NV12 format
	void toNV12(ImageNV12& dest, ThreadPoolBase& pool = defaultPool) const;

	//convert to BGR format for bmp files
	ImageBaseRgb& toBaseRgb(ImageBaseRgb& dest, ThreadPoolBase& pool = defaultPool) const;

	ImageBGR toBGR() const;

	ImageRGBA toRGBA() const;

	ImageBGRA toBGRA() const;

	ImageNV12 toNV12(int strideNV12, ThreadPoolBase& pool = defaultPool) const;

	ImageNV12 toNV12() const;

	void readFromPGM(const std::string& filename);

	//convert to ImageBGR and save to file, for repeated use BGR image should be preallocated
	bool saveAsColorBMP(const std::string& filename) const;
};


//image as NV12 data
class ImageNV12 : public im::ImageBase<unsigned char> {

private:
	size_t addrOffset(size_t idx, size_t r, size_t c) const;

public:
	ImageNV12(int h, int w, int stride);

	ImageNV12();

	ImageType type() const override;

	unsigned char* addr(size_t idx, size_t r, size_t c) override;

	const unsigned char* addr(size_t idx, size_t r, size_t c) const override;

	void toYuv(ImageYuv& dest, ThreadPoolBase& pool = defaultPool) const;

	ImageYuv toYuv() const;
};


//planer YUV in float
class ImageYuvFloat : public im::ImageBase<float> {

protected:
	im::LocalColor<float> getLocalColor(const Color& color) const override;

public:
	ImageYuvFloat(int h, int w, int stride);

	ImageYuvFloat();

	ImageYuv& toYuv(ImageYuv& dest, ThreadPoolBase& pool = defaultPool) const;

	ImageBaseRgb& toBaseRgb(ImageBaseRgb& dest, ThreadPoolBase& pool = defaultPool) const;

	void toNV12(ImageNV12& dest, ThreadPoolBase& pool = defaultPool) const;
};


class ImageBaseRgb : public im::ImagePacked<unsigned char> {

protected:
	ImageBaseRgb(int h, int w, int stride, int numPlanes, int arraysize);

	ImageBaseRgb(int h, int w, int stride, int numPlanes, unsigned char* data, int arraysize);

	im::LocalColor<unsigned char> getLocalColor(const Color& color) const override;

public:
	virtual std::vector<int> indexRgba() const = 0;

	void copyTo(ImageBaseRgb& dest, size_t r0 = 0, size_t c0 = 0, ThreadPoolBase& pool = defaultPool) const;

	bool saveAsColorBMP(const std::string& filename) const;

	bool saveAsPPM(const std::string& filename) const;
};


//packed rgba
class ImageRGBA : public ImageBaseRgb {

protected:
	std::vector<int> indexRgba() const {
		return { 0, 1, 2, 3 };
	}

public:
	ImageRGBA(int h, int w, int stride, unsigned char* data);

	ImageRGBA(int h, int w, int stride);

	ImageRGBA(int h, int w);

	ImageRGBA();

	ImageType type() const override;
};


//packed bgra
class ImageBGRA : public ImageBaseRgb {

protected:
	std::vector<int> indexRgba() const {
		return { 2, 1, 0, 3 };
	}

public:
	ImageBGRA(int h, int w, int stride, unsigned char* data);

	ImageBGRA(int h, int w, int stride);

	ImageBGRA(int h, int w);

	ImageBGRA();

	ImageType type() const override;
};


//packed 8bit in order BGR
class ImageBGR : public ImageBaseRgb {

protected:
	std::vector<int> indexRgba() const {
		return { 2, 1, 0 };
	}

public:
	ImageBGR(int h, int w, int stride, unsigned char* data);

	ImageBGR(int h, int w, int stride);

	ImageBGR(int h, int w);

	ImageBGR();

	ImageYuv toYUV() const;

	static ImageBGR readFromBMP(const std::string& filename);
};
