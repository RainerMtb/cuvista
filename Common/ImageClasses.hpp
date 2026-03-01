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

#include "ImageBase.hpp"

namespace im {

	template <class T> class ImageMatShared : public ImageBase<T> {

	public:
		ImageMatShared(int h, int w, int stride, T* plane0, T* plane1, T* plane2, T maxValue) {
			int planeSize = stride * h;
			std::vector<std::span<T>> store = { 
				std::span<T>(plane0, planeSize), 
				std::span<T>(plane1, planeSize), 
				std::span<T>(plane2, planeSize) 
			};
			this->storePtr = std::make_shared<ImageStoreShared<T>>(store);
			this->typePtr = std::make_shared<ImageTypePlanar<T>>(this->storePtr, h, w, stride, 3);
			this->colorPtr = std::make_shared<ImageColorRgb<T>>(this->typePtr, std::array<int, 4>{ 0, 1, 2 }, maxValue);
		}

		ImageMatShared() :
			ImageMatShared<T>(0, 0, 0, nullptr, nullptr, nullptr, 0)
		{}

		constexpr ImageType imageType() const override { return ImageType::RGB; }
	};


	class ImageYuvFloat : public ImageBase<float> {

	public:
		ImageYuvFloat(int h, int w, int stride);
		ImageYuvFloat(int h, int w);
		ImageYuvFloat();

		constexpr ImageType imageType() const override { return ImageType::YUV; }
	};


	class Image8 : public ImageBase<uchar> {};

	class ImageYuv : public Image8 {

	public:
		ImageYuv(int h, int w, int stride);
		ImageYuv(int h, int w, size_t stride);
		ImageYuv(int h, int w);
		ImageYuv();

		static ImageYuv readPgmFile(const std::string& filename);

		static ImageYuv readBmpFile(const std::string& filename);

		constexpr ImageType imageType() const override { return ImageType::YUV; }

		double lumaRms() const;

		void adjustGamma(float value);
	};


	class ImageNV12 : public Image8 {

	public:
		ImageNV12(int h, int w, int stride);
		ImageNV12(int h, int w);
		ImageNV12();

		constexpr ImageType imageType() const override { return ImageType::NV12; }
	};


	class ImageBgr : public Image8 {

	public:
		ImageBgr(int h, int w);
		ImageBgr();

		constexpr ImageType imageType() const override { return ImageType::BGR; }

		static ImageBgr readBmpFile(const std::string& filename);

		virtual void saveBmpColor(const std::string& filename) const override;
	};


	class ImageBGRA : public Image8 {

	public:
		ImageBGRA(int h, int w, int stride, uchar* data);
		ImageBGRA(int h, int w, int stride);
		ImageBGRA(int h, int w);
		ImageBGRA();

		constexpr ImageType imageType() const override { return ImageType::BGRA; }
	};


	class ImageRGBA : public Image8 {

	public:
		ImageRGBA(int h, int w, int stride, uchar* data);
		ImageRGBA(int h, int w, int stride);
		ImageRGBA(int h, int w);
		ImageRGBA();

		constexpr ImageType imageType() const override { return ImageType::RGBA; }
	};

	template <class T> void ImageBase<T>::saveBmpColor(const std::string& filename) const {
		ImageBgr bgr(height(), width());
		this->convertTo(bgr);
		bgr.saveBmpColor(filename);
	}

	template <class T> void ImageBase<T>::saveBmpPlanes(const std::string& filename) const {
		colorPtr->saveBmpPlanes(filename);
	}
}
