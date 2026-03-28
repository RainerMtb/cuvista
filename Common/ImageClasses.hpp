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

#include "BaseData.hpp"
#include "ImageBase.hpp"

namespace im {

	template <class T> class ImageY : public ImageBase<T> {

	public:
		ImageY(int h, int w, int stride, T maxValue) {
			this->storePtr = std::make_shared<ImageStoreLocal<T>>(h * stride);
			this->typePtr = std::make_shared<ImageTypePlanar<T>>(this->storePtr, h, w, stride, 1);
			this->colorPtr = std::make_shared<ImageColorYuv<T>>(this->typePtr, std::array<int, 4>{ 0, 1, 2, 3 }, maxValue);
		}

		ImageY(int h, int w, int stride, T* data, T maxValue) {
			std::span<float> span(data, h * stride);
			this->storePtr = std::make_shared<ImageStoreSharedSingle<T>>(span);
			this->typePtr = std::make_shared<ImageTypePlanar<T>>(this->storePtr, h, w, stride, 1);
			this->colorPtr = std::make_shared<ImageColorYuv<T>>(this->typePtr, std::array<int, 4>{ 0, 1, 2, 3 }, maxValue);
		}

		ImageY() :
			ImageY<T>(0, 0, 0, 0)
		{}

		constexpr ImageType imageType() const override { return ImageType::Y; }
	};


	class ImageVuyxFloat : public ImageBase<float> {

	public:
		ImageVuyxFloat(int h, int w, int stride, float* data);
		ImageVuyxFloat(int h, int w, int stride);
		ImageVuyxFloat(int h, int w);
		ImageVuyxFloat();

		constexpr ImageType imageType() const override { return ImageType::VUYX; }
	};


	class Image8 : public ImageBase<uchar> {};

	class Image8yuv : public Image8 {};

	class ImageYuv : public Image8yuv {

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


	class ImageVuyx : public Image8yuv {

	public:
		ImageVuyx(int h, int w, int stride);
		ImageVuyx(int h, int w, size_t stride);
		ImageVuyx(int h, int w);
		ImageVuyx();

		static ImageVuyx readPgmFile(const std::string& filename);

		static ImageVuyx readBmpFile(const std::string& filename);

		constexpr ImageType imageType() const override { return ImageType::VUYX; }

		virtual uchar* addr(size_t idx, size_t r, size_t c) override;
		virtual const uchar* addr(size_t idx, size_t r, size_t c) const override;

		virtual uchar& at(size_t idx, size_t r, size_t c) override;
		virtual const uchar& at(size_t idx, size_t r, size_t c) const override;

		virtual uchar* row(size_t r) override;
		virtual const uchar* row(size_t r) const override;
	};


	class ImageNV12 : public Image8yuv {

	public:
		ImageNV12(int h, int w, int stride);
		ImageNV12(int h, int w);
		ImageNV12();

		constexpr ImageType imageType() const override { return ImageType::NV12; }
	};


	class Image8bgr : public Image8 {};

	class ImageBgr : public Image8bgr {

	public:
		ImageBgr(int h, int w);
		ImageBgr();

		constexpr ImageType imageType() const override { return ImageType::BGR; }

		static ImageBgr readBmpFile(const std::string& filename);

		virtual void saveBmpColor(const std::string& filename) const override;
	};


	class ImageBGRA : public Image8bgr {

	public:
		ImageBGRA(int h, int w, int stride, uchar* data);
		ImageBGRA(int h, int w, int stride);
		ImageBGRA(int h, int w);
		ImageBGRA();

		constexpr ImageType imageType() const override { return ImageType::BGRA; }

		virtual void saveBmpColor(const std::string& filename) const override;
	};


	class ImageRGBA : public Image8bgr {

	public:
		ImageRGBA(int h, int w, int stride, uchar* data);
		ImageRGBA(int h, int w, int stride);
		ImageRGBA(int h, int w);
		ImageRGBA();

		constexpr ImageType imageType() const override { return ImageType::RGBA; }
	};


	//implement template method here
	template <class T> void ImageBase<T>::saveBmpColor(const std::string& filename) const {
		ImageBgr bgr(h(), w());
		this->convertTo(bgr);
		bgr.saveBmpColor(filename);
	}
}
