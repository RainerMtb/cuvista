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

#include <cmath>
#include "ImageType.hpp"

namespace im {

	template <class T> class ImageColorBase;

	template <class T> struct ImagePixel {
		T x = 0, y = 0, z = 0;
		virtual void readFrom(const ImageColorBase<T>* src, size_t r, size_t c);
		virtual void writeTo(ImageColorBase<uchar>* dest, size_t r, size_t c) const = 0;
		virtual void writeTo(ImageColorBase<float>* dest, size_t r, size_t c) const = 0;
	};

	//Image Color
	template <class T> class ImageColorBase {

	private:
		uchar convertValue(uchar value) const {
			return value;
		}

		uchar convertValue(float value) const {
			return (uchar) std::round(value * 255);
		}

	protected:
		std::shared_ptr<ImageTypeBase<T>> typePtr;
		T maxValue;

	public:
		std::array<int, 4> colorIndex;

		ImageColorBase(std::shared_ptr<ImageTypeBase<T>> typePtr, std::array<int, 4> colorIndex, T maxValue) :
			typePtr { typePtr },
			colorIndex { colorIndex },
			maxValue { maxValue }
		{}

		ImageColorBase() :
			ImageColorBase<T>({}, {}, 0)
		{}

		virtual LocalColor<T> getLocalColor(const Color& color) const = 0;

		virtual constexpr ColorBase getColorBase() const = 0;

		virtual std::shared_ptr<ImagePixel<T>> createPixel() const = 0;

		virtual void gray(ThreadPoolBase& pool = defaultPool) = 0;

		const T& atColor(size_t colorIdx, size_t r, size_t c) const {
			return typePtr->at(colorIndex[colorIdx], r, c);
		}

		T& atColor(size_t colorIdx, size_t r, size_t c) {
			return typePtr->at(colorIndex[colorIdx], r, c);
		}

		T* addrColor(size_t colorIdx, size_t r, size_t c) {
			return typePtr->addr(colorIndex[colorIdx], r, c);
		}

		template <class R> void convertTo(ImageColorBase<R>* dest, ThreadPoolBase& pool) const {
			auto func = [&] (size_t r) {
				auto pixel = createPixel();
				for (int c = 0; c < typePtr->w; c++) {
					pixel->readFrom(this, r, c);
					pixel->writeTo(dest, r, c);
				}
			};
			pool.addAndWait(func, 0, typePtr->h);
		}
		
		void saveBmpPlanes(const std::string& filename) const {
			std::ofstream os(filename, std::ios::binary);
			int h = typePtr->h;
			int w = typePtr->w;
			int planes = this->typePtr->planes;

			BmpGrayHeader(w, h * planes).writeHeader(os);
			int stridedWidth = util::alignValue(w, 4);
			std::vector<uchar> imageRow(stridedWidth);

			for (int i = planes - 1; i >= 0; i--) {
				int idx = colorIndex[i];
				for (int r = h - 1; r >= 0; r--) {
					//prepare one line of data
					for (int c = 0; c < w; c++) {
						imageRow[c] = convertValue(typePtr->at(idx, r, c));
					}
					//write strided line
					os.write(reinterpret_cast<char*>(imageRow.data()), stridedWidth);
				}
			}
		}

		void setColorPlane(int plane, T colorValue) {
			typePtr->setColorPlane(plane, colorValue);
		}

		void setColor(const Color& color) {
			typePtr->setColor(getLocalColor(color));
		}

		void setPixel(size_t row, size_t col, const Color& color) {
			LocalColor<T> local = getLocalColor(color);
			for (int i = 0; i < typePtr->planes; i++) {
				this->typePtr->at(i, row, col) = local.colorData[i];
			}
		}
	};

	template <class T> void ImagePixel<T>::readFrom(const ImageColorBase<T>* src, size_t r, size_t c) {
		x = src->atColor(0, r, c);
		y = src->atColor(1, r, c);
		z = src->atColor(2, r, c);
	}

	template <class T> struct ImagePixelRgb : public ImagePixel<T> {};

	template <> struct ImagePixelRgb<uchar> : public ImagePixel<uchar> {
		virtual void writeTo(ImageColorBase<uchar>* dest, size_t r, size_t c) const override;
		virtual void writeTo(ImageColorBase<float>* dest, size_t r, size_t c) const override;
	};

	template <> struct ImagePixelRgb<float> : public ImagePixel<float> {
		virtual void writeTo(ImageColorBase<uchar>* dest, size_t r, size_t c) const override;
		virtual void writeTo(ImageColorBase<float>* dest, size_t r, size_t c) const override;
	};

	template <class T> struct ImagePixelYuv : public ImagePixel<T> {};

	template <> struct ImagePixelYuv<uchar> : public ImagePixel<uchar> {
		virtual void writeTo(ImageColorBase<uchar>* dest, size_t r, size_t c) const override;
		virtual void writeTo(ImageColorBase<float>* dest, size_t r, size_t c) const override;
	};

	template <> struct ImagePixelYuv<float> : public ImagePixel<float> {
		virtual void writeTo(ImageColorBase<uchar>* dest, size_t r, size_t c) const override;
		virtual void writeTo(ImageColorBase<float>* dest, size_t r, size_t c) const override;
	};


	template <class T> class ImageColorRgb : public ImageColorBase<T> {

	public:
		ImageColorRgb(std::shared_ptr<ImageTypeBase<T>> type, std::array<int, 4> colorIndex, T maxValue) :
			ImageColorBase<T>(type, colorIndex, maxValue)
		{}

		ImageColorRgb() :
			ImageColorRgb<T>({}, {}, 0)
		{}

		virtual constexpr ColorBase getColorBase() const override {
			return ColorBase::RGB;
		}

		virtual LocalColor<T> getLocalColor(const Color& color) const override {
			LocalColor<T> local = {};
			for (size_t idx = 0; idx < this->colorIndex.size(); idx++) {
				local.colorData[idx] = color.getChannel(this->colorIndex[idx]);
			}
			local.alpha = color.getAlpha();
			return local;
		}

		virtual std::shared_ptr<ImagePixel<T>> createPixel() const override {
			return std::make_shared<ImagePixelRgb<T>>();
		}

		virtual void gray(ThreadPoolBase& pool = defaultPool) override {
			auto fcn = [&] (size_t r) {
				for (size_t c = 0; c < this->typePtr->w; c++) {
					unsigned char y = im::rgb_to_y(this->atColor(0, r, c), this->atColor(1, r, c), this->atColor(2, r, c));
					this->atColor(0, r, c) = this->atColor(1, r, c) = this->atColor(2, r, c) = y;
				}
			};
			pool.addAndWait(fcn, 0, this->typePtr->h);
		}
	};


	template <class T> class ImageColorYuv : public ImageColorBase<T> {

	public:
		ImageColorYuv(std::shared_ptr<ImageTypeBase<T>> type, std::array<int, 4> colorIndex, T maxValue) :
			ImageColorBase<T>(type, colorIndex, maxValue)
		{}

		ImageColorYuv() :
			ImageColorYuv<T>({}, {}, 0)
		{}

		virtual constexpr ColorBase getColorBase() const override {
			return ColorBase::YUV;
		}

		virtual LocalColor<T> getLocalColor(const Color& color) const override {
			LocalColor<T> out = {};
			rgb_to_yuv(color.getChannel(0), color.getChannel(1), color.getChannel(2), &out.colorData[0], &out.colorData[1], &out.colorData[2]);
			out.alpha = color.getAlpha();
			return out;
		}

		virtual std::shared_ptr<ImagePixel<T>> createPixel() const override {
			return std::make_shared<ImagePixelYuv<T>>();
		}

		virtual void gray(ThreadPoolBase& pool = defaultPool) override {
			this->setColorPlane(1, this->maxValue / 2);
			this->setColorPlane(2, this->maxValue / 2);
		}
	};

} //namespace
