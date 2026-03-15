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

	//Image Color
	template <class T> class ImageColorBase {

	private:
		void convertValue(T value, uchar* dest) const;
		void convertValue(T value, float* dest) const;

	protected:
		std::shared_ptr<ImageTypeBase<T>> typePtr;

	public:
		T maxValue;
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

		virtual constexpr ColorBase colorBase() const = 0;

		virtual void gray(ThreadPoolBase& pool = defaultPool) = 0;

		ImagePixel<T> pixelAt(size_t r, size_t c) const {
			return typePtr->pixelAt(r, c, colorIndex);
		}

		template <class R> friend class ImageColorBase;

		template <class R> void convertValuesTo(std::shared_ptr<ImageColorBase<R>> dest, ThreadPoolBase& pool = defaultPool) const {
			for (int r = 0; r < typePtr->rows(); r++) {
				T* srcPtr = typePtr->row(r);
				R* destPtr = dest->typePtr->row(r);
				for (int c = 0; c < typePtr->cols(); c++) {
					convertValue(*srcPtr, destPtr);
					srcPtr++;
					destPtr++;
				}
			}
		}

		template <class R> void convertTo(std::shared_ptr<ImageColorBase<R>> dest, ThreadPoolBase& pool = defaultPool) const {
			auto fcn = [&] (size_t r) {
				ImagePixel<T> srcPixel = pixelAt(r, 0);
				ImagePixel<R> destPixel = dest->pixelAt(r, 0);
				for (size_t c = 0; c < typePtr->w; c++) {
					srcPixel.writeTo(colorBase(), dest->colorBase(), destPixel);
					srcPixel.advance();
					destPixel.advance();
				}
			};
			pool.addAndWait(fcn, 0, typePtr->h);
		}

		void convertToNV12(std::shared_ptr<ImageColorBase<uchar>> dest, ThreadPoolBase& pool = defaultPool) const {
			auto fcn = [&] (size_t r) {
				uchar y = 0, u = 0, v = 0;
				ImagePixel<uchar> p = { &y, &u, &v };
				ImagePixel<T> src;
				size_t rr = r * 2;
				uchar* destY = dest->typePtr->row(rr);
				uchar* destUV = dest->typePtr->row(dest->typePtr->h + r);
				for (size_t c = 0; c < dest->typePtr->w / 2; c++) {
					size_t cc = c * 2;
					int sumU = 0, sumV = 0;

					src = pixelAt(rr, cc);
					src.writeTo(colorBase(), ColorBase::YUV, p);
					destY[cc] = *p.x;
					sumU += *p.y; sumV += *p.z;

					src = pixelAt(rr, cc + 1);
					src.writeTo(colorBase(), ColorBase::YUV, p);
					destY[cc + 1] = *p.x;
					sumU += *p.y; sumV += *p.z;

					src = pixelAt(rr + 1, cc);
					src.writeTo(colorBase(), ColorBase::YUV, p);
					destY[cc + dest->typePtr->stride] = *p.x;
					sumU += *p.y; sumV += *p.z;

					src = pixelAt(rr + 1, cc + 1);
					src.writeTo(colorBase(), ColorBase::YUV, p);
					destY[cc + dest->typePtr->stride + 1] = *p.x;
					sumU += *p.y; sumV += *p.z;

					destUV[cc] = sumU / 4;
					destUV[cc + 1] = sumV / 4;
				}
			};
			pool.addAndWait(fcn, 0, dest->typePtr->h / 2);
		}

		template <class R> void convertFromNV12(std::shared_ptr<ImageColorBase<R>> dest, ThreadPoolBase& pool = defaultPool) const {
			for (size_t r = 0; r < typePtr->h / 2; r++) {
				size_t rr = r * 2;
				uchar* srcY = typePtr->row(rr);
				uchar* srcUV = typePtr->row(dest->typePtr->h + r);
				ImagePixel<uchar> srcPix;
				ImagePixel<R> destPix;
				for (size_t c = 0; c < typePtr->w / 2; c++) {
					size_t cc = c * 2;

					srcPix = { srcY + cc, srcUV + cc, srcUV + cc + 1 };
					destPix = dest->pixelAt(rr, cc);
					srcPix.writeTo(ColorBase::YUV, dest->colorBase(), destPix);

					srcPix = { srcY + cc + 1, srcUV + cc, srcUV + cc + 1 };
					destPix = dest->pixelAt(rr, cc + 1);
					srcPix.writeTo(ColorBase::YUV, dest->colorBase(), destPix);

					srcPix = { srcY + typePtr->stride + cc, srcUV + cc, srcUV + cc + 1 };
					destPix = dest->pixelAt(rr + 1, cc);
					srcPix.writeTo(ColorBase::YUV, dest->colorBase(), destPix);

					srcPix = { srcY + typePtr->stride + cc + 1, srcUV + cc, srcUV + cc + 1 };
					destPix = dest->pixelAt(rr + 1, cc + 1);
					srcPix.writeTo(ColorBase::YUV, dest->colorBase(), destPix);
				}
			}
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
				//int idx = colorIndex[i];
				for (int r = h - 1; r >= 0; r--) {
					//prepare one line of data
					for (int c = 0; c < w; c++) {
						convertValue(typePtr->at(i, r, c), imageRow.data() + c);
					}
					//write strided line
					os.write(reinterpret_cast<char*>(imageRow.data()), stridedWidth);
				}
			}
			assert(os.good() && "error writing file");
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


	template <class T> class ImageColorRgb : public ImageColorBase<T> {

	public:
		ImageColorRgb(std::shared_ptr<ImageTypeBase<T>> type, std::array<int, 4> colorIndex, T maxValue) :
			ImageColorBase<T>(type, colorIndex, maxValue)
		{}

		ImageColorRgb() :
			ImageColorRgb<T>({}, {}, 0)
		{}

		virtual constexpr ColorBase colorBase() const override {
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

		virtual void gray(ThreadPoolBase& pool = defaultPool) override {
			auto fcn = [&] (size_t r) {
				ImagePixel<T> pixel = this->pixelAt(r, 0);
				for (size_t c = 0; c < this->typePtr->w; c++) {
					T gray = im::rgb_to_y(*pixel.x, *pixel.y, *pixel.z);
					*pixel.x = *pixel.y = *pixel.z = gray;
					pixel.advance();
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

		virtual constexpr ColorBase colorBase() const override {
			return ColorBase::YUV;
		}

		virtual LocalColor<T> getLocalColor(const Color& color) const override {
			LocalColor<T> out = {};
			rgb_to_yuv(color.getChannel(0), color.getChannel(1), color.getChannel(2), &out.colorData[0], &out.colorData[1], &out.colorData[2]);
			out.alpha = color.getAlpha();
			return out;
		}

		virtual void gray(ThreadPoolBase& pool = defaultPool) override {
			this->setColorPlane(1, this->maxValue / 2);
			this->setColorPlane(2, this->maxValue / 2);
		}
	};

} //namespace
