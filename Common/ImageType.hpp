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

#include "ImageStore.hpp"
#include "Color.hpp"

namespace im {

	//container for pointers to one pixel
	template <class T> struct ImagePixel {
		T* s0 = nullptr;
		T* s1 = nullptr;
		T* s2 = nullptr;
		T* s3 = nullptr;
		int offset = 0;

		void advance() {
			s0 += offset;
			s1 += offset;
			s2 += offset;
			if (s3 != nullptr) s3 += offset;
		}

		void writeTo(ColorBase srcColor, ColorBase destColor, ImagePixel<uchar>& dest) const;
		void writeTo(ColorBase srcColor, ColorBase destColor, ImagePixel<float>& dest) const;
	};


	//Data Type
	template <class T> class ImageTypeBase {

	protected:
		std::shared_ptr<ImageStoreBase<T>> storePtr;

	public:
		int h, w, stride, planes;

		ImageTypeBase(std::shared_ptr<ImageStoreBase<T>> storePtr, int h, int w, int stride, int planes) :
			storePtr { storePtr },
			h { h },
			w { w },
			stride { stride },
			planes { planes }
		{}

		ImageTypeBase() :
			ImageTypeBase<T>({}, 0, 0, 0, 0)
		{}

		virtual int rows() const = 0;
		virtual int cols() const = 0;
		virtual int pixelOffset() const = 0;

		virtual T* row(size_t r) { return storePtr->row(r, h, stride); }
		virtual const T* row(size_t r) const { return storePtr->row(r, h, stride); }

		virtual T* plane(size_t idx) { return row(idx * h); }
		virtual const T* plane(size_t idx) const { return row(idx * h); }

		virtual T* addr(size_t idx, size_t r, size_t c) = 0;
		virtual const T* addr(size_t idx, size_t r, size_t c) const = 0;

		virtual T& at(size_t idx, size_t r, size_t c) { return *addr(idx, r, c); }
		virtual const T& at(size_t idx, size_t r, size_t c) const { return *addr(idx, r, c); }

		virtual void setColor(int idx, T colorValue) = 0;
		virtual void setColor(const LocalColor<T>& localColor) = 0;

		virtual void copyRow(size_t r, std::shared_ptr<ImageTypeBase<T>> dest) const = 0;

		ImagePixel<T> pixelAt(size_t r, size_t c, std::array<int, 4> colorIndex) {
			ImagePixel<T> pix;
			pix.s0 = addr(colorIndex[0], r, c);
			pix.s1 = addr(colorIndex[1], r, c);
			pix.s2 = addr(colorIndex[2], r, c);
			if (planes == 4) pix.s3 = addr(colorIndex[3], r, c);
			pix.offset = pixelOffset();
			return pix;
		}

		virtual void crc(util::CRC64& base) const {
			for (int r = 0; r < rows(); r++) {
				const T* src = row(r);
				for (int c = 0; c < cols(); c++) {
					base.addDirect(src[c]);
				}
			}
		}

		virtual util::CRC64 crc() const {
			util::CRC64 base;
			crc(base);
			return base;
		}
	};

	template <class T> class ImageTypePacked : public ImageTypeBase<T> {

	public:
		ImageTypePacked(std::shared_ptr<ImageStoreBase<T>> store, int h, int w, int stride, int planes) :
			ImageTypeBase<T>(store, h, w, stride, planes)
		{}

		ImageTypePacked() :
			ImageTypePacked<T>({}, 0, 0, 0, 0)
		{}

		virtual int rows() const override {
			return this->h;
		}

		virtual int cols() const override {
			return this->w * this->planes;
		}

		virtual int pixelOffset() const override {
			return this->planes;
		}

		virtual T* addr(size_t idx, size_t r, size_t c) override {
			assert(idx < this->planes && r < this->h && c < this->w && "invalid address");
			return this->row(r) + c * this->planes + idx;
		}

		virtual const T* addr(size_t idx, size_t r, size_t c) const override {
			assert(idx < this->planes && r < this->h && c < this->w && "invalid address");
			return this->row(r) + c * this->planes + idx;
		}

		virtual void setColor(int idx, T colorValue) override {
			for (int r = 0; r < this->h; r++) {
				T* dest = addr(idx, r, 0);
				for (int c = 0; c < this->w; c++) {
					*dest = colorValue;
					dest += this->planes;
				}
			}
		}

		virtual void setColor(const LocalColor<T>& localColor) override {
			for (int r = 0; r < this->h; r++) {
				for (int c = 0; c < this->w; c++) {
					T* dest = addr(0, r, c);
					for (int z = 0; z < this->planes; z++) {
						*dest = localColor.colorData[z];
						dest++;
					}
				}
			}
		}

		virtual void copyRow(size_t r, std::shared_ptr<ImageTypeBase<T>> dest) const override {
			std::copy_n(this->row(r), this->w * this->planes, dest->row(r));
		}
	};

	template <class T> class ImageTypePlanar : public ImageTypeBase<T> {

	public:
		ImageTypePlanar(std::shared_ptr<ImageStoreBase<T>> store, int h, int w, int stride, int planes) :
			ImageTypeBase<T>(store, h, w, stride, planes)
		{}

		ImageTypePlanar() :
			ImageTypePlanar<T>({}, 0, 0, 0, 0)
		{}

		virtual int rows() const override {
			return this->h * this->planes;
		}

		virtual int cols() const override {
			return this->w;
		}

		virtual int pixelOffset() const override {
			return 1;
		}

		virtual T* addr(size_t idx, size_t r, size_t c) override {
			assert(idx < this->planes && r < this->h && c < this->w && "invalid address");
			return this->row(idx * this->h + r) + c;
		}

		virtual const T* addr(size_t idx, size_t r, size_t c) const override {
			assert(idx < this->planes && r < this->h && c < this->w && "invalid address");
			return this->row(idx * this->h + r) + c;
		}

		virtual void setColor(int idx, T colorValue) override {
			std::fill_n(this->row(1ull * idx * this->h), this->stride * this->h, colorValue);
		}

		virtual void setColor(const LocalColor<T>& localColor) override {
			for (int i = 0; i < this->planes; i++) {
				setColor(i, localColor.colorData[i]);
			}
		}

		virtual void copyRow(size_t r, std::shared_ptr<ImageTypeBase<T>> dest) const override {
			std::copy_n(this->row(r), this->w, dest->row(r));
		}
	};

	template <class T> class ImageTypeNV12 : public ImageTypeBase<T> {

	public:
		ImageTypeNV12(std::shared_ptr<ImageStoreBase<T>> store, int h, int w, int stride, int planes) :
			ImageTypeBase<T>(store, h, w, stride, planes)
		{}

		ImageTypeNV12() :
			ImageTypeNV12<T>({}, 0, 0, 0, 0)
		{}

		virtual int rows() const override {
			return this->h * 3 / 2;
		}

		virtual int cols() const override {
			return this->w;
		}

		virtual int pixelOffset() const override {
			return 1;
		}

		virtual uchar* addr(size_t idx, size_t r, size_t c) override {
			assert(idx < this->planes && r < this->h && c < this->w && "invalid address");
			return this->row(0) + r * this->stride + c;
		}

		virtual const uchar* addr(size_t idx, size_t r, size_t c) const override {
			assert(idx < this->planes && r < this->h && c < this->w && "invalid address");
			return this->row(0) + r * this->stride + c;
		}

		virtual void setColor(int idx, T colorValue) override {
			assert(false && "unsupported operation");
		}

		virtual void setColor(const LocalColor<T>& localColor) override {
			assert(false && "unsupported operation");
		}

		virtual void copyRow(size_t r, std::shared_ptr<ImageTypeBase<T>> dest) const override {
			std::copy_n(this->row(r), this->w, dest->row(r));
		}
	};

} //namespace
