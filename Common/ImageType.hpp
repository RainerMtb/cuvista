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
		std::shared_ptr<ImageStore<T>> storePtr;

	public:
		ImageTypeBase(std::shared_ptr<ImageStore<T>> storePtr) :
			storePtr { storePtr }
		{}

		ImageTypeBase() :
			ImageTypeBase<T>(std::shared_ptr<ImageStore<T>>())
		{}

		int h() const { return storePtr->h; }
		int w() const { return storePtr->w; }
		int stride() const { return storePtr->stride; }
		int planes() const { return storePtr->planes; }

		virtual int rows() const = 0;
		virtual int cols() const = 0;
		virtual int pixelOffset() const = 0;

		virtual T* row(size_t r) { return storePtr->row(r); }
		virtual const T* row(size_t r) const { return storePtr->row(r); }

		virtual T* plane(size_t idx) { return row(idx * storePtr->h); }
		virtual const T* plane(size_t idx) const { return row(idx * storePtr->h); }

		virtual T* addr(size_t idx, size_t r, size_t c) = 0;
		virtual const T* addr(size_t idx, size_t r, size_t c) const = 0;

		virtual T& at(size_t idx, size_t r, size_t c) { return *addr(idx, r, c); }
		virtual const T& at(size_t idx, size_t r, size_t c) const { return *addr(idx, r, c); }

		virtual void setColor(int idx, T colorValue) = 0;
		virtual void setColor(const LocalColor<T>& localColor) = 0;

		virtual void copyRow(size_t r, std::shared_ptr<ImageTypeBase<T>> dest) const = 0;

		ImagePixel<T> pixelAt(size_t r, size_t c, std::vector<int> colorIndex) {
			ImagePixel<T> pix;
			if (colorIndex.size() > 0) pix.s0 = addr(colorIndex[0], r, c);
			if (colorIndex.size() > 1) pix.s1 = addr(colorIndex[1], r, c);
			if (colorIndex.size() > 2) pix.s2 = addr(colorIndex[2], r, c);
			if (colorIndex.size() > 3) pix.s3 = addr(colorIndex[3], r, c);
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
		ImageTypePacked(std::shared_ptr<ImageStore<T>> storePtr) :
			ImageTypeBase<T>(storePtr)
		{}

		virtual int rows() const override {
			return this->storePtr->h;
		}

		virtual int cols() const override {
			return this->storePtr->w * this->storePtr->planes;
		}

		virtual int pixelOffset() const override {
			return this->storePtr->planes;
		}

		virtual T* addr(size_t idx, size_t r, size_t c) override {
			assert(idx < this->storePtr->planes && r < this->storePtr->h && c < this->storePtr->w && "invalid address");
			return this->row(r) + c * this->storePtr->planes + idx;
		}

		virtual const T* addr(size_t idx, size_t r, size_t c) const override {
			assert(idx < this->storePtr->planes && r < this->storePtr->h && c < this->storePtr->w && "invalid address");
			return this->row(r) + c * this->storePtr->planes + idx;
		}

		virtual void setColor(int idx, T colorValue) override {
			for (int r = 0; r < this->storePtr->h; r++) {
				T* dest = addr(idx, r, 0);
				for (int c = 0; c < this->storePtr->w; c++) {
					*dest = colorValue;
					dest += this->storePtr->planes;
				}
			}
		}

		virtual void setColor(const LocalColor<T>& localColor) override {
			//fill first row
			for (int c = 0; c < this->storePtr->w; c++) {
				T* dest = addr(0, 0, c);
				for (int z = 0; z < this->storePtr->planes; z++) {
					*dest = localColor.colorData[z];
					dest++;
				}
			}

			//copy rows
			int siz = this->storePtr->w * this->storePtr->planes;
			for (int r = 1; r < this->storePtr->h; r++) {
				std::copy_n(this->row(0), siz, this->row(r));
			}
		}

		virtual void copyRow(size_t r, std::shared_ptr<ImageTypeBase<T>> dest) const override {
			std::copy_n(this->row(r), this->storePtr->w * this->storePtr->planes, dest->row(r));
		}
	};

	template <class T> class ImageTypePlanar : public ImageTypeBase<T> {

	public:
		ImageTypePlanar(std::shared_ptr<ImageStore<T>> storePtr) :
			ImageTypeBase<T>(storePtr)
		{}

		virtual int rows() const override {
			return this->storePtr->h * this->storePtr->planes;
		}

		virtual int cols() const override {
			return this->storePtr->w;
		}

		virtual int pixelOffset() const override {
			return 1;
		}

		virtual T* addr(size_t idx, size_t r, size_t c) override {
			assert(idx < this->storePtr->planes && r < this->storePtr->h && c < this->storePtr->w && "invalid address");
			return this->row(idx * this->storePtr->h + r) + c;
		}

		virtual const T* addr(size_t idx, size_t r, size_t c) const override {
			assert(idx < this->storePtr->planes && r < this->storePtr->h && c < this->storePtr->w && "invalid address");
			return this->row(idx * this->storePtr->h + r) + c;
		}

		virtual void setColor(int idx, T colorValue) override {
			std::fill_n(this->row(1ull * idx * this->storePtr->h), this->storePtr->stride * this->storePtr->h, colorValue);
		}

		virtual void setColor(const LocalColor<T>& localColor) override {
			for (int i = 0; i < this->storePtr->planes; i++) {
				setColor(i, localColor.colorData[i]);
			}
		}

		virtual void copyRow(size_t r, std::shared_ptr<ImageTypeBase<T>> dest) const override {
			std::copy_n(this->row(r), this->storePtr->w, dest->row(r));
		}
	};

	template <class T> class ImageTypeNV12 : public ImageTypeBase<T> {

	public:
		ImageTypeNV12(std::shared_ptr<ImageStore<T>> storePtr) :
			ImageTypeBase<T>(storePtr)
		{}

		virtual int rows() const override {
			return this->storePtr->h * 3 / 2;
		}

		virtual int cols() const override {
			return this->storePtr->w;
		}

		virtual int pixelOffset() const override {
			return 1;
		}

		virtual uchar* addr(size_t idx, size_t r, size_t c) override {
			assert(idx < this->storePtr->planes && r < this->storePtr->h && c < this->storePtr->w && "invalid address");
			return this->row(0) + r * this->storePtr->stride + c;
		}

		virtual const uchar* addr(size_t idx, size_t r, size_t c) const override {
			assert(idx < this->storePtr->planes && r < this->storePtr->h && c < this->storePtr->w && "invalid address");
			return this->row(0) + r * this->storePtr->stride + c;
		}

		virtual void setColor(int idx, T colorValue) override {
			assert(false && "unsupported operation");
		}

		virtual void setColor(const LocalColor<T>& localColor) override {
			assert(false && "unsupported operation");
		}

		virtual void copyRow(size_t r, std::shared_ptr<ImageTypeBase<T>> dest) const override {
			std::copy_n(this->row(r), this->storePtr->w, dest->row(r));
		}
	};

} //namespace
