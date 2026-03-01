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

		virtual T* row(size_t r) { return storePtr->row(r, h, stride); }
		virtual const T* row(size_t r) const { return storePtr->row(r, h, stride); }

		virtual T* plane(size_t idx) { return row(idx * h); }
		virtual const T* plane(size_t idx) const { return row(idx * h); }

		virtual T* addr(size_t idx, size_t r, size_t c) = 0;
		virtual const T* addr(size_t idx, size_t r, size_t c) const = 0;

		virtual T& at(size_t idx, size_t r, size_t c) { return *addr(idx, r, c); }
		virtual const T& at(size_t idx, size_t r, size_t c) const { return *addr(idx, r, c); }

		virtual void setColorPlane(int idx, T colorValue) = 0;
		virtual void setColor(const LocalColor<T>& localColor) = 0;

		virtual void copyRow(size_t r, std::shared_ptr<ImageTypeBase<T>> dest) const = 0;
		virtual void copy2D(size_t r0, size_t c0, size_t h0, size_t w0, std::shared_ptr<ImageTypeBase<T>> dest, size_t destRow, size_t destCol) const = 0;
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

		virtual T* addr(size_t idx, size_t r, size_t c) override {
			assert(idx < this->planes && r < this->h && c < this->w && "invalid address");
			return this->row(r) + c * this->planes + idx;
		}

		virtual const T* addr(size_t idx, size_t r, size_t c) const override {
			assert(idx < this->planes && r < this->h && c < this->w && "invalid address");
			return this->row(r) + c * this->planes + idx;
		}

		virtual void setColorPlane(int idx, T colorValue) override {
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
				T* dest = addr(0, r, 0);
				for (int c = 0; c < this->w; c++) {
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

		virtual void copy2D(size_t r0, size_t c0, size_t h0, size_t w0, std::shared_ptr<ImageTypeBase<T>> dest, size_t destRow, size_t destCol) const override {
			for (size_t r = 0; r < h0; r++) {
				std::copy_n(this->addr(0, r + r0, c0), w0 * this->planes, dest->addr(0, destRow + r, destCol));
			}
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

		virtual T* addr(size_t idx, size_t r, size_t c) override {
			assert(idx < this->planes && r < this->h && c < this->w && "invalid address");
			return this->row(idx * this->h + r) + c;
		}

		virtual const T* addr(size_t idx, size_t r, size_t c) const override {
			assert(idx < this->planes && r < this->h && c < this->w && "invalid address");
			return this->row(idx * this->h + r) + c;
		}

		virtual void setColorPlane(int idx, T colorValue) override {
			std::fill_n(this->row(1ull * idx * this->h), this->stride * this->h, colorValue);
		}

		virtual void setColor(const LocalColor<T>& localColor) override {
			for (int i = 0; i < this->planes; i++) {
				setColorPlane(i, localColor.colorData[i]);
			}
		}

		virtual void copyRow(size_t r, std::shared_ptr<ImageTypeBase<T>> dest) const override {
			std::copy_n(this->row(r), this->w, dest->row(r));
		}

		virtual void copy2D(size_t r0, size_t c0, size_t h0, size_t w0, std::shared_ptr<ImageTypeBase<T>> dest, size_t destRow, size_t destCol) const override {
			for (size_t z = 0; z < this->planes; z++) {
				for (size_t r = 0; r < h0; r++) {
					std::copy_n(this->addr(z, r + r0, c0), w0, dest->addr(z, destRow + r, destCol));
				}
			}
		}
	};

	template <class T> class ImageTypeNV12 : public ImageTypeBase<T> {

	private:
		size_t addrOffset(size_t idx, size_t r, size_t c) const {
			assert((idx == 0 && r < this->h && c < this->w) || (idx > 0 && idx < this->planes && r < this->h / 2 && c < this->w / 2) && "invalid pixel address");
			return idx == 0 ? r * this->stride + c : (this->h + r) * this->stride + c * 2 + idx - 1;
		}

	public:
		ImageTypeNV12(std::shared_ptr<ImageStoreBase<T>> store, int h, int w, int stride, int planes) :
			ImageTypeBase<T>(store, h, w, stride, planes)
		{}

		ImageTypeNV12() :
			ImageTypeNV12<T>({}, 0, 0, 0, 0)
		{}

		virtual int rows() const override {
			return this->h * 2;
		}

		virtual uchar* addr(size_t idx, size_t r, size_t c) override {
			return this->row(0) + addrOffset(idx, r, c);
		}

		virtual const uchar* addr(size_t idx, size_t r, size_t c) const override {
			return this->row(0) + addrOffset(idx, r, c);
		}

		virtual void setColorPlane(int idx, T colorValue) override {
			assert(false && "unsupported operation");
		}

		virtual void setColor(const LocalColor<T>& localColor) override {
			assert(false && "unsupported operation");
		}

		virtual void copyRow(size_t r, std::shared_ptr<ImageTypeBase<T>> dest) const override {
			assert(false && "unsupported operation");
		}

		virtual void copy2D(size_t r0, size_t c0, size_t h0, size_t w0, std::shared_ptr<ImageTypeBase<T>> dest, size_t destRow, size_t destCol) const override {
			assert(false && "unsupported operation");
		}
	};

} //namespace
