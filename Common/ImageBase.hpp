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

#include "ImageInterface.hpp"
#include "ImageHeaders.hpp"
#include "Util.hpp"

namespace im {

	//Data Storage
	template <class T> class ImageStoreBase {

	public:
		int planes;

		virtual T* plane(size_t idx) = 0;
		virtual const T* plane(size_t idx) const = 0;
		virtual std::vector<T> bytes() const = 0;

		ImageStoreBase<T>(int planes) :
			planes { planes }
		{}

		ImageStoreBase<T>() :
			ImageStoreBase<T>(0)
		{}
	};

	template <class T> class ImageStoreLocal : public ImageStoreBase<T> {

	public:
		std::vector<T> store;
		int planeOffset;

		ImageStoreLocal<T>(int siz, int planes, int planeOffset) :
			ImageStoreBase<T>(planes),
			store(siz),
			planeOffset { planeOffset }
		{}

		virtual T* plane(size_t idx) override {
			assert(idx < this->planes && "invalid plane index");
			return store.data() + idx * planeOffset;
		}

		virtual const T* plane(size_t idx) const override {
			assert(idx < this->planes && "invalid plane index");
			return store.data() + idx * planeOffset;
		}

		virtual std::vector<T> bytes() const override {
			return store;
		}
	};

	//template <class T> class ImageStoreShared : public ImageStoreBase<T> {

	//private:
	//	std::vector<std::span<T>> store;
	//};


	//Data Type
	template <class T> class ImageTypeBase {

	public:
		std::shared_ptr<ImageStoreBase<T>> store;
		int h, w, stride;

		ImageTypeBase<T>(std::shared_ptr<ImageStoreBase<T>> store, int h, int w, int stride) :
			store { store },
			h { h },
			w { w },
			stride { stride }
		{}

		ImageTypeBase<T>() :
			ImageTypeBase<T>({}, 0, 0, 0, 0)
		{}

		virtual T* addr(size_t idx, size_t r, size_t c) = 0;
		virtual const T* addr(size_t idx, size_t r, size_t c) const = 0;
	};

	template <class T> class ImageTypePacked : public ImageTypeBase<T> {

	public:
		ImageTypePacked<T>(std::shared_ptr<ImageStoreBase<T>> store, int h, int w, int stride) :
			ImageTypeBase<T>(store, h, w, stride)
		{}

		ImageTypePacked<T>() :
			ImageTypePacked<T>({}, 0, 0, 0)
		{}

		virtual T* addr(size_t idx, size_t r, size_t c) override {
			assert(idx < this->store->planes && r < this->h && c < this->w && "invalid address");
			return this->store->plane(0) + r * this->stride + c * this->store->planes + idx;
		}

		virtual const T* addr(size_t idx, size_t r, size_t c) const override {
			assert(idx < this->store->planes && r < this->h && c < this->w && "invalid address");
			return this->store->plane(0) + r * this->stride + c * this->store->planes + idx;
		}
	};

	template <class T> class ImageTypePlanar : public ImageTypeBase<T> {

	public:
		ImageTypePlanar<T>(std::shared_ptr<ImageStoreBase<T>> store, int h, int w, int stride) :
			ImageTypeBase<T>(store, h, w, stride)
		{}

		ImageTypePlanar<T>() :
			ImageTypePlanar<T>({}, 0, 0, 0)
		{}

		virtual T* addr(size_t idx, size_t r, size_t c) override {
			assert(idx < this->store->planes && r < this->h && c < this->w && "invalid address");
			return this->store->plane(idx) + r * this->stride + c;
		}

		virtual const T* addr(size_t idx, size_t r, size_t c) const override {
			assert(idx < this->store->planes && r < this->h && c < this->w && "invalid address");
			return this->store->plane(idx) + r * this->stride + c;
		}
	};


	//Image Color
	template <class T> class ImageColorBase {

	protected:
		const T* addr(size_t idx, size_t r, size_t c) const { return this->type->addr(idx, r, c); }

	public:
		std::shared_ptr<ImageTypeBase<T>> type;

		std::vector<int> colorIndex;
		T maxValue;

		ImageColorBase<T>(std::shared_ptr<ImageTypeBase<T>> type, std::vector<int> colorIndex, T maxValue) :
			type { type },
			colorIndex { colorIndex },
			maxValue { maxValue }
		{}

		ImageColorBase<T>() :
			ImageColorBase<T>({}, {}, 0)
		{}

		virtual void saveBmpColor(const std::string& filename) const = 0;
		virtual void saveBmpPlanes(const std::string& filename) const = 0;
	};


	template <class T> class ImageColorRgb : public ImageColorBase<T> {

		using ImageColorBase<T>::addr;

	public:
		ImageColorRgb(std::shared_ptr<ImageTypeBase<T>> type, std::vector<int> colorIndex, T maxValue) :
			ImageColorBase<T>(type, colorIndex, maxValue)
		{}

		ImageColorRgb() :
			ImageColorRgb<T>({},  {}, 0)
		{}

		virtual void saveBmpColor(const std::string& filename) const override {
			std::ofstream os(filename, std::ios::binary);
			int h = this->type->h;
			int w = this->type->w;
			BmpColorHeader(w, h).writeHeader(os);

			int stridedWidth = util::alignValue(w * 3, 4);
			std::vector<uchar> imageRow(stridedWidth);

			int ir = this->colorIndex[0];
			int ig = this->colorIndex[1];
			int ib = this->colorIndex[2];
			for (int r = h - 1; r >= 0; r--) {
				//prepare one line of data
				uchar* ptr = imageRow.data();
				for (int c = 0; c < w; c++) {
					*ptr++ = convert(*addr(ib, r, c));
					*ptr++ = convert(*addr(ig, r, c));
					*ptr++ = convert(*addr(ir, r, c));
				}
				//write strided line
				os.write(reinterpret_cast<char*>(imageRow.data()), stridedWidth);
			}
		}

		virtual void saveBmpPlanes(const std::string& filename) const override {
			std::ofstream os(filename, std::ios::binary);
			int h = this->type->h;
			int w = this->type->w;
			int planes = this->type->store->planes;

			BmpGrayHeader(w, h * planes).writeHeader(os);
			int stridedWidth = util::alignValue(w, 4);
			std::vector<uchar> imageRow(stridedWidth);

			for (int i = 0; i < planes; i++) {
				int idx = this->colorIndex[i];
				for (int r = h - 1; r >= 0; r--) {
					//prepare one line of data
					const T* src = addr(idx, r, 0);
					for (int c = 0; c < w; c++) {
						imageRow[c] = convert(*src);
						src += planes;
					}
					//write strided line
					os.write(reinterpret_cast<char*>(imageRow.data()), stridedWidth);
				}
			}
		}

	private:
		uchar convert(T value) const {
			if constexpr (std::is_same_v<uchar, T>) {
				return value;

			} else if constexpr (std::is_same_v<ushort, T>) {
				return value / 256;

			} else {
				return (uchar) std::round(value * 255 / this->maxValue);
			}
		}
	};


	//Image Adapter
	template <class T> class ImageAdapter : public IImage<T> {

	protected:
		std::shared_ptr<ImageStoreBase<T>> store;
		std::shared_ptr<ImageTypeBase<T>> type;
		std::shared_ptr<ImageColorBase<T>> color;

	public:
		virtual T* addr(size_t idx, size_t r, size_t c) { return type->addr(idx, r, c); }
		virtual const T* addr(size_t idx, size_t r, size_t c) const { return type->addr(idx, r, c); }

		virtual T* plane(size_t idx) { return store->plane(idx); }
		virtual const T* plane(size_t idx) const { return store->plane(idx); }

		virtual int height() const { return type->h; }
		virtual int width() const { return type->w; }
		virtual int planes() const { return store->planes; }
		virtual int widthInBytes() const { return type->w * sizeof(T); };
		virtual int stride() const { return type->stride; }
		virtual int strideInBytes() const { return type->stride * sizeof(T); }

		virtual std::vector<T> bytes() const { return store->bytes(); }

		virtual void saveBmpColor(const std::string& filename) const { color->saveBmpColor(filename); }
		virtual void saveBmpPlanes(const std::string& filename) const { color->saveBmpPlanes(filename); }
	};
}
