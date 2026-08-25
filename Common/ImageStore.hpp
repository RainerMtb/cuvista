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

	template <class T> class ImageStore {

	public:
		int h, w, stride, planes;

	protected:
		int storeSize;
		YAxisDir ydir;
		std::shared_ptr<T[]> store;
		T* firstRow = nullptr;
		int rowOffset = 0;

		ImageStore(int h, int w, int stride, int planes, int storeSize, YAxisDir ydir, std::shared_ptr<T[]> store) :
			h { h },
			w { w },
			stride { stride },
			planes { planes },
			storeSize { storeSize },
			ydir { ydir },
			store { store }
		{
			if (ydir == YAxisDir::DOWN) {
				firstRow = this->store.get();
				rowOffset = stride;

			} else {
				firstRow = this->store.get() + (h - 1) * stride;
				rowOffset = -stride;
			}
		}

	public:
		ImageStore(int h, int w, int stride, int planes, int storeSize, YAxisDir ydir, T* data) :
			ImageStore<T>(h, w, stride, planes, storeSize, ydir, std::shared_ptr<T[]>(data, [] (auto ptr) {}))
		{}

		ImageStore(int h, int w, int stride, int planes, int storeSize, YAxisDir ydir) :
			ImageStore<T>(h, w, stride, planes, storeSize, ydir, std::make_shared<T[]>(storeSize))
		{}

		ImageStore() :
			ImageStore<T>(0, 0, 0, 0, 0, YAxisDir::DOWN)
		{}

		virtual T* row(size_t r) {
			T* ptr = firstRow + r * rowOffset;
			assert(ptr >= this->store.get() && ptr < this->store.get() + this->storeSize && "invalid row");
			return ptr;
		}

		virtual const T* row(size_t r) const {
			const T* ptr = firstRow + r * rowOffset;
			assert(ptr >= this->store.get() && ptr < this->store.get() + this->storeSize && "invalid row");
			return ptr;
		}

		virtual T* data() {
			return this->store.get();
		}

		virtual const T* data() const {
			return this->store.get();
		}

		virtual size_t sizeInBytes() const {
			return this->storeSize;
		}

		virtual std::vector<T> bytes() const {
			return { this->store.get(), this->store.get() + this->storeSize };
		}

		virtual void write(std::ostream& os) const {
			os.write(reinterpret_cast<const char*>(this->store.get()), this->storeSize * sizeof(T));
		}
	};

} //namespace
