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
		virtual T* row(size_t r, size_t h, size_t stride) = 0;
		virtual const T* row(size_t r, size_t h, size_t stride) const = 0;

		virtual T* data() = 0;
		virtual const T* data() const = 0;

		virtual size_t sizeInBytes() const = 0;
		virtual std::vector<T> bytes() const = 0;
		virtual void write(std::ostream& os) const = 0;
	};

	template <class T> class ImageStoreLocal : public ImageStoreBase<T> {

	private:
		std::vector<T> store;

	public:
		ImageStoreLocal(int siz = 0) :
			store(siz)
		{}

		virtual T* row(size_t r, size_t h, size_t stride) override {
			assert(r * stride < store.size() && "invalid row");
			return store.data() + r * stride;
		}

		virtual const T* row(size_t r, size_t h, size_t stride) const override {
			assert(r * stride < store.size() && "invalid row");
			return store.data() + r * stride;
		}

		virtual T* data() override {
			return store.data();
		}

		virtual const T* data() const override {
			return store.data();
		}

		virtual size_t sizeInBytes() const override {
			return store.size();
		}

		virtual std::vector<T> bytes() const override {
			return store;
		}

		virtual void write(std::ostream& os) const override {
			os.write(reinterpret_cast<const char*>(store.data()), store.size() * sizeof(T));
		}
	};

	template <class T> class ImageStoreShared : public ImageStoreBase<T> {

	private:
		std::vector<std::span<T>> store;

	public:
		ImageStoreShared(std::vector<std::span<T>> store = {{}}) :
			store { store }
		{}

		virtual T* row(size_t r, size_t h, size_t stride) override {
			size_t idx = r / h;
			size_t rr = r % h;
			assert(idx < store.size() && rr * stride < store[idx].size() && "invalid row");
			return store[idx].data() + rr * stride;
		}

		virtual const T* row(size_t r, size_t h, size_t stride) const override {
			size_t idx = r / h;
			size_t rr = r % h;
			assert(idx < store.size() && rr * stride < store[idx].size() && "invalid row");
			return store[idx].data() + rr * stride;
		}

		virtual T* data() override {
			return store.front().data();
		}

		virtual const T* data() const override {
			return store.front().data();
		}

		virtual size_t sizeInBytes() const {
			size_t siz = 0;
			for (auto& s : store) siz += s.size();
			return siz;
		}

		virtual std::vector<T> bytes() const {
			std::vector<T> data;
			for (auto& s : store) std::copy(s.begin(), s.end(), std::back_inserter(data));
			return data;
		}

		virtual void write(std::ostream& os) const override {
			for (auto& s : store) os.write(reinterpret_cast<const char*>(s.data()), s.size() * sizeof(T));
		}
	};

	template <class T> class ImageStoreSharedSingle : public ImageStoreBase<T> {

	private:
		std::span<T> store;

	public:
		ImageStoreSharedSingle(std::span<T> store = {}) :
			store { store }
		{}

		virtual T* row(size_t r, size_t h, size_t stride) override {
			assert(r * stride < store.size() && "invalid row");
			return store.data() + r * stride;
		}

		virtual const T* row(size_t r, size_t h, size_t stride) const override {
			assert(r * stride < store.size() && "invalid row");
			return store.data() + r * stride;
		}

		virtual T* data() override {
			return store.data();
		}

		virtual const T* data() const override {
			return store.data();
		}

		virtual size_t sizeInBytes() const {
			return store.size();
		}

		virtual std::vector<T> bytes() const {
			return std::vector<T>(store.begin(), store.end());
		}

		virtual void write(std::ostream& os) const override {
			os.write(reinterpret_cast<const char*>(store.data()), store.size() * sizeof(T));
		}
	};

} //namespace
