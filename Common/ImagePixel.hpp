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

#include <cassert>
#include <iostream>

namespace im {

	//container for pointers to one pixel
	template <class T> struct ImagePixel {
		std::array<T*, 4> pix = { nullptr, nullptr, nullptr, nullptr };
		int offset = 0;
		size_t size = 0;

		ImagePixel(std::initializer_list<T*> pix, int offset) :
			offset { offset },
			size { pix.size() }
		{
			std::copy(pix.begin(), pix.end(), this->pix.begin());
		}

		ImagePixel(T* a, T* b, T* c, T* d) :
			pix { a, b, c, d },
			offset { 1 },
			size { 4 }
		{}

		ImagePixel(T* a, T* b, T* c) :
			pix { a, b, c },
			offset { 1 },
			size { 3 }
		{}

		ImagePixel(T* a) :
			pix { a },
			offset { 1 },
			size { 1 }
		{}

		ImagePixel(size_t planes) :
			offset { 1 },
			size { planes }
		{}

		ImagePixel() :
			offset { 0 },
			size { 0 }
		{}

		ImagePixel<T>& operator ++ () {
			for (size_t i = 0; i < size; i++) pix[i] += offset;
			return *this;
		}

		ImagePixel<T> operator ++ (int ignore) {
			ImagePixel<T> old = *this;
			for (size_t i = 0; i < size; i++) pix[i] += offset;
			return old;
		}

		T* operator [] (size_t index) {
			assert(index < size && "invalid pixel");
			return pix[index];
		}
		
		const T* operator [] (size_t index) const {
			assert(index < size && "invalid pixel");
			return pix[index];
		}

		ImagePixel<T>& operator += (int delta) {
			for (size_t i = 0; i < pix.size(); i++) {
				pix[i] += delta;
			}
			return *this;
		}

		friend std::ostream& operator << (std::ostream& out, const ImagePixel<T>& pixel) {
			out << "values =";
			for (T* ptr : pixel.pix) out << " " << *ptr;
			out << ", offset = " << pixel.offset;
			return out;
		}

		void writeTo(ColorBase srcColor, ColorBase destColor, ImagePixel<uchar>& dest) const;

		void writeTo(ColorBase srcColor, ColorBase destColor, ImagePixel<float>& dest) const;
	};

}