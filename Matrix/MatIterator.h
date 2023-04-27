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

//Iterator through elements of a Mat

template <class T> class MatIterator {

protected:
	T* ptr;
	size_t i;
	std::function<size_t(size_t)> indexFunc;

public:
	using iterator_category = std::random_access_iterator_tag;
	using difference_type   = std::ptrdiff_t;
	using value_type        = T;
	using pointer           = T*;
	using reference         = T&;

	MatIterator(T* ptr, size_t i, std::function<size_t(size_t)> indexFunc) : 
		ptr { ptr }, 
		i { i }, 
		indexFunc { indexFunc } 
	{}

	MatIterator() : MatIterator(nullptr, 0, nullptr) {}

	T& operator * () { return *(ptr + indexFunc(i)); }

	T& operator [] (size_t ii) { return *(ptr + indexFunc(ii)); }

	T* operator -> () { return ptr + indexFunc(i); }

	bool operator != (const MatIterator& other) const { return ptr + i != other.ptr + other.i; }

	bool operator == (const MatIterator& other) const { return ptr + i == other.ptr + other.i; }

	bool operator <=> (const MatIterator& other) const { return other.ptr + other.i - ptr - i; }

	MatIterator& operator ++ () { i++; return *this; }

	MatIterator operator ++ (int) { MatIterator temp = *this; i++; return temp; }

	MatIterator& operator -- () { i--; return *this; }

	MatIterator operator -- (int) { MatIterator temp = *this; i--; return temp; }

	difference_type operator - (const MatIterator& other) const { return other.i - i; }
};
