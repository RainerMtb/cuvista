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

#include "CoreMat.hpp"

//Iterator through elements of a Mat
template <class T> class MatIterator {

private:
	size_t i;
	Direction dir;
	CoreMat<T>* mat;

	T* index(size_t i) {
		size_t r = 0;
		size_t c = 0;
		if (dir == Direction::HORIZONTAL) {
			r = i / mat->cols();
			c = i % mat->cols();

		} else {
			r = i % mat->rows();
			c = i / mat->rows();
		}
		return mat->addr(r, c);
	}

public:
	using iterator_category = std::random_access_iterator_tag;
	using difference_type   = std::ptrdiff_t;
	using value_type        = T;
	using pointer           = T*;
	using reference         = T&;

	MatIterator(size_t i, Direction dir, CoreMat<T>* mat) :
		i { i }, 
		dir { dir },
		mat { mat } 
	{}

	MatIterator() :
		MatIterator<T>(0, Direction::HORIZONTAL, nullptr)
	{}

	T& operator * () { return *index(i); }

	T& operator [] (size_t ii) { return *index(ii); }

	T* operator -> () { return index(i); }

	bool operator != (const MatIterator& other) { return i != other.i; }

	bool operator == (const MatIterator& other) { return i == other.i; }

	bool operator < (const MatIterator& other) { return i < other.i; }

	MatIterator& operator ++ () { i++; return *this; }

	MatIterator operator ++ (int) { MatIterator temp = *this; i++; return temp; }

	MatIterator& operator -- () { i--; return *this; }

	MatIterator operator -- (int) { MatIterator temp = *this; i--; return temp; }

	difference_type operator - (const MatIterator& other) const { return i - other.i; }
};


//Iterator through elements of a const Mat
template <class T> class ConstMatIterator {

private:
	size_t i;
	Direction dir;
	const CoreMat<T>* mat;

	const T* index(size_t i) const {
		size_t r = 0;
		size_t c = 0;
		if (dir == Direction::HORIZONTAL) {
			r = i / mat->cols();
			c = i % mat->cols();

		} else {
			r = i % mat->rows();
			c = i / mat->rows();
		}
		return mat->addr(r, c);
	}

public:
	using iterator_category = std::random_access_iterator_tag;
	using difference_type   = std::ptrdiff_t;
	using value_type        = T;
	using pointer           = T*;
	using reference         = T&;

	ConstMatIterator(size_t i, Direction dir, const CoreMat<T>* mat) :
		i { i }, 
		dir { dir },
		mat { mat } 
	{}

	ConstMatIterator() :
		ConstMatIterator<T>(0, Direction::HORIZONTAL, nullptr)
	{}

	const T& operator * () const { return *index(i); }

	const T& operator [] (size_t ii) const { return *index(ii); }

	const T* operator -> () const { return index(i); }

	bool operator != (const ConstMatIterator& other) { return i != other.i; }

	bool operator == (const ConstMatIterator& other) {  return i == other.i; }

	bool operator < (const ConstMatIterator& other) { return i < other.i; }

	ConstMatIterator& operator ++ () { i++; return *this; }

	ConstMatIterator operator ++ (int) { ConstMatIterator temp = *this; i++; return temp; }

	ConstMatIterator& operator -- () { i--; return *this; }

	ConstMatIterator operator -- (int) { ConstMatIterator temp = *this; i--; return temp; }

	difference_type operator - (const ConstMatIterator& other) const { return i - other.i; }
};
