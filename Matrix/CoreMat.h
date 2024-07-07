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

#include <cstddef>
#include <cassert>
#include <algorithm>
#include <optional>

template <class T> class CoreMat {

private:

	template <class R = T> R interpFunc(size_t ix, size_t iy, R dx, R dy) const;

	template <class R = T> R interpFunc(R x, R y) const;

	template <class R = T> std::optional<R> interpFunc(R x, R y, size_t x0, size_t y0, size_t w, size_t h) const;

protected:
	T* array;		//main data array
	size_t h, w;	//dimensions of mat
	bool ownData;	//object owns and therefore destructs data array

	//create new mat, allocate data array
	CoreMat(size_t rows, size_t cols) : 
		CoreMat<T>(new T[rows * cols], rows, cols, true) {}

	//create new mat filled with given value
	CoreMat(size_t rows, size_t cols, T value) :
		CoreMat<T>(rows, cols)
	{
		std::fill(array, array + rows * cols, value);
	}

	//create mat using existing data array, sharing memory
	CoreMat(T* array, size_t rows, size_t cols, bool ownData) : 
		array { array },
		h { rows },
		w { cols },
		ownData { ownData } {}

	//return height for index 0 or width for index 1
	size_t dim(size_t dimIdx) const;

	//check if given index values are valid to this mat
	virtual bool isValidIndex(size_t row, size_t col) const;

	//create index into data array
	virtual size_t index(size_t row, size_t col) const;

public:
	//default constructor produces invalid mat
	CoreMat();

	//copy constructor
	CoreMat(const CoreMat<T>& other);

	//move constructor
	CoreMat(CoreMat<T>&& other) noexcept;

	//virtual destructor
	virtual ~CoreMat();

	//copy assignment
	CoreMat<T>& operator = (const CoreMat<T>& other);

	//move assignment
	CoreMat<T>& operator = (CoreMat<T>&& other) noexcept;

	//return pointer to element without boundary check
	virtual T* addr(size_t row, size_t col);

	//return pointer to element without boundary check
	virtual const T* addr(size_t row, size_t col) const;

	friend void swap(CoreMat<T>& a, CoreMat<T>& b) noexcept {
		std::swap(a.array, b.array);
		std::swap(a.w, b.w);
		std::swap(a.h, b.h);
		std::swap(a.ownData, b.ownData);
	}

	CoreMat<T> shareData();

	//reference to value at given position
	T& at(size_t row, size_t col);

	//reference to value at given position
	const T& at(size_t row, size_t col) const;

	//pointer to data array
	T* data();

	//constant pointer to data array
	const T* data() const;

	//number of rows
	virtual size_t rows() const;

	//number of columns
	virtual size_t cols() const;

	//number of elements
	virtual size_t numel() const;

	//interpolate this mat at given point x and y, return optional, return nullopt when outside this mat
	std::optional<float> interp2(float x, float y) const;

	//interpolate this mat at given point x and y, return optional, return nullopt when outside this mat
	std::optional<double> interp2(double x, double y) const;

	//interpolate this mat at given point x and y, clamp to boundaries
	float interp2clamped(float x, float y) const;

	//interpolate this mat at given point x and y, clamp to boundaries
	double interp2clamped(double x, double y) const;

	//interpolate this mat at given points through index ix and iy and fractions dx and dy and dx*dy
	float interp2(size_t ix, size_t iy, float dx, float dy) const;

	//interpolate this mat at given points through index ix and iy and fractions dx and dy and dx*dy
	double interp2(size_t ix, size_t iy, double dx, double dy) const;
};