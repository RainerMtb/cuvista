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

#include <cmath>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <optional>

template <class T> class CoreMat {

private:

	template <class R = T> R interpFunc(size_t ix, size_t iy, R dx, R dy) const {
		T f00 = at(iy, ix);
		size_t x = dx != 0;
		size_t y = dy != 0;
		T f01 = at(iy, ix + x);
		T f10 = at(iy + y, ix);
		T f11 = at(iy + y, ix + x);
		R result = (R) ((1 - dx) * (1 - dy) * f00 + (1 - dx) * dy * f10 + dx * (1 - dy) * f01 + dx * dy * f11);
		return result;
	}

	template <class R = T> R interpFunc(R x, R y) const {
		R flx = std::floor(x), fly = std::floor(y);
		R dx = x - flx, dy = y - fly;
		size_t ix = size_t(flx), iy = size_t(fly);
		return interpFunc(ix, iy, dx, dy);
	}

	template <class R = T> std::optional<R> interpFunc(R x, R y, size_t x0, size_t y0, size_t w, size_t h) const {
		if (x < x0 || x > x0 + w - 1 || y < y0 || y > y0 + h - 1) return std::nullopt;
		return interpFunc(x, y);
	}

protected:
	T* array;		//main data array
	size_t h, w;	//dimensions of mat
	bool ownData;	//object owns and therefore destructs data array

	//create new mat, allocate data array
	CoreMat(size_t rows, size_t cols) : 
		CoreMat<T>(new T[rows * cols], rows, cols, true) 
	{}

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
		ownData { ownData } 
	{}

	//return height for index 0 or width for index 1
	size_t dim(size_t dimIdx) const {
		return dimIdx == 0 ? rows() : cols();
	}

	//check if given index values are valid to this mat
	virtual bool isValidIndex(size_t row, size_t col) const {
		return row < h && col < w;
	}

	//create index into data array
	virtual size_t index(size_t row, size_t col) const {
		return row * w + col;
	}

public:
	//default constructor produces invalid mat
	CoreMat() :
		CoreMat<T>(nullptr, 0, 0, true) 
	{}

	//copy constructor
	CoreMat(const CoreMat<T>& other) :
		CoreMat<T>(nullptr, other.h, other.w, true)
	{
		//std::cout << "!! 1 copy constructor" << std::endl;
		//when template type is const, need to allocate a non const array to be able to write to it
		using TT = std::remove_const_t<T>;
		size_t numel = other.h * other.w;

		TT* newArray = new TT[numel];
		std::copy_n(other.array, numel, newArray);
		array = newArray;
	}

	//move constructor
	CoreMat(CoreMat<T>&& other) noexcept :
		CoreMat<T>(other.array, other.h, other.w, true)
	{
		//std::cout << "!! 2 move constructor" << std::endl;
		other.array = nullptr;
		other.h = 0;
		other.w = 0;
		other.ownData = false;
	}

	//virtual destructor
	virtual ~CoreMat() {
		//std::cout << "!! destructor " << &array << " " << h << ", " << w << std::endl;
		if (ownData) {
			delete[] array;
		}
	}

	//copy assignment
	CoreMat<T>& operator = (const CoreMat<T>& other) {
		//std::cout << "!! copy assignment" << std::endl;
		if (this != &other) {
			CoreMat<T> matCopy = other;
			swap(*this, matCopy);
		}
		return *this;
	}

	//move assignment
	CoreMat<T>& operator = (CoreMat<T>&& other) noexcept {
		//std::cout << "!! move assignment" << std::endl;
		if (this != &other) {
			swap(*this, other);
		}
		return *this;
	}

	//return pointer to element without boundary check
	virtual T* addr(size_t row, size_t col) {
		return array + row * w + col;
	}

	//return pointer to element without boundary check
	virtual const T* addr(size_t row, size_t col) const {
		return array + row * w + col;
	}

	friend void swap(CoreMat<T>& a, CoreMat<T>& b) noexcept {
		std::swap(a.array, b.array);
		std::swap(a.w, b.w);
		std::swap(a.h, b.h);
		std::swap(a.ownData, b.ownData);
	}

	//new CoreMat sharing data
	CoreMat<T> shareData() {
		return CoreMat<T>(array, h, w, false);
	}

	//reference to value at given position
	T& at(size_t row, size_t col) {
		assert(row < h && col < w && "mat access out of bounds");
		return *addr(row, col);
	}

	//reference to value at given position
	const T& at(size_t row, size_t col) const {
		assert(row < h && col < w && "mat access out of bounds");
		return *addr(row, col);
	}

	//pointer to data array
	T* data() {
		return array;
	}

	//constant pointer to data array
	const T* data() const {
		return array;
	}

	//number of rows
	virtual size_t rows() const {
		return h;
	}

	//number of columns
	virtual size_t cols() const {
		return w;
	}

	//number of elements
	virtual size_t numel() const {
		return cols() * rows();
	}

	//interpolate this mat at given point x and y, return optional, return nullopt when outside this mat
	std::optional<float> interp2(float x, float y) const {
		return interpFunc(x, y, 0, 0, cols(), rows());
	}

	//interpolate this mat at given point x and y, return optional, return nullopt when outside this mat
	std::optional<double> interp2(double x, double y) const {
		return interpFunc(x, y, 0, 0, cols(), rows());
	}

	//interpolate this mat at given point x and y, clamp to boundaries
	float interp2clamped(float x, float y) const {
		return interpFunc(std::clamp(x, 0.0f, cols() - 1.0f), std::clamp(y, 0.0f, rows() - 1.0f));
	}

	//interpolate this mat at given point x and y, clamp to boundaries
	double interp2clamped(double x, double y) const {
		return interpFunc(std::clamp(x, 0.0, cols() - 1.0), std::clamp(y, 0.0, rows() - 1.0));
	}

	//interpolate this mat at given points through index ix and iy and fractions dx and dy and dx*dy
	float interp2(size_t ix, size_t iy, float dx, float dy) const {
		return interpFunc(ix, iy, dx, dy);
	}

	//interpolate this mat at given points through index ix and iy and fractions dx and dy and dx*dy
	double interp2(size_t ix, size_t iy, double dx, double dy) const {
		return interpFunc(ix, iy, dx, dy);
	}
};