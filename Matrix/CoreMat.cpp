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

#include "CoreMat.h"
#include <cmath>

//copy assignment
template <class T> CoreMat<T>& CoreMat<T>::operator = (const CoreMat<T>& other) {
	//std::cout << "!! copy assignment" << std::endl;
	if (this != &other) {
		CoreMat<T> matCopy = other;
		swap(*this, matCopy);
	}
	return *this;
}

//move assignment
template <class T> CoreMat<T>& CoreMat<T>::operator = (CoreMat<T>&& other) noexcept {
	//std::cout << "!! move assignment" << std::endl;
	if (this != &other) {
		swap(*this, other);
	}
	return *this;
}

//number of rows
template <class T> size_t CoreMat<T>::rows() const { 
	return h; 
}

//number of columns
template <class T> size_t CoreMat<T>::cols() const { 
	return w; 
}

//number of elements
template <class T> size_t CoreMat<T>::numel() const { 
	return cols() * rows(); 
}

//pointer to data array
template <class T> T* CoreMat<T>::data() { 
	return array; 
}

//constant pointer to data array
template <class T> const T* CoreMat<T>::data() const { 
	return array; 
}

//return height for index 0 or width for index 1
template <class T> size_t CoreMat<T>::dim(size_t dimIdx) const {
	return dimIdx == 0 ? rows() : cols();
}

//check if given index values are valid to this mat
template <class T> bool CoreMat<T>::isValidIndex(size_t row, size_t col) const {
	return row < h && col < w;
}

//create index into data array
template <class T> size_t CoreMat<T>::index(size_t row, size_t col) const {
	return row * w + col;
}

template <class T> template <class R> R CoreMat<T>::interpFunc(size_t ix, size_t iy, R dx, R dy) const {
	T f00 = at(iy, ix);
	T f01 = dx == 0 ? f00 : at(iy, ix + 1);
	T f10 = dy == 0 ? f00 : at(iy + 1, ix);
	T f11 = dx == 0 || dy == 0 ? f00 : at(iy + 1, ix + 1);
	return (R) ((1 - dx) * (1 - dy) * f00 + (1 - dx) * dy * f10 + dx * (1 - dy) * f01 + dx * dy * f11);
}

template <class T> float CoreMat<T>::interp2(size_t ix, size_t iy, float dx, float dy) const {
	return interpFunc(ix, iy, dx, dy);
}

template <class T> double CoreMat<T>::interp2(size_t ix, size_t iy, double dx, double dy) const {
	return interpFunc(ix, iy, dx, dy);
}

template <class T> template <class R> std::optional<R> CoreMat<T>::interpFunc(R x, R y, size_t x0, size_t y0, size_t w, size_t h) const {
	if (x < x0 || x > x0 + w - 1 || y < y0 || y > y0 + h - 1) return std::nullopt;
	R flx = std::floor(x), fly = std::floor(y);
	R dx = x - flx, dy = y - fly;
	size_t ix = size_t(flx), iy = size_t(fly);
	return interp2(ix, iy, dx, dy);
}

template <class T> std::optional<float> CoreMat<T>::interp2(float x, float y, size_t x0, size_t y0, size_t w, size_t h) const {
	return interpFunc(x, y, x0, y0, w, h);
}

template <class T> std::optional<double> CoreMat<T>::interp2(double x, double y, size_t x0, size_t y0, size_t w, size_t h) const {
	return interpFunc(x, y, x0, y0, w, h);
}

template <class T> std::optional<float> CoreMat<T>::interp2(float x, float y) const {
	return interp2(x, y, 0, 0, cols(), rows());
}

template <class T> std::optional<double> CoreMat<T>::interp2(double x, double y) const {
	return interp2(x, y, 0, 0, cols(), rows());
}

//------------------------------------------------
//explicitly instantiate class specializations
//------------------------------------------------

template class CoreMat<unsigned char>;
template class CoreMat<double>;
template class CoreMat<float>;
template class CoreMat<int>;
