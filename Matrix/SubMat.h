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

template <class T> class Mat;
template <class T> class MatRow;

template <class T> class SubMat : public Mat<T> {

	friend class Mat<T>; //constructor of SubMat can only be called from inside Mat
	friend class MatRow<T>;

protected:
	T* subArray;
	size_t r0, c0;		//offset into data used for shared mat
	size_t hs, ws;		//dimensions of this mat

	//return pointer to element without boundary check
	virtual T* addr(size_t row, size_t col) {
		return subArray + row * this->w + col;
	}

	//return pointer to element without boundary check
	virtual const T* addr(size_t row, size_t col) const {
		return subArray + row * this->w + col;
	}

private:
	SubMat<T>(T* array, size_t r0, size_t c0, size_t hs, size_t ws, size_t h, size_t w) : 
		Mat<T>(array, h, w, false), 
		subArray { array + r0 * w + c0 }, 
		r0 { r0 }, 
		c0 { c0 }, 
		hs { hs }, 
		ws { ws } {}

	//check if given index values are valid to this mat
	virtual bool isValidIndex(size_t row, size_t col) const override {
		return row < hs && col < ws;
	}

	//create index into data array
	virtual size_t index(size_t row, size_t col) const override {
		return (r0 + row) * this->w + c0 + col;
	}

	//translate index into submat to index into main array based on direction row major or col major
	virtual std::function<size_t(size_t)> indexFunc(Mat<T>::Direction dir) const {
		if (dir == Mat<T>::Direction::HORIZONTAL) return [&] (size_t i) { return r0 * this->w + c0 + i / ws * this->w + i % ws; };
		else return [&] (size_t i) { return r0 * this->w + c0 + i % hs * this->w + i / hs; };
	}

public:
	SubMat<T>() : 
		Mat<T>() {}

	virtual MatRow<T> operator [] (size_t row) override {
		assert(row < hs && "row index out of bounds");
		return MatRow<T>(subArray + row * this->w, ws);
	}

	virtual MatRow<const T> operator [] (size_t row) const override {
		assert(row < hs && "row index out of bounds");
		return MatRow<const T>(subArray + row * this->w, ws);
	}

	//number of rows of this submat
	virtual size_t rows() const override { 
		return hs; 
	}

	//number of columns of this submat
	virtual size_t cols() const override { 
		return ws; 
	}

	//create shared sub mat of shared sub mat
	virtual SubMat<T> subMatShared(size_t r0, size_t c0, size_t hs, size_t ws) override {
		assert(this->r0 + r0 + hs <= this->h && this->c0 + c0 + ws <= this->w && "invalid index arguments");
		return SubMat<T>(this->array, this->r0 + r0, this->c0 + c0, hs, ws, this->h, this->w);
	}
};