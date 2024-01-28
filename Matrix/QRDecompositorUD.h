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

#include "Decompositor.h"

template <class T> class QRDecompositorUD : public Decompositor<T> {

private:
	Mat<T> A; //store transposed mat here to use in QR
	QRDecompositor<T> qr;

public:
	QRDecompositorUD(Mat<T>& mat) : 
		A { mat.trans() }, 
		qr(A) {}

	QRDecompositorUD<T>& compute() override {
		qr.compute();
		return *this;
	}

	Mat<T> getQ() {
		return qr.getQ();
	}

	Mat<T> getR() {
		return qr.getR();
	}

	bool isFullRank() {
		return qr.isFullRank();
	}

	std::optional<Mat<T>> solve(const Mat<T>& b) override {
		size_t m = qr.m;
		size_t n = qr.n;

		if (qr.dirty) compute();
		if (n >= m) return std::nullopt;
		if (isFullRank() == false) return std::nullopt;
		if (b.rows() != n) return std::nullopt;

		// x = Q * (R' \ b)
		Mat<T> y = b;
		size_t nx = b.cols();
		Mat<T> x = Mat<T>::zeros(m, nx);
		Mat<T> q = getQ();

		//columns of b
		for (size_t j = 0; j < nx; j++) {
			//rows of b
			for (size_t r = 0; r < n; r++) {
				y[r][j] /= qr.rd[r];
				for (size_t i = r + 1; i < n; i++) y[i][j] -= A[r][i] * y[r][j];
				for (size_t xr = 0; xr < m; xr++) x[xr][j] += y[r][j] * q[xr][r];
			}
		}
		return x;
	}
};
