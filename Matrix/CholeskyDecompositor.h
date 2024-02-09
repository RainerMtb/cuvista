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

// compute Cholesky Decomposition for square mat such that L * L' * x = b
template <class T> class CholeskyDecompositor : public Decompositor<T> {

	using Decompositor<T>::dirty;

protected:
	Mat<T>& A;
	size_t m, n;
	Mat<T> L;
	bool isspd = false;

public:
	CholeskyDecompositor(Mat<T>& mat) : 
		A { mat }, 
		m { mat.rows() }, 
		n { mat.cols() }, 
		L { Mat<T>::allocate(m, n) } {}

	CholeskyDecompositor<T>& compute() override {
		isspd = (m == n);

		// Main loop.
		for (size_t j = 0; j < n; j++) {
			MatRow<T> Lrowj = L[j];
			T d = (T) 0;
			for (size_t k = 0; k < j; k++) {
				MatRow<T> Lrowk = L[k];
				T s = (T) 0;
				for (size_t i = 0; i < k; i++) s += Lrowk[i] * Lrowj[i];
				Lrowj[k] = s = (A[j][k] - s) / L[k][k];
				d += s * s;
				isspd = isspd & (A[k][j] == A[j][k]); 
			}
			d = A[j][j] - d;
			isspd = isspd & (d > (T) 0);
			L[j][j] = std::sqrt(std::max(d, (T) 0));
			for (size_t k = j + 1; k < n; k++) L[j][k] = 0.0;
		}
		dirty = false;
		return *this;
	}

	bool isSPD () {
		if (dirty) compute();
		return isspd;
	}

	Mat<T> getL() {
		if (dirty) compute();
		return L.subMat(0, 0, m, n);
	}

	std::optional<Mat<T>> solve(const Mat<T>& b) override {
		if (dirty) compute();
		if (!isspd) return std::nullopt;
		if (b.rows() != m) return std::nullopt;

		// Copy right hand side.
		Mat<T> x = b;
		size_t nx = b.cols();

		// Solve L*Y = B;
		for (size_t k = 0; k < n; k++) {
			for (size_t j = 0; j < nx; j++) {
				for (size_t i = 0; i < k; i++) x[k][j] -= x[i][j] * L[k][i];
				x[k][j] /= L[k][k];
			}
		}

		// Solve L'*X = Y;
		for (size_t kk = n; kk > 0; kk--) {
			size_t k = kk - 1;
			for (size_t j = 0; j < nx; j++) {
				for (size_t i = k + 1; i < n; i++) x[k][j] -= x[i][j] * L[i][k];
				x[k][j] /= L[k][k];
			}
		}
		return x;
	}
};
