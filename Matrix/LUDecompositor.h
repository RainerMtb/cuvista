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
#include "MatInverter.h"

template <class T> class LUDecompositor : public Decompositor<T>, public MatInverter<T> {

	using Decompositor<T>::dirty;

protected:
	Mat<T>& A;
	size_t m, n;
	int pivsign = 0;
	std::vector<size_t> piv;

public:
	LUDecompositor(Mat<T>& mat) : 
		A { mat }, 
		m { mat.rows() }, 
		n { mat.cols() }, 
		piv(mat.rows()) {}

	LUDecompositor<T>& compute() override {
		for (size_t i = 0; i < m; i++) piv[i] = i;
		pivsign = 1;

		for (size_t j = 0; j < n; j++) {
			// Apply previous transformations.
			// Most of the time is spent in the following dot product.
			
			// more accurate version
			//for (size_t i = 1; i < m; i++) {
			//	T s = (T) 0;
			//	for (size_t k = 0; k < i && k < j; k++) s += A[i][k] * A[k][j];
			//	A[i][j] -= s;
			//}

			for (size_t i = 0; i < j; i++) {
				for (size_t k = i + 1; k < m; k++) {
					A[k][j] -= A[k][i] * A[i][j];
				}
			}

			//find pivot and exchange if necessary
			size_t p = j;
			for (size_t i = j + 1; i < m; i++) {
				if (std::abs(A[i][j]) > std::abs(A[p][j])) p = i;
			}
			if (p != j) {
				for (size_t k = 0; k < n; k++) std::swap(A[p][k], A[j][k]);
				std::swap(piv[p], piv[j]);
				pivsign = -pivsign;
			}

			// Compute multipliers.
			for (size_t i = j + 1; i < m; i++) A[i][j] /= A[j][j];
		}
		dirty = false;
		return *this;
	}

	/**
	 * L lower triangular matrix
	 * @return m x min(m, n)
	 */
	Mat<T> getL() {
		if (dirty) compute();
		return Mat<T>::generate(m, std::min(m, n), [this](size_t r, size_t c) {return r > c ? A[r][c] : r == c ? 1.0 : 0.0; });
	}

	/**
	 * U upper triangular matrix
	 * @return min(m, n) x n
	 */
	Mat<T> getU() {
		if (dirty) compute();
		return Mat<T>::generate(std::min(m, n), n, [this](size_t r, size_t c) {return r <= c ? A[r][c] : 0.0; });
	}

	/**
	 * P permutation matrix
	 * @return m x m
	 */
	Mat<T> getP() {
		if (dirty) compute();
		return Mat<T>::generate(m, m, [this](size_t r, size_t c) {return piv[r] == c ? 1.0 : 0.0; });
	}

	bool isSingular() {
		if (dirty) compute();
		for (size_t i = 0; i < std::min(m, n); i++) {
			if (std::abs(A[i][i]) < A.eps()) return true;
		}
		return false;
	}

	T det() {
		if (dirty) compute();
		if (m != n) throw std::runtime_error("Matrix must be square");
		T result = pivsign;
		for (size_t i = 0; i < std::min(m, n); i++) result *= A[i][i];
		return result;
	}

	bool solve(const Mat<T>& b, Mat<T>& x) {
		if (dirty) compute();
		if (b.rows() != m) return false;
		size_t nx = b.cols();

		// B(piv,:)
		for (size_t k = 0; k < nx; k++) {
			for (size_t i = 0; i < m; i++) {
				x[i][k] = b.at(piv[i], k);
			}
		}
		// Solve L*Y = B(piv,:)
		for (size_t k = 0; k < n; k++) {
			for (size_t i = k + 1; i < n; i++) {
				for (size_t j = 0; j < nx; j++) {
					x[i][j] -= x[k][j] * A[i][k]; //most time spent here
				}
			}
		}
		// Solve U*X = Y;
		for (size_t kk = n; kk > 0; kk--) {
			size_t k = kk - 1;
			for (size_t j = 0; j < nx; j++) {
				x[k][j] /= A[k][k];
			}
			for (size_t i = 0; i < k; i++) {
				for (size_t j = 0; j < nx; j++) {
					x[i][j] -= x[k][j] * A[i][k]; //most time spent here
				}
			}
		}
		return true;
	}

	std::optional<Mat<T>> solve(const Mat<T>& b) override {
		Mat<T> x = Mat<T>::allocate(b.rows(), b.cols());
		bool result = solve(b, x);
		if (result) return x;
		else return std::nullopt;
	}

	std::optional<Mat<T>> inv(Mat<T>& A) override {
		this->A = A;
		this->dirty = true;
		return solve(Mat<T>::eye(m));
	}

	std::optional<Mat<T>> inv() override {
		return solve(Mat<T>::eye(m));
	}

	void solveAffine(const Mat<T>& b, Mat<T>& dest) {
		compute();

		T* xp = dest.data();
		const T* bp = b.data();
		// B(piv,:)
		for (size_t i = 0; i < m; i++) {
			xp[i] = bp[piv[i]];
		}
		// Solve L*Y = B(piv,:)
		for (size_t k = 0; k < n; k++) {
			for (size_t i = k + 1; i < n; i++) {
				xp[i] -= xp[k] * A[i][k]; 
			}
		}
		// Solve U*X = Y;
		for (int64_t k = n - 1; k >= 0; k--) {
			xp[k] /= A[k][k];
			for (int64_t i = 0; i < k; i++) {
				xp[i] -= xp[k] * A[i][k]; 
			}
		}
	}
};
