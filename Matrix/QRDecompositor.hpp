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

#include "Decompositor.hpp"

//need a forward declaration before we can declare friedship
template <class T> class QRDecompositorUD;

template <class T> class QRDecompositor : public Decompositor<T> {

	using Decompositor<T>::dirty;

	friend class QRDecompositorUD<T>;

protected:
	Mat<T>& A;
	size_t m, n;
	std::vector<T> rd;
	int pivsign = 0;

public:
	QRDecompositor(Mat<T>& mat) : 
		A { mat }, 
		m { mat.rows() }, 
		n { mat.cols() }, 
		rd(mat.cols()) {}

	QRDecompositor<T>& compute() override {
		//std::cout << "qr " << m << " by " << n << std::endl;
		for (size_t k = 0; k < n; k++) {
			T norm = (T) 0;
			for (size_t i = k; i < m; i++) norm = std::hypot(norm, A[i][k]);
			if (norm > std::numeric_limits<T>::epsilon()) {
				if (A[k][k] < T(0)) norm = -norm;
				for (size_t i = k; i < m; i++) A[i][k] /= norm;
				A[k][k] += 1.0;
				for (size_t j = k + 1; j < n; j++) {
					T acc = T (0);
					for (size_t i = k; i < m; i++) acc += A[i][k] * A[i][j];
					acc /= -A[k][k];
					for (size_t i = k; i < m; i++) A[i][j] += acc * A[i][k];
				}
			}
			rd[k] = -norm;
		}
		dirty = false;
		return *this;
	}

	std::optional<Mat<T>> solve(const Mat<T>& b) override {
		if (dirty) compute();
		/*
		 * to solve m < n
		 * A -> make qr of A' -> then solve Q * (R' \ b) -> x -> A * x = b
		 * should then be equal to pinv(A) * b -> x
		 */
		if (n > m) return std::nullopt;
		if (!isFullRank()) return std::nullopt;
		if (b.rows() != m) return std::nullopt;

		// Copy right hand side
		Mat<T> x = b;
		size_t nx = x.cols();

		// Compute Y = Q' * B
		for (size_t k = 0; k < n; k++) {
			for (size_t j = 0; j < nx; j++) {
				T s = T (0);
				for (size_t i = k; i < m; i++) s += A[i][k] * x[i][j];
				s /= -A[k][k];
				for (size_t i = k; i < m; i++) x[i][j] += s * A[i][k];
			}
		}
		// Solve R * X = Y;
		for (size_t kk = n; kk > 0; kk--) {
			size_t k = kk - 1;
			for (size_t j = 0; j < nx; j++) x[k][j] /= rd[k];
			for (size_t i = 0; i < k; i++) {
				for (size_t j = 0; j < nx; j++) {
					x[i][j] -= x[k][j] * A[i][k];
				}
			}
		}
		return x.subMat(0, 0, n, nx);
	}

	Mat<T>& getQ(Mat<T>& Q) { //dim [m x n]
		if (dirty) compute();
		size_t u = std::min(m, n);
		Q.setValues(T(0));

		for (size_t kk = u; kk > 0; kk--) {
			size_t k = kk - 1;
			Q[k][k] = T(1);
			for (size_t j = k; j < u; j++) {
				if (std::abs(A[k][k]) > std::numeric_limits<T>::epsilon()) {
					T s = T(0);
					for (size_t i = k; i < m; i++) s += A[i][k] * Q[i][j];
					s /= -A[k][k];
					for (size_t i = k; i < m; i++) Q[i][j] += s * A[i][k];
				}
			}
		}
		return Q;
	}

	Mat<T> getQ() { //dim [m x n]
		if (dirty) compute();
		Mat<T> q = Mat<T>::allocate(m, n);
		return getQ(q);
	}

	Mat<T>& getR(Mat<T>& r) { //dim [n x n]
		if (dirty) compute();
		auto fcn = [&] (size_t r, size_t c) {
			if (r < c)
				return A[r][c];
			else if (r == c)
				return rd[r];
			else
				return T(0);
		};
		return r.setArea(fcn);
	}

	Mat<T> getR() { //dim [n x n]
		if (dirty) compute();
		Mat<T> r = Mat<T>::allocate(n, n);
		return getR(r);
	}

	bool isFullRank() {
		if (dirty) compute();
		for (size_t i = 0; i < std::min(m, n); i++) {
			if (std::abs(rd[i]) < std::numeric_limits<T>::epsilon() * 1000) return false;
		}
		return true;
	}

	//compute eigenvalues via repeated qr decomposition
	Mat<T> eig() {
		if (m != n) return Mat<T>();

		Mat<T> Anew = Mat<T>::allocate(m, m);
		Mat<T> Q = Mat<T>::allocate(m, m);
		Mat<T> d = Mat<T>::zeros(m, 1);

		T norm = (T) 1;
		int i = 0;
		while (norm > std::numeric_limits<T>::epsilon() && i < 500) {
			//decompose qr
			compute();

			//calcluate r * q
			getQ(Q);
			for (size_t r = 0; r < m; r++) {
				for (size_t c = r; c < m; c++) {
					T sum = rd[r] * Q.at(r, c);
					for (size_t i = r + 1; i < m; i++) {
						sum += A.at(r, i) * Q.at(i, c);
					}
					Anew.at(r, c) = Anew.at(c, r) = sum;
				}
			}
			std::swap(Anew.array, A.array);

			//see if iteration has reached stable state
			norm = (T) 0;
			for (size_t i = 0; i < m; i++) {
				double dnew = A.at(i, i);
				double delta = d.at(i, 0) - dnew;
				norm += delta * delta;
				d.at(i, 0) = dnew;
			}

			i++;
		}
		return d;
	}

	T norm2() {
		return eig().at(0, 0);
	}
};
