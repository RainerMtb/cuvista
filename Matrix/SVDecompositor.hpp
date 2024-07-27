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
#include "MatInverter.hpp"

template <class T> class SVDecompositor : public Decompositor<T>, public MatInverter<T> {

	using Decompositor<T>::dirty;

protected:
	Mat<T>& A;
	size_t m, n;

private:
	Mat<T> U; // needs m x n
	Mat<T> V; // needs n x n
	std::vector<T> s, sinv, e, work;

public:
	SVDecompositor(Mat<T>& mat) : 
		A { mat }, 
		m { mat.rows() }, 
		n { mat.cols() },
		U { Mat<T>::zeros(mat.rows(), mat.cols()) }, 
		V { Mat<T>::allocate(mat.cols(), mat.cols()) },
		s(std::min(mat.rows() + 1, mat.cols())), 
		sinv(std::min(mat.rows(), mat.cols())), 
		e(mat.cols()), 
		work(mat.rows()) {}

	SVDecompositor<T>& compute() override {
		int n = (int) this->n;
		int m = (int) this->m;
		if (n > m) throw std::runtime_error("cannot decompose underdetermined systems n > m");
		int nu = std::min(m, n);
		bool wantu = true;
		bool wantv = true;

		// Reduce A to bidiagonal form, storing the diagonal elements in s and the super-diagonal elements in e.
		int nct = std::min(m - 1, n);
		int nrt = std::max(0, std::min(n - 2, m));
		for (int k = 0; k < std::max(nct, nrt); k++) {
			if (k < nct) {
				// Compute the transformation for the k-th column and
				// place the k-th diagonal in s[k].
				// Compute 2-norm of k-th column without under/overflow.
				s[k] = 0;
				for (int i = k; i < m; i++) s[k] = hypot(s[k], A[i][k]);
				if (s[k] != (T) 0) {
					if (A[k][k] < (T) 0) s[k] = -s[k];
					for (int i = k; i < m; i++) A[i][k] /= s[k];
					A[k][k] += (T) 1;
				}
				s[k] = -s[k];
			}
			for (int j = k + 1; j < n; j++) {
				if ((k < nct) && (s[k] != (T) 0)) {

					// Apply the transformation.
					T t = 0;
					for (int i = k; i < m; i++) t += A[i][k] * A[i][j];
					t = -t / A[k][k];
					for (int i = k; i < m; i++) A[i][j] += t * A[i][k];
				}

				// Place the k-th row of A into e for the subsequent calculation of the row transformation.
				e[j] = A[k][j];
			}
			if (wantu && (k < nct)) {
				// Place the transformation in U for subsequent back multiplication.
				for (int i = k; i < m; i++) U[i][k] = A[i][k];
			}
			if (k < nrt) {
				// Compute the k-th row transformation and place the k-th super-diagonal in e[k].
				// Compute 2-norm without under/overflow.
				e[k] = 0;
				for (int i = k + 1; i < n; i++) e[k] = hypot(e[k], e[i]);
				if (e[k] != (T) 0) {
					if (e[k + 1] < (T) 0) e[k] = -e[k];
					for (int i = k + 1; i < n; i++) e[i] /= e[k];
					e[k + 1] += (T) 1;
				}
				e[k] = -e[k];
				if ((k + 1 < m) && (e[k] != (T) 0)) {

					// Apply the transformation.
					for (int i = k + 1; i < m; i++) work[i] = (T) 0;
					for (int j = k + 1; j < n; j++) {
						for (int i = k + 1; i < m; i++) {
							work[i] += e[j] * A[i][j];
						}
					}
					for (int j = k + 1; j < n; j++) {
						T t = -e[j] / e[k + 1];
						for (int i = k + 1; i < m; i++) A[i][j] += t * work[i];
					}
				}
				if (wantv) {
					// Place the transformation in V for subsequent back multiplication.
					for (int i = k + 1; i < n; i++) V[i][k] = e[i];
				}
			}
		}

		// Set up the final bidiagonal matrix or order p.
		int p = std::min(n, m + 1);
		if (nct < n) s[nct] = A[nct][nct];
		if (m < p) s[p - 1] = (T) 0;
		if (nrt + 1 < p) e[nrt] = A[nrt][p - 1];
		e[p - 1] = (T) 0;

		// If required, generate U.
		if (wantu) {
			for (int j = nct; j < nu; j++) {
				for (int i = 0; i < m; i++) U[i][j] = (T) 0;
				U[j][j] = (T) 1;
			}
			for (int kk = nct; kk > 0; kk--) {
				int k = kk - 1;
				if (s[k] != (T) 0) {
					for (int j = k + 1; j < nu; j++) {
						T t = (T) 0;
						for (int i = k; i < m; i++) t += U[i][k] * U[i][j];
						t = -t / U[k][k];
						for (int i = k; i < m; i++) U[i][j] += t * U[i][k];
					}
					for (int i = k; i < m; i++) U[i][k] = -U[i][k];
					U[k][k] = (T) 1 + U[k][k];
					for (int i = 0; i < k - 1; i++) U[i][k] = (T) 0;

				}
				else {
					for (int i = 0; i < m; i++) U[i][k] = (T) 0;
					U[k][k] = (T) 1;
				}
			}
		}

		// If required, generate V.
		if (wantv) {
			for (int kk = n; kk > 0; kk--) {
				int k = kk - 1;
				if ((k < nrt) && (e[k] != (T) 0)) {
					for (int j = k + 1; j < nu; j++) {
						T t = 0;
						for (int i = k + 1; i < n; i++) t += V[i][k] * V[i][j];
						t = -t / V[k + 1][k];
						for (int i = k + 1; i < n; i++) V[i][j] += t * V[i][k];
					}
				}
				for (int i = 0; i < n; i++) V[i][k] = (T) 0;
				V[k][k] = (T) 1;
			}
		}

		// Main iteration loop for the singular values.
		int pp = p - 1;
		int iter = 0;
		while (p > 0) {
			int k, kase;

			// Here is where a test for too many iterations would go.

			// This section of the program inspects for
			// negligible elements in the s and e arrays.  On
			// completion the variables kase and k are set as follows.
			// kase = 1     if s(p) and e[k-1] are negligible and k<p
			// kase = 2     if s(k) is negligible and k<p
			// kase = 3     if e[k-1] is negligible, k<p, and s(k), ..., s(p) are not negligible (qr step).
			// kase = 4     if e(p-1) is negligible (convergence).

			for (k = p - 2; k >= -1; k--) {
				if (k == -1) break;
				if (std::abs(e[k]) <= std::numeric_limits<T>::min() + std::numeric_limits<T>::epsilon() * (std::abs(s[k]) + std::abs(s[k + 1]))) {
					e[k] = (T) 0;
					break;
				}
			}
			if (k == p - 2) {
				kase = 4;

			}
			else {
				int ks;
				for (ks = p - 1; ks >= k; ks--) {
					if (ks == k) break;
					T t = (ks != p ? std::abs(e[ks]) : 0.0) + (ks != k + 1 ? std::abs(e[ks - 1]) : 0.0);
					if (std::abs(s[ks]) <= std::numeric_limits<T>::min() + std::numeric_limits<T>::epsilon() * t) {
						s[ks] = (T) 0;
						break;
					}
				}
				if (ks == k) {
					kase = 3;

				}
				else if (ks == p - 1) {
					kase = 1;

				}
				else {
					kase = 2;
					k = ks;
				}
			}
			k++;

			// Perform the task indicated by kase.
			switch (kase) {

			// Deflate negligible s(p).
			case 1: {
				T f = e[p - 2];
				e[p - 2] = (T) 0;
				for (int j = p - 2; j >= k; j--) {
					T t = hypot(s[j], f);
					T cs = s[j] / t;
					T sn = f / t;
					s[j] = t;
					if (j != k) {
						f = -sn * e[j - 1];
						e[j - 1] = cs * e[j - 1];
					}
					if (wantv) {
						for (int i = 0; i < n; i++) {
							t = cs * V[i][j] + sn * V[i][p - 1];
							V[i][p - 1] = -sn * V[i][j] + cs * V[i][p - 1];
							V[i][j] = t;
						}
					}
				}
			}
			break;

			// Split at negligible s(k).
			case 2: {
				T f = e[k - 1];
				e[k - 1] = (T) 0;
				for (int j = k; j < p; j++) {
					T t = hypot(s[j], f);
					T cs = s[j] / t;
					T sn = f / t;
					s[j] = t;
					f = -sn * e[j];
					e[j] = cs * e[j];
					if (wantu) {
						for (int i = 0; i < m; i++) {
							t = cs * U[i][j] + sn * U[i][k - 1];
							U[i][k - 1] = -sn * U[i][j] + cs * U[i][k - 1];
							U[i][j] = t;
						}
					}
				}
			}
			break;

			// Perform one qr step.
			case 3: {
				// Calculate the shift.
				T scale = std::max({std::abs(s[p - 1]), std::abs(s[p - 2]), std::abs(e[p - 2]), std::abs(s[k]), std::abs(e[k])});
				T sp = s[p - 1] / scale;
				T spm1 = s[p - 2] / scale;
				T epm1 = e[p - 2] / scale;
				T sk = s[k] / scale;
				T ek = e[k] / scale;
				T b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
				T c = (sp * epm1) * (sp * epm1);
				T shift = (T) 0;
				if ((b != (T) 0) || (c != (T) 0)) {
					shift = std::sqrt(b * b + c);
					if (b < (T) 0) shift = -shift;
					shift = c / (b + shift);
				}
				T f = (sk + sp) * (sk - sp) + shift;
				T g = sk * ek;

				// Chase zeros.
				for (int j = k; j < p - 1; j++) {
					T t = hypot(f, g);
					T cs = f / t;
					T sn = g / t;
					if (j != k) e[j - 1] = t;
					f = cs * s[j] + sn * e[j];
					e[j] = cs * e[j] - sn * s[j];
					g = sn * s[j + 1];
					s[j + 1] = cs * s[j + 1];
					if (wantv) {
						for (int i = 0; i < n; i++) {
							t = cs * V[i][j] + sn * V[i][j + 1];
							V[i][j + 1] = -sn * V[i][j] + cs * V[i][j + 1];
							V[i][j] = t;
						}
					}
					t = hypot(f, g);
					cs = f / t;
					sn = g / t;
					s[j] = t;
					f = cs * e[j] + sn * s[j + 1];
					s[j + 1] = -sn * e[j] + cs * s[j + 1];
					g = sn * e[j + 1];
					e[j + 1] = cs * e[j + 1];
					if (wantu && (j < m - 1)) {
						for (int i = 0; i < m; i++) {
							t = cs * U[i][j] + sn * U[i][j + 1];
							U[i][j + 1] = -sn * U[i][j] + cs * U[i][j + 1];
							U[i][j] = t;
						}
					}
				}
				e[p - 2] = f;
				iter = iter + 1;
			}
			break;

			 // Convergence.
			case 4: {
				// Make the singular values positive.
				if (s[k] <= (T) 0) {
					s[k] = (s[k] < (T) 0 ? -s[k] : (T) 0);
					if (wantv) {
						for (int i = 0; i <= pp; i++) V[i][k] = -V[i][k];
					}
				}

				// Order the singular values.
				while (k < pp) {
					if (s[k] >= s[k + 1]) break;
					T t = s[k];
					s[k] = s[k + 1];
					s[k + 1] = t;
					if (wantv && (k < n - 1)) {
						for (int i = 0; i < n; i++) {
							t = V[i][k + 1]; V[i][k + 1] = V[i][k]; V[i][k] = t;
						}
					}
					if (wantu && (k < m - 1)) {
						for (int i = 0; i < m; i++) {
							t = U[i][k + 1]; U[i][k + 1] = U[i][k]; U[i][k] = t;
						}
					}
					k++;
				}
				iter = 0;
				p--;
			}
			break;

			}
		}
		this->dirty = false;
		return *this;
	}

	std::optional<Mat<T>> solve(const Mat<T>& b) override {
		if (dirty) compute();
		if (b.rows() != this->m) return std::nullopt;
		return inv().value().times(b);
	}

	// return U Matrix
	Mat<T> getU() {
		if (dirty) compute();
		return U.subMat(0, 0, this->m, this->n);
	}

	// return S Matrix
	Mat<T> getS() {
		if (dirty) compute();
		return Mat<T>::generate(this->n, this->n, [this] (size_t r, size_t c) {return r == c ? s[r] : (T) 0;});
	}

	// return V Matrix
	Mat<T> getV() {
		if (dirty) compute();
		return V.subMat(0, 0, this->n, this->n);
	}

	std::optional<Mat<T>> inv() override {
		if (dirty) compute();
		int m = (int) this->m;
		int n = (int) this->n;

		// sinv = 0 for tiny s, otherwise sinv = 1/s
		int u = std::min(m, n);
		for (int i = 0; i < u; i++) sinv[i] = std::abs(s[i]) < std::numeric_limits<T>::epsilon() ? (T) 0 : (T) 1 / s[i];

		// compute V * diag(sinv) * U'
		Mat<T> p = Mat<T>::zeros(n, m);
		for (int k = 0; k < m; k++) {
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < u; j++) {
					p[i][k] += U[k][j] * V[i][j] * sinv[j];
				}
			}
		}
		return p;
	}

	std::optional<Mat<T>> inv(Mat<T>& A) override {
		this->A = A;
		this->dirty = true;
		return inv();
	}

	T cond() {
		if (dirty) compute();
		return s[0] / s[std::min(this->m + 1, this->n) - 1]; // max(s) divided by min(s)
	}

};
