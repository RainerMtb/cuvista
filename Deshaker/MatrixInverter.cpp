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

#include "MatrixInverter.hpp"


double IterativePseudoInverse1::normF2(size_t r, size_t c, std::function<double(size_t, size_t)> mat) {
	double sum = 0.0;
	for (size_t rr = 0; rr < r; rr++) {
		sum += sqr(mat(rr, rr));
		for (size_t cc = rr + 1; cc < c; cc++) {
			sum += sqr(mat(rr, cc)) * 2;
		}
	}
	return sum;
}

double IterativePseudoInverse1::normF2(Matd& mat) {
	return normF2(mat.rows(), mat.cols(), [&] (size_t r, size_t c) { return mat.at(r, c); });
}

void IterativePseudoInverse1::matMult(const Matd& A, std::function<double(size_t, size_t)> fcn, Matd& dest) {
	for (size_t r = 0; r < A.rows(); r++) {
		for (size_t c = 0; c < A.cols(); c++) {
			double sum = 0.0;
			for (size_t i = 0; i < A.cols(); i++) {
				sum += A.at(r, i) * fcn(i, c);
			}
			dest.at(r, c) = sum;
		}
	}
}

void IterativePseudoInverse1::matMult(const Matd& A, const Matd& B, Matd& dest) {
	matMult(A, [&] (size_t r, size_t c) { return B.at(r, c); }, dest);
}

void IterativePseudoInverse1::matMult(const Matd& A, double f, Matd& dest) {
	for (size_t r = 0; r < A.rows(); r++) {
		for (size_t c = r; c < A.cols(); c++) {
			dest.at(r, c) = dest.at(c, r) = A.at(r, c) * f;
		}
	}
}

void IterativePseudoInverse1::matAdd(const Matd& A, std::function<double(size_t, size_t)> fcn, Matd& dest) {
	for (size_t r = 0; r < A.rows(); r++) {
		for (size_t c = r; c < A.cols(); c++) {
			dest.at(r, c) = dest.at(c, r) = A.at(r, c) + fcn(r, c);
		}
	}
}

void IterativePseudoInverse1::matAdd(const Matd& A, const Matd& B, Matd& dest) {
	matAdd(A, [&] (size_t r, size_t c) { return B.at(r, c); }, dest);
}

std::optional<Matd> IterativePseudoInverse1::inv(Matd& A) {
	double n = normF2(A);
	double err = 1e-12;
	if (n < err) return std::nullopt;

	Xk.setDiag(1.0 / std::sqrt(n));
	int i = 0;
	while (n > err && i < 8) {
		matMult(A, Xk, X1);
		matMult(X1, [&] (size_t r, size_t c) { return 7.0 * X1.at(r, c) - 16.0 * I.at(r, c); }, X2);
		matAdd(X2, [&] (size_t r, size_t c) { return 11.0 * I.at(r, c); }, X2);
		matMult(X2, X1, X3);
		matMult(Xk, [&] (size_t r, size_t c) { return I.at(r, c) - X3.at(r, c) / 4; }, X4);
		matMult(X4, X2, Xk1);

		n = normF2(A.rows(), A.cols(), [&] (size_t r, size_t c) { return Xk.at(r, c) - Xk1.at(r, c); });
		std::swap(Xk.array, Xk1.array);
		i++;
	}

	return n > err ? std::nullopt : std::optional(Xk);
}

std::optional<Matd> IterativePseudoInverse1::inv() {
	return std::nullopt;
}


//------------------------------------------------------------------------

std::optional<Matd> IterativePseudoInverse2::inv() {
	return std::nullopt;
}

std::optional<Matd> IterativePseudoInverse2::inv(Matd& A) {
	double n = std::numeric_limits<double>::max();
	Mat X = beta * A;
	for (int i = 0; i < 500 && n > eps * eps; i++) {
		X = X.timesEach(1 + beta) - X.timesEach(beta).times(A).times(X);
		n = X.times(A).minus(I).normF();
	}
	return n > eps * eps ? std::nullopt : std::optional(X);
}