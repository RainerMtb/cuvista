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

#include "MatInverter.hpp"
#include "Mat.hpp"

class IterativePseudoInverse1 : public MatInverter<double> {

private:
	Matd X1, X2, X3, X4, Xk, Xk1, I;

	double normF2(size_t r, size_t c, std::function<double(size_t, size_t)> mat);
	double normF2(Matd& mat);
	void matMult(const Matd& A, std::function<double(size_t, size_t)> fcn, Matd& dest);
	void matMult(const Matd& A, const Matd& B, Matd& dest);
	void matMult(const Matd& A, double f, Matd& dest);
	void matAdd(const Matd& A, std::function<double(size_t, size_t)> fcn, Matd& dest);
	void matAdd(const Matd& A, const Matd& B, Matd& dest);

public:
	IterativePseudoInverse1(size_t s) :
		X1 { Matd::allocate(s, s) },
		X2 { Matd::allocate(s, s) },
		X3 { Matd::allocate(s, s) },
		X4 { Matd::allocate(s, s) },
		Xk { Matd::allocate(s, s) },
		Xk1 { Matd::allocate(s, s) },
		I { Matd::eye(s) } {}

	std::optional<Matd> inv(Matd& A) override;
};


class IterativePseudoInverse2 : public MatInverter<double> {

private:
	Matd X1, X2, I;

	double beta = 0.1;
	double eps = 1e-4;

public:
	IterativePseudoInverse2(size_t s) :
		X1 { Matd::allocate(s, s) },
		X2 { Matd::allocate(s, s) },
		I { Matd::eye(s) } {}

	std::optional<Matd> inv(Matd& A) override;
};