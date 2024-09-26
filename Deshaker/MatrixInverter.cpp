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

//Matd deltaSym(const Matd& A) {
//	return Matd::generate(A.rows(), A.cols(), [&] (size_t r, size_t c) {return A.at(r, c) - A.at(c, r); });
//}

PseudoInverter::PseudoInverter(Matd& A, size_t s) :
	MatInverter<double>(A),
	s { s },
	I { Matd::eye(s) },
	Xk { Matd::allocate(s, s) },
	Xk1 { Matd::allocate(s, s) },
	Yk { Matd::allocate(s, s) } {}

std::optional<Matd> PseudoInverter::inv() {
	int maxIter = 20;
	double eps = 0.05;
	double de = 1.0;
	double e = 0.0;

	Xk = Matd::eye(s).timesEach(1 / A.normF());

	int i = 0;
	for (; i < maxIter && de > eps; i++) {
		Yk = I.minus(A.timesSymmetric(Xk));
		Xk = Xk.times(I.plus(Yk.timesSymmetric(I.plus(Yk.timesSymmetric(I.plus(Yk))))));
		double e1 = Yk.normF2();
		de = std::abs(e - e1);
		e = e1;
		//Xk.toConsole("Xk", 8);
		//Yk.toConsole("Yk", 16);
		//std::cout << "------- i=" << i << " e=" << e << std::endl;
	}

	if (i < maxIter && std::isfinite(de)) return { Xk };
	else return std::nullopt;
}
