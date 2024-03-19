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

#include "TestMain.hpp"
#include "SubMat.h"
#include "MatrixInverter.hpp"

void pinvTest() {
	size_t s = 6;
	Matd I = Matd::eye(s);
	Matd N = Matd::values(s, s, nan(""));
	Matd M = Matd::fromBinaryFile("d:/VideoTest/s.dat");
	IterativePseudoInverse1 ip(s);
	int count = 0;

	for (size_t i = 0; i < M.rows(); i += s) {
		size_t idx = i / s;
		Matd A = M.subMat(i, 0, s, s);

		{
			Matd X;
			std::chrono::time_point t1 = std::chrono::system_clock::now();
			auto result = ip.inv(A);
			std::chrono::time_point t2 = std::chrono::system_clock::now();
			std::printf("iter %04zd", idx);
			if (result.has_value()) {
				X = result.value();
				double nrm = X.times(A).minus(I).normF();
				std::printf(" time [us] %.0f nrm %.4e", std::chrono::duration<double, std::micro>(t2 - t1).count(), nrm);
				count++;
			}
		}
		{
			std::chrono::time_point t1 = std::chrono::system_clock::now();
			Matd X = A.inv().value();
			std::chrono::time_point t2 = std::chrono::system_clock::now();
			double nrm = X.times(A).minus(I).normF();
			std::printf(", classic: time [us] %.0f nrm %.4e\n", std::chrono::duration<double, std::micro>(t2 - t1).count(), nrm);
		}
	}
	double valid = 100.0 * count / (M.rows() / 6);
	std::printf("\nvalid results %d of %zd = %.1f%%\n", count, M.rows()/6, valid);
}

void matTest() {
	std::cout << "----------------------------" << std::endl << "Mat Test:" << std::endl;
	Matd md = Matd::fromRowData(2, 3, { 1, 2, 3, 4, 5, 6 }).toConsole();
	std::cout << "output by row: ";
	std::copy(md.begin(), md.end(), std::ostream_iterator<double>(std::cout, ", "));
	std::cout << std::endl;

	std::cout << "output by column: ";
	std::copy(md.begin(Matd::Direction::VERTICAL), md.end(Matd::Direction::VERTICAL), std::ostream_iterator<double>(std::cout, ", "));
	std::cout << std::endl;

	Matd out = Mat<double>::fromRowData(4, 3, {
		2.0,		1.045e-5,		2056.456,
		20.56e9,	0,				1.56472e-200,
		-13.456,	-0.1 + 0.101 - 0.001, -42.678,
		-255,		5.7564523,		1 + 1e-9
		}).trans();
	out.toConsole("out");
	std::cout << "row output: " << out[0] << std::endl;

	out.trans().toConsole("out trans");

	Matd c1 = Matd::rand(2, 3).toConsole("c1");
	Matd c2 = Matd::fromRow({ 15, 15, -15 }).toConsole("c2");
	Matd c3 = Matd::hilb(3).toConsole("c3");
	Matd m = Matd::concat(Matd::Direction::VERTICAL, { c1, c2, c3 });
	m.toConsole("vertcat");
	m.subMat(1, 1, 2, 2).toConsole("subMat");

	Matd h1 = Matd::fromRowData(2, 1, { 3, 5 });
	Matd h2 = Matd::values(2, 2, -3.6);
	Matd hm = Matd::concatHorz({ h1, h2 }).toConsole("horzcat");
	std::cout << std::endl;

	Matd::zeros(2, 3).plus(4).timesEach(2.5).toConsole("unaryTest 10");
	(Matd::values(4, 4, 1.0) + 2 * Matd::eye(4) * -3.567).toConsole("diag -6,134");

	Matd h = Matd::hilb(3).repeat(2, 3).toConsole("hilb repeat");
	Matd hs = h.subMat(2, 3, 2, 5).toConsole("hilb subMat");
	std::cout << "is shared (no): " << h.isShared(hs) << " and " << hs.isShared(h) << std::endl;

	SubMat<double> hss = SubMat<double>::from(h, 2, 3, 2, 5);
	hss.toConsole("subMatShared");
	std::cout << "sub == subshared (yes): " << (hs == hss) << std::endl;
	std::cout << "is shared (yes): " << h.isShared(hss) << " and " << hss.isShared(h) << std::endl;

	Matd b = Matd::fromRow(2.0, -1.0, -4.0);
	b.toConsole("b");
	(b - 2).toConsole("b-2");
	(3 - b).toConsole("3-b");
	(b / 2.5).toConsole("b / 2.5");
	(-1 / b).toConsole("-1 / b");
	std::cout << std::endl;

	Matd a = Matd::fromRowData(3, 3, { 12, 6, -4, -51, 167, 24, 4, -68, -41 }).toConsole("A");
	std::cout << "ptr[0][0] = " << a[0][0] << std::endl;
	std::cout << "ptr[1][1] = " << a[1][1] << std::endl;
	a[0][2] += 4;
	a.toConsole("A[0][2] += 4");

	//move assignment
	a = 2 * Matd::eye(4) * -3.567;
	a.toConsole("a");

	//rand
	Matd r = Matd::rand(4, 6);
	r.toConsole("rand 1", 1);
	r.toConsole("rand 8", 8);
}

void qrdec() {
	Matd A = Matd::fromRowData(4, 3, { 1, 2, 3, 5, 6, 7, 10, 12, 8, 5, 2, 3 });
	A.toConsole("A");
	Matd b = Matd::fromRow({ 5, 2, 4, 1 }).trans().toConsole("b");
	Matd input = A;
	QRDecompositor qr = QRDecompositor(input);
	Matd x = qr.solve(b).value();
	x.toConsole("x");
	Matd Q = qr.getQ().toConsole("Q");
	Matd R = qr.getR().toConsole("R");
	Q.times(R).toConsole("Q*R=A");
}

void subMat() {
	Matd A = Matd::fromRowData(3, 3, { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
	A.toConsole();
	SubMat<double> ss = SubMat<double>::from(A, 1, 1, 2, 2);
	ss.toConsole("ss");
	Matd s = A.subMat(1, 1, 2, 2);
	s.toConsole("s");
	std::cout << "equal: " << (s == ss) << std::endl;

	std::cout << "s[0][0] = " << ss[0][0] << std::endl;
}

void iteratorTest() {
	Matd A = Matd::fromRowData(3, 3, { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
	SubMat<double> ss = SubMat<double>::from(A, 1, 1, 2, 2);
	ss.toConsole("ss");
	Matd s = A.subMat(1, 1, 2, 2);

	std::cout << "iter row: ";
	for (double& d : s) std::cout << d << ", ";
	std::cout << std::endl;
	std::cout << "iter row: ";
	for (double& d : ss) std::cout << d << ", ";
	std::cout << std::endl;

	std::cout << "iter cols: ";
	for (auto it = s.begin(Matd::Direction::VERTICAL); it != s.end(Matd::Direction::VERTICAL); it++) std::cout << *it << ", ";
	std::cout << std::endl;
	std::cout << "iter cols: ";
	for (auto it = ss.begin(Matd::Direction::VERTICAL); it != ss.end(Matd::Direction::VERTICAL); it++) std::cout << *it << ", ";
	std::cout << std::endl;
}