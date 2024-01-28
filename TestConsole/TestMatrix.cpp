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
#include <iterator>

//specialized algorithms for symmetric Matrix
class MatSym : public Mat<double> {

private:
	MatSym(size_t rows, size_t cols) :
		Mat<double>(rows, cols) {}

public:
	MatSym(const Mat<double>& mat) : 
		Mat<double>(mat) {}

	MatSym(Mat<double>&& mat) noexcept :
		Mat<double>(std::move(mat)) {}

	static MatSym allocate(size_t rows, size_t cols) {
		assert(rows > 0 && cols > 0 && rows < MAX_DIM && cols < MAX_DIM && "invalid dimensions for allocation");
		return MatSym(rows, cols);
	}

	Mat<double> diag() const {
		return Mat<double>::generate(rows(), 1, [&] (size_t r, size_t c) { return at(r, r); });
	}

	//compute eigenvalues via repeated qr decomposition
	Mat<double> eigen() const {
		size_t s = rows();
		Mat<double> A = *this;
		//A.toConsole("A");
		Mat<double> I = Mat<double>::eye(s);
		Mat<double> q = Mat<double>::allocate(s, s);
		Mat<double> r = Mat<double>::allocate(s, s);

		QRDecompositor qrd(A);
		double norm = 2.0;
		int i = 0;
		while (norm > 1e-3 && i < 500) {
			qrd.compute();
			qrd.getQ(q);
			qrd.getR(r);
			//q.toConsole("Q");
			//r.toConsole("R");
			norm = q.abs().minus(I).normF();
			A = r.times(q);
			i++;
		}
		return MatSym(A).diag();
	}

	//compute 2-norm as the maximum eigenvalue
	double norm2() const {
		return eigen().at(0, 0);
	}

	//matrix multiplication between two symmetric matrices
	MatSym& times(const MatSym& b, MatSym& dest) const {
		for (size_t r = 0; r < dest.rows(); r++) {
			for (size_t c = r; c < dest.cols(); c++) {
				double sum = 0.0;
				for (size_t i = 0; i < dest.cols(); i++) {
					sum += at(r, i) * at(c, i) + at(c, i) * at(r, i);
				}
				sum /= 2.0;
				dest.at(r, c) = sum;
				dest.at(c, r) = sum;
			}
		}
	}

	MatSym times(const MatSym& b) const {
		MatSym out = MatSym::allocate(rows(), b.cols());
		return times(b, out);
	}
};

void eigen() {
	Matd M = Matd::fromBinaryFile("d:/VideoTest/s.dat");
	for (size_t i = 0; i < M.rows(); i += 6) {
		MatSym A = M.subMat(i, 0, 6, 6);
		Matd e = A.eigen();
	}
}

int s = 6;
Matd X1 = Matd::allocate(s, s);
Matd X2 = Matd::allocate(s, s);

struct Index {
	int r, c;
	double ival;
};

Index sidx[] = {
	{0,0,1.0}, {0,1}, {0,2}, {0,3}, {0,4}, {0,5},
		   {1,1,1.0}, {1,2}, {1,3}, {1,4}, {1,5},
				  {2,2,1.0}, {2,3}, {2,4}, {2,5},
						 {3,3,1.0}, {3,4}, {3,5},
								{4,4,1.0}, {4,5},
									   {5,5,1.0},
};

Matd pinvIter(const Matd& s) {
	double n = std::numeric_limits<double>::max();
	double beta = 0.1;
	double eps = 1e-4;
	Mat g = beta * s;
	Mat I = Matd::eye(6);
	for (int i = 0; i < 500 && n > eps * eps; i++) {
		//g = g.timesEach(1 + beta) - g.timesEach(beta).times(s).times(g);
		Matd G = g.timesEach(1 + beta) - g.timesEach(beta).times(s).times(g);
		//G.toConsole("check");


		for (int i = 0; i < 21; i++) {
			Index idx = sidx[i];
			double sum = 0.0;
			for (int k = 0; k < 6; k++) {
				sum += beta * g.at(idx.r, k) * s.at(idx.c, k);
			}
			X1.at(idx.r, idx.c) = sum;
			X1.at(idx.c, idx.r) = sum;
		}
		for (int i = 0; i < 21; i++) {
			Index idx = sidx[i];
			double sum = 0.0;
			for (int k = 0; k < 6; k++) {
				sum += X1.at(idx.r, k) * g.at(idx.c, k);
			}
			X2.at(idx.r, idx.c) = sum;
			X2.at(idx.c, idx.r) = sum;
		}
		for (int i = 0; i < 21; i++) {
			Index idx = sidx[i];
			double x = g.at(idx.r, idx.c) * (1 + beta) - X2.at(idx.r, idx.c);
			g.at(idx.r, idx.c) = x;
			g.at(idx.c, idx.r) = x;
		}


		//double nn = g.times(s).minus(I).norm2();
		n = 0.0;
		for (int i = 0; i < 21; i++) {
			Index idx = sidx[i];
			double sum = 0.0;
			for (int k = 0; k < 6; k++) {
				sum += g.at(idx.r, k) * s.at(idx.c, k);
			}
			double s = idx.ival == 1.0 ? sqr(sum - 1.0) : 2 * sqr(sum);
			n += s;
		}
		//n = std::sqrt(n);
		//std::cout << i << " " << n << " " << std::endl;
		std::cout << i << " " << n << " " << g.minus(G).normF() << std::endl;
	}
	//g.toConsole();
	//g.times(s).toConsole();
	return g;
}

void pinvTest() {
	Matd s = Matd::fromRows(6, 6, {
		0.1, 0.25, -0.6, 0.4, -0.25, 0.3,
		0.4, 0.75, -0.34, -0.15, -0.8, 0.45,
		0.0, 0.2, 0.5, -0.3, -0.45, 0.6,
		0.24, -0.67, 0.84, -0.23, -0.6, 0.75,
		0.5, 0.15, 0.84, -0.56, -0.72, -0.8,
		-0.7, -0.43, 0.75, 0.23, 0.4, -0.2
		});
	s = s.timesTransposed();
	Matd I = Matd::eye(6);

	//s = Matd::fromBinaryFile("f:/s.dat");

	for (int i = 0; i < 1; i++) {
		std::chrono::time_point t1 = std::chrono::system_clock::now();
		Matd A = pinvIter(s);
		//A.times(s).toConsole("I");
		std::chrono::time_point t2 = std::chrono::system_clock::now();
		double nrm = A.times(s).minus(I).normF();
		std::cout << "iter " << i << ", time [us] " << std::chrono::duration<double, std::micro>(t2 - t1).count() << ", nrm " << nrm << std::endl;
	}

	for (int i = 0; i < 10; i++) {
		std::chrono::time_point t1 = std::chrono::system_clock::now();
		Matd A = s.inv().value();
		std::chrono::time_point t2 = std::chrono::system_clock::now();
		double nrm = A.times(s).minus(I).normF();
		std::cout << "iter " << i << ", time [us] " << std::chrono::duration<double, std::micro>(t2 - t1).count() << ", nrm " << nrm << std::endl;
	}
}

void matTest() {
	std::cout << "----------------------------" << std::endl << "Mat Test:" << std::endl;
	Matd md = Matd::fromRows(2, 3, { 1, 2, 3, 4, 5, 6 }).toConsole();
	std::cout << "output by row: ";
	std::copy(md.begin(), md.end(), std::ostream_iterator<double>(std::cout, ", "));
	std::cout << std::endl;

	std::cout << "output by column: ";
	std::copy(md.begin(Matd::Direction::VERTICAL), md.end(Matd::Direction::VERTICAL), std::ostream_iterator<double>(std::cout, ", "));
	std::cout << std::endl;

	Matd out = Mat<double>::fromRows(4, 3, {
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

	Matd h1 = Matd::fromRows(2, 1, { 3, 5 });
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

	Matd a = Matd::fromRows(3, 3, { 12, 6, -4, -51, 167, 24, 4, -68, -41 }).toConsole("A");
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
	Matd A = Matd::fromRows(4, 3, { 1, 2, 3, 5, 6, 7, 10, 12, 8, 5, 2, 3 });
	Matd b = Matd::fromRow({ 5, 2, 4, 1 }).trans();
	Matd x = QRDecompositor(A).solve(b).value();
	x.toConsole();
}

void subMat() {
	Matd A = Matd::fromRows(3, 3, { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
	A.toConsole();
	SubMat<double> ss = SubMat<double>::from(A, 1, 1, 2, 2);
	ss.toConsole("ss");
	Matd s = A.subMat(1, 1, 2, 2);
	s.toConsole("s");
	std::cout << "equal: " << (s == ss) << std::endl;

	std::cout << "s[0][0] = " << ss[0][0] << std::endl;
}

void iteratorTest() {
	Matd A = Matd::fromRows(3, 3, { 1, 2, 3, 4, 5, 6, 7, 8, 9 });
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