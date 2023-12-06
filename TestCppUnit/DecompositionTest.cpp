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

#include "pch.h"
#include "CppUnitTest.h"
#include "Mat.h"
#include "QRDecompositorUD.h"
#include "CholeskyDecompositor.h"
#include "Utils.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;

namespace MatrixDecompTest {

	TEST_CLASS(DecompositionTest) {

public:

	Matd A[9] = {
		Matd::fromRows(3, 3, {12, 6, -4, -51, 167, 24, 4, -68, -41}).trans(),
		Matd::fromRows(3, 3, {2.5, 0.7, -22.6, 17.3, -45, -9.9, 0.3456, -0.2345, 5.2345}),
		Matd::fromRows(3, 3, {12, 47, 18, 36, 4, -13, 0, 5, 28}),
		Matd::eye(3),
		Matd::eye(1),
		Matd::fromRow(0.5),
		Matd::eye(50),
		Matd::fromRows(3, 3, {2, -2, 7, 14, 9, -3, 7, -5, 8}),
		Matd::hilb(4)
	};
	Matd B[9] = {
		Matd::fromRow(2.0, -1.0, -4.0).trans(),
		Matd::fromRow(2.0, -1.0, -4.0).trans(),
		Matd::fromRow({2, -1, -4}).trans(),
		Matd::fromRow({1, 1, 1}).trans(),
		Matd::fromRow(1.0),
		Matd::fromRow(0.2),
		Matd::values(50, 1, 1.0),
		Matd::fromRow({2, -1, 4}).trans(),
		Matd::fromRow({1, 1, 2, 2}).trans()
	};
	Matd overA[5] = {
		Matd::fromRows(4, 3, {12, 47, 18, 36, 4, -13, 0, 5, 28, 34, -16, 9}),
		Matd::fromRows(5, 3, {12, 47, 18, 36, 4, -13, 0, 5, 28, 34, -16, 9, 2.3, -17.54, 38}),
		Matd::fromRows(6, 3, {12, 47, 18, 36, 4, -13, 0, 5, 28, 34, -16, 9, 2.3, -17.54, 38, 235, -853, 919}),
		Matd::fromRows(2, 3, {1, 2, 3, 4, 5, 6}).trans(),
		Matd::fromRow({1, 1, 1}).trans()
	};
	Matd overB[5] = {
		Matd::fromRow({2, -1, -4, 6}).trans(),
		Matd::fromRow({2, -1, -4, 6, -3}).trans(),
		Matd::fromRow({2, -1, -4, 6, -3, 12}).trans(),
		Matd::fromRow(2.0, -1.0, 4.0).trans(),
		Matd::fromRow(2.0, 2.0, 2.0).trans()
	};
	Matd underA[3] = {
		Matd::fromRows(4, 5, {12, 47, 18, 36, 4, -13, 0, 5, 28, 34, -16, 9, 2.3, -17.54, 38, 235, -853, 919, -35, 75}),
		Matd::fromRows(2, 3, {1, 2, 3, 4, 5, 6}),
		Matd::fromRow(1.0, 1.0, 1.0)
	};
	Matd underB[3] = {
		Matd::fromRow(2.0, 5.0, -7.0, 0.0).trans(),
		Matd::fromRows(3, 2, {2, -1, 0.5, 1.5, -7, 23}).trans(),
		Matd::fromRow(1.0)
	};

	TEST_METHOD(qr) {
		for (int i = 0; i < 9; i++) {
			Matd a = A[i];
			Matd in = a;
			QRDecompositor<double> qrdec(in);
			Matd Q = qrdec.getQ();
			Matd R = qrdec.getR();
			Assert::AreEqual(a, Q.times(R));
			Assert::AreEqual(Matd::eye(a.rows()), Q.times(Q.trans()));
			Matd b = B[i];
			Matd x = qrdec.solve(b).value();
			Assert::AreEqual(b, a.times(x));
		}
	}

	TEST_METHOD(qrSingular) {
		Matd m = Matd::fromRows(3, 3, {0, 1, 2, 3, 4, 5, 6, 7, 8}).trans();
		Matd b = Matd::values(m.cols(), 1, 1.0);
		//Assert::ExpectException<exception>([m, b] {QRDecompositor<double>(m).solve(b); });
		//Assert::ExpectException<exception>([m, b] {QRDecompositor<double>(m.trans()).solve(b); });
	}

	TEST_METHOD(qrRank) {
		Matd m = Matd::fromRows(3, 3, {2, 2, 2, 4, 4, 4, 8, 8, 8});
		Assert::IsFalse(QRDecompositor(m).isFullRank());
	}

	TEST_METHOD(qrOver) {
		for (int i = 0; i < 5; i++) {
			Matd A = overA[i];
			Matd b = overB[i];
			Matd in = A;
			QRDecompositor<double> qrdec(in);
			Matd Q = qrdec.getQ();
			Assert::AreEqual(Matd::eye(A.cols()), Q.trans().times(Q));	// I = Q' * Q
			Assert::AreEqual(A, Q.times(qrdec.getR()));					// A = Q * R
			auto x = qrdec.solve(b);
			Assert::IsTrue(x.has_value(), L"optional has no value");
		}
	}

	TEST_METHOD(qrUnder) {
		for (int i = 0; i < 3; i++) {
			Matd A = underA[i];
			QRDecompositorUD<double> qrud(A);
			Matd Q = qrud.getQ();
			Assert::AreEqual(Matd::eye(A.rows()), Q.trans().times(Q));	// I = Q' * Q
			Assert::IsTrue(A.trans().equals(Q.times(qrud.getR()), 1e-10)); 			// A' = Q * R because now A = A' for underdetermined
			Matd b = underB[i];
			auto x = qrud.solve(b);
			Assert::IsTrue(x.has_value(), L"optional has no value");
			Assert::AreEqual(b, A.times(*x));				// A * x = b
		}
	}

	TEST_METHOD(svd) {
		for (Matd m : A) {
			Matd in = m;
			SVDecompositor<double> svd(in);
			Mat U = svd.getU();
			Mat S = svd.getS();
			Mat V = svd.getV();
			Assert::AreEqual(m, U.times(S).times(V.trans())); // A == U * S * V'
			Assert::IsTrue(svd.pinv().times(m).equalsIdentity(1e-10));	// I == A * pinv(A)
		}
	}

	TEST_METHOD(lu) {
		for (Matd m : A) {
			Matd in = m;
			LUDecompositor<double> lu(in);
			Assert::AreEqual(lu.getL().times(lu.getU()), lu.getP().times(m)); // L * U == P * A
		}
	}

	TEST_METHOD(lu_1) {
		Matd a = Matd::fromRows(3, 3, { 12, 6, -4, -51, 167, 24, 4, -68, -41 });
		Matd in = a;
		LUDecompositor<double> lu(in);
		lu.compute();
		Assert::AreEqual(lu.getL().times(lu.getU()), lu.getP().times(a));
		Matd b = Matd::fromRow(2.0, -1.0, -4.0).trans();
		Matd x = lu.solve(b).value();
		Assert::AreEqual(a.times(x), b);
		Assert::AreEqual(a.inv().value().times(a), Matd::eye(a.rows()));
		Assert::AreEqual(a.times(a.solve(b).value()), b);
		Assert::AreEqual(a.solve(b).value(), x);
	}

	TEST_METHOD(lu_2) {
		std::vector<Matd> mats = {
			Matd::fromRows(3, 3, { 2, 8, 1, 4, 16, -1, -1, 2, 12 }),
			Matd::hilb(6),
			Matd::fromRows(3, 3, { 1, 0, 0, 4, 5, 6, -3, -2, -1 })
		};

		for (Matd m : mats) {
			Matd in = m;
			LUDecompositor lu(in);
			lu.compute();
			Mat inv = lu.inv().value();
			Mat checkIdentity = m.times(inv);
			double maxVal = (checkIdentity - Matd::eye(m.rows())).abs().max();
			Assert::IsTrue(maxVal < 1e-9);
		}
	}

	TEST_METHOD(cholesky) {
		for (int i = 0; i < 30; i++) {
			Matd a = Matd::rand(5, 5, 0, 1, 1000);
			a = a.times(a.trans()); // symmetric and positive definit
			CholeskyDecompositor<double> dec(a);
			Matd L = dec.getL();
			Assert::AreEqual(a, L.times(L.trans()));
			Matd b = Matd::rand(5, 1, 0, 1, 1000);
			Matd x = dec.solve(b).value();
			Assert::IsTrue(b.equals(a.times(x), 1e-8));
		}
	}
	};
}