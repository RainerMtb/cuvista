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
#include "Utils.hpp"
#include "AffineTransform.hpp"
#include "SubMat.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace MatrixTest {

	TEST_CLASS(MatrixTest) {


	void log(Matd& out) {
		Logger::WriteMessage(out.toString().c_str());
	}

public:
	TEST_METHOD(times) {
		Matd a = Matd::fromRows(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
		Matd result = Matd::fromRows(3, 3, {14, 32, 50, 32, 77, 122, 50, 122, 194});
		Assert::AreEqual(result, a.times(a.trans()));
		Assert::AreEqual(result, a.timesTransposed());

		Matd b = Matd::fromRows(5, 5, {2, 2, 2, 2, 2, -2, 1, 2, 3, -2, -2, 4, 5, 6, -2, -2, 7, 8, 9, -2, 2, 2, 2, 2, 2});
		SubMat<double> sb = SubMat<double>::from(b, 1, 1, 3, 3);
		Assert::AreEqual(result, sb.times(sb.trans()));
		Assert::AreEqual(result, sb.timesTransposed());
	}

	TEST_METHOD(output) {
		std::string str = Matd::eye(2).toString();
		std::string expect = "[[   1.0   0.0   ]\n [   0.0   1.0   ]]\n";
		Assert::AreEqual(expect, Matd::eye(2).toString());
		Assert::AreEqual(toWString(expect), Matd::eye(2).toWString());
	}

	TEST_METHOD(basic) {
		Matd a = Matd::fromRows(3, 3, { 12, 6, -4, -51, 167, 24, 4, -68, -41 });
		Assert::AreEqual(a, a);
		Assert::IsTrue(a == a);
		Assert::AreEqual(167.0, a.max());
		Assert::AreEqual(-68.0, a.min());
		Assert::AreEqual(12.0, a.at(0, 0));
		Assert::AreEqual(-4.0, a.at(0, 2));

		Mat<int> ms = Mat<int>::generate(3, 3, [a] (size_t r, size_t c) { return (int) a.at(r, c); });
		Assert::IsTrue(a == ms);

		a = (Matd::values(4, 4, 1.0) + 2 * Matd::eye(4) * -3.567);
		Assert::AreEqual(-6.134, a.at(0, 0));
		Assert::AreEqual(-6.134, a.at(1, 1));

		a = Matd::eye(4).timesEach(3.5);
		Assert::AreEqual(14.0, a.sum());

		a += a;
		Assert::AreEqual(28.0, a.sum());

		Assert::IsTrue(a.timesTransposed().isSymmetric());

		Matd m = 4;
		Assert::AreEqual(4.0, m.at(0, 0));
	}

	TEST_METHOD(compare) {
		Matd a = Matd::fromRow({ 1.5 });
		Matd b = Matd::fromRow({ 2.5 });
		Assert::IsTrue(a < b);
		Assert::IsFalse(b < a);

		Matd aa = Matd::values(2, 3, 1.5);
		Matd bb = Matd::values(2, 3, 1.5);
		Assert::IsTrue(aa == bb);
	}

	TEST_METHOD(swap) {
		Matd a = Matd::fromRows(2, 3, { 1, 2, 3, 4, 5, 6 });
		Matd b = Matd::fromRows(1, 4, { -1, -2, -3, -4 });
		Matd s1 = a, s2 = b;
		std::swap(s1, s2);
		Assert::AreEqual(a, s2);
		Assert::AreEqual(b, s1);
	}

	TEST_METHOD(qr) {
		Matd A[] = {
				Matd::fromRows(3, 3, {12, 6, -4, -51, 167, 24, 4, -68, -41}).trans(),
				Matd::fromRows(3, 3, {2.5, 0.7, -22.6, 17.3, -45, -9.9, 0.3456, -0.2345, 5.2345}),
				Matd::fromRows(3, 3, {12, 47, 18, 36, 4, -13, 0, 5, 28}),
				Matd::eye(3),
				Matd::eye(1),
				Matd::fromRow({0.5}),
				Matd::eye(50),
				Matd::fromRows(3, 3, {2, -2, 7, 14, 9, -3, 7, -5, 8}),
				Matd::hilb(4)
		};
		Matd B[] = {
				Matd::fromRow({2, -1, -4}).trans(),
				Matd::fromRow({2, -1, -4}).trans(),
				Matd::fromRow({2, -1, -4}).trans(),
				Matd::fromRow({1, 1, 1}).trans(),
				Matd::fromRow({1}).trans(),
				Matd::fromRow({0.2}).trans(),
				Matd::values(50, 1, 1.0),
				Matd::fromRow({2, -1, 4}).trans(),
				Matd::fromRow({1, 1, 2, 2}).trans()
		};

		for (int i = 0; i < 9; i++) {
			Matd in = A[i];
			QRDecompositor<double> qrdec(in);
			Matd Q = qrdec.getQ();
			Matd R = qrdec.getR();
			Assert::AreEqual(A[i], Q.times(R));
			Assert::AreEqual(Matd::eye(A[i].rows()), Q.times(Q.trans()));
			Matd x = qrdec.solve(B[i]).value();
			Assert::AreEqual(B[i], A[i].times(x));
		}
	}

	TEST_METHOD(svd) {
		Matd s = Matd::fromRows(3, 3, { 2, 3, 4, -4, 2.5, -20, 0, 3, 6 });
		Matd sp = s.pinv().value().pinv().value();
		Assert::AreEqual(s, sp);

		s = Matd::fromRows(7, 7, {
			0.56115, 0.06098, 0.09795, 0.69631, 0.46300, 0.94497, 0.41785,
			0.73846, 0.55851, 0.81792, 0.40089, 0.46976, 0.17597, 0.46457,
			0.85967, 0.49835, 0.29663, 0.84098, 0.23546, 0.39177, 0.94879,
			0.81144, 0.04844, 0.30349, 0.55241, 0.40108, 0.94707, 0.87954,
			0.60899, 0.35013, 0.26136, 0.72672, 0.55628, 0.84193, 0.42204,
			0.10501, 0.06406, 0.04323, 0.76693, 0.32525, 0.45771, 0.28060,
			0.94509, 0.77685, 0.08106, 0.10140, 0.03811, 0.97761, 0.90337
			});
		Matd in = s;
		SVDecompositor<double> svd(in);
		Matd si = svd.pinv().pinv().value();
		Assert::IsTrue(s.equals(si, 1e-10));
		Assert::IsTrue(svd.cond() < 15000);
	}

	TEST_METHOD(subMat) {
		Matd a = Matd::eye(7);
		Matd sm = a.subMat(1, 2, 4, 3);
		SubMat<double> sms = SubMat<double>::from(a, 1, 2, 4, 3);
		Assert::IsTrue(sms == sm, L"subMat not equal");
		Assert::AreEqual(4ull, sm.rows());
		Assert::AreEqual(3ull, sm.cols());
		Assert::AreEqual(4ull, sms.rows());
		Assert::AreEqual(3ull, sms.cols());
		Assert::IsTrue(a.isShared(sms));
		Assert::IsTrue(sms.isShared(a));
		Assert::IsFalse(sm.isShared(a));
		Assert::IsFalse(sm.isShared(sms));

		sms.setValues(10);
		Assert::AreEqual(10.0, a.at(1, 2));
		Assert::AreEqual(10.0, a.at(2, 3));

		sms.setValues([] (size_t r, size_t c) {return r * 2.0 + c * 4.0 + 2.0; });

		SubMat<double> sub2 = SubMat<double>::from(sms, 1, 0, 2, 1);
		sub2.set(1, 0, 100);
		Assert::AreEqual(4.0, a.at(2, 2));
		Assert::AreEqual(100.0, a.at(3, 2));
		Assert::AreEqual(4.0, sub2[0][0]);
		Assert::AreEqual(100.0, sub2[1][0]);
		Assert::AreEqual(100.0, sub2.at(1, 0));

		Assert::AreEqual(206.0, a.sum());
		Assert::AreEqual(202.0, sms.sum());
		Assert::AreEqual(104.0, sub2.sum());

		Matd& subMatRef = sub2;
		Assert::AreEqual(subMatRef[0][0], sub2[0][0]);
		Assert::AreEqual(subMatRef[1][0], sub2[1][0]);
	}

	TEST_METHOD(constMat) {
		const Matf f = Matf::fromRows(2, 3, { 1, 2, 3, 4, 5, 6 });
		Assert::AreEqual(5.0f, f[1][1]);
		auto r = f[0];
		Assert::AreEqual(2.0f, r[1]);
		Matf g;
		g = f;
		Assert::AreEqual(f, g);
	}

	TEST_METHOD(interpolate) {
		Matd a = Matd::hilb(5);
		double tol = 1e-15;

		Assert::AreEqual(7.0 / 12.0, a.interp2(0.5, 0.5).value(), tol);
		Assert::AreEqual(49.0 / 240.0, a.interp2(1.5, 2.5).value(), tol);
		Assert::AreEqual(1.0, a.interp2(0.0, 0.0).value(), tol);
		Assert::AreEqual(1.0 / 9.0, a.interp2(4.0, 4.0).value(), tol);
		Assert::AreEqual(95.0 / 100.0, a.interp2(0.1, 0.0).value(), tol);
		Assert::AreEqual(95.0 / 100.0, a.interp2(0.0, 0.1).value(), tol);
	}

	TEST_METHOD(iterate) {
		Matd m = Matd::fromRow({ 1, 2, 3, 4 });
		auto iter = m.cbegin();
		Assert::AreEqual(1.0, *iter);
		iter++;
		Assert::AreEqual(2.0, *iter);

		auto i2 = m.cend();
		i2--;
		Assert::AreEqual(4.0, *i2);
	}

	};
}
