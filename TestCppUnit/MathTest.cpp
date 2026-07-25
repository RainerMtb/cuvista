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
#include "ThreadPool.hpp"
#include "AffineTransform.hpp"
#include "AvxWrapper.hpp"
#include "Avx2Wrapper.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace MathTest {

	TEST_CLASS(MathTest) {

private:
	bool testTransform(Affine2D& tf, PointResult& pr) {
		auto p = tf.transform(pr.x, pr.y);
		return std::abs(p.x - pr.x - pr.u) < 1e-10 && std::abs(p.y - pr.y - pr.v) < 1e-10;
	}

public:

	TEST_METHOD(transformSimilar) {
		std::vector<PointResult> pr {
			{ 0, 0, 0, 832.0, -232.0, -25.7400,  2.5287, PointResultType::SUCCESS_ABSOLUTE_ERR },
			{ 1, 0, 0,  64.0,  -40.0, -24.2886,  0.0104, PointResultType::SUCCESS_ABSOLUTE_ERR }
		};

		AffineTransform at;
		at.computeSimilarDirect(pr[0], pr[1]);
		testTransform(at, pr[0]);
		testTransform(at, pr[1]);
	}

	TEST_METHOD(transformAffine) {
		std::vector<PointResult> pr {
			{ 0, 0, 0, 832.0, -232.0, -25.7400,  2.5287, PointResultType::SUCCESS_ABSOLUTE_ERR },
			{ 1, 0, 0,  64.0,  -40.0, -24.2886,  0.0104, PointResultType::SUCCESS_ABSOLUTE_ERR },
			{ 2, 0, 0, 176.0,  104.0, -23.5065, -3.3990, PointResultType::SUCCESS_ABSOLUTE_ERR }
		};

		AffineTransform tf;
		bool hasValue = tf.computeAffine(pr);
		Assert::IsTrue(hasValue);
		Assert::IsTrue(testTransform(tf, pr[0]));
		Assert::IsTrue(testTransform(tf, pr[1]));
		Assert::IsTrue(testTransform(tf, pr[2]));
	}

	TEST_METHOD(transformDirect) {
		//point sets to test
		std::vector<std::vector<PointResult>> pointSets = {
			{
			{ 0, 0, 0, -4,  2, 0.5, 0.4 },
			{ 1, 0, 0,  1, -4, 0.5, 0.4 },
			{ 2, 0, 0,  3,  4, 0.5, 0.4 },
			{ 3, 0, 0,  2,  2, 0.5, 0.4 },
			},
			{
			{ 0, 0, 0, 3, 3, 0.75, 0.3 },
			{ 1, 0, 0, 4, 3, 0.75, 0.3 },
            },
			{
			{ 0, 0, 0, -1, -1, -0.5,  0.5 },
			{ 1, 0, 0,  1,  1,  0.5, -0.5 },
			},
			{
			{ 0, 0, 0, -2,  1, 1.5, -2.3 },
			{ 1, 0, 0, -2,  3, 1.5, -2.3 },
			{ 2, 0, 0, -2,  7, 1.5, -2.3 },
			{ 3, 0, 0, -2, -4, 1.5, -2.3 },
			},
			{
			{ 0, 0, 0, -2,  1, 15, -23 },
			{ 1, 0, 0, -2,  3, 15, -23 },
			},
		};

		//expected results
		std::vector<Affine2D> resultSet = {
			Affine2D::fromParam(1.0, 0.0, 0.5, 0.4),
			Affine2D::fromParam(1.0, 0.0, 0.75, 0.3),
			Affine2D::fromParam(1.0, 0.5, 0.0, 0.0),
			Affine2D::fromParam(1.0, 0.0, 1.5, -2.3),
			Affine2D::fromParam(1.0, 0.0, 15, -23),
		};

		//checks
		for (int i = 0; i < pointSets.size() && i < resultSet.size(); i++) {
			auto points = pointSets[i];
			{
				AffineSolver as;
				as.setSolver(std::make_shared<AffineSolverSimple>(points.size()));
				const AffineTransform& trf = as.computeSimilar(points);
				std::wstring str = L"check " + std::to_wstring(i) + L" simple";
				Assert::IsTrue(resultSet[i].equals(trf, 1e-12), str.c_str());
			}

			{
				ThreadPool pool(4);
				AffineSolver as;
				as.setSolver(std::make_shared<AffineSolverFast>(pool, points.size()));
				const AffineTransform& trf = as.computeSimilar(points);
				as.computeSimilar(points);
				std::wstring str = L"check " + std::to_wstring(i) + L" fast";
				Assert::IsTrue(resultSet[i].equals(trf, 1e-12), str.c_str());
			}

			{
				ThreadPool pool(4);
				AffineSolver as;
				as.setSolver(std::make_shared<AffineSolverAvx512>(pool, points.size()));
				const AffineTransform& trf = as.computeSimilar(points);
				as.computeSimilar(points);
				std::wstring str = L"check " + std::to_wstring(i) + L" avx512";
				Assert::IsTrue(resultSet[i].equals(trf, 1e-12), str.c_str());
			}

			{
				ThreadPool pool(4);
				AffineSolver as;
				as.setSolver(std::make_shared<AffineSolverAvx2>(pool, points.size()));
				const AffineTransform& trf = as.computeSimilar(points);
				std::wstring str = L"check " + std::to_wstring(i) + L" avx2";
				Assert::IsTrue(resultSet[i].equals(trf, 1e-12), str.c_str());
			}
		}
	}

	TEST_METHOD(transformDirectRandom) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> distrib(-50, 50);

		std::vector<int> sizes = { 2, 4, 6, 20, 100 };
		for (int siz : sizes) {
			for (int k = 0; k < 100; k++) {
				//generate points
				std::vector<PointResult> points(siz);
				for (int i = 0; i < siz; i++) {
					double x = distrib(gen);
					double y = distrib(gen);
					double u = distrib(gen) / 50.0;
					double v = distrib(gen) / 50.0;
					points[i] = { i, 0, 0, x, y, u, v };
				}

				//compute transforms
				AffineSolver as1;
				as1.setSolver(std::make_shared<AffineSolverSimple>(points.size()));
				const AffineTransform& trf1 = as1.computeSimilar(points);

				ThreadPool pool(2);
				AffineSolver as2;
				as2.setSolver(std::make_shared<AffineSolverFast>(pool, points.size()));
				const AffineTransform& trf2 = as2.computeSimilar(points);

				Assert::IsTrue(trf1.equals(trf2, 1e-12));
			}
		}
	}

	TEST_METHOD(transformExpected) {
		PointResult p1 = { 0, 0, 0, 4, 5, 1.0, 0.5 };
		PointResult p2 = { 1, 0, 0, 6, 10, 1.0, 0.5 };
		PointResult p3 = { 2, 0, 0, 8, 14, 1.0, 0.5 };

		AffineSolver as;
		as.setSolver(std::make_shared<AffineSolverSimple>(8));
		AffineTransform trf;

		trf = as.computeSimilarDirect(p1, p2);
		Assert::AreEqual(1.0, trf.scale(), 1e-14);
		Assert::AreEqual(0.0, trf.rot(), 1e-14);
		Assert::AreEqual(1.0, trf.dX(), 1e-14);
		Assert::AreEqual(0.5, trf.dY(), 1e-14);

		std::vector p = { p1, p2 };
		trf = as.computeSimilar(p);
		Assert::AreEqual(1.0, trf.scale(), 1e-14);
		Assert::AreEqual(0.0, trf.rot(), 1e-14);
		Assert::AreEqual(1.0, trf.dX(), 1e-14);
		Assert::AreEqual(0.5, trf.dY(), 1e-14);

		trf = as.computeAffineDirect(p1, p2, p3);
		Assert::AreEqual(1.0, trf.scale(), 1e-14);
		Assert::AreEqual(0.0, trf.rot(), 1e-14);
		Assert::AreEqual(1.0, trf.dX(), 1e-14);
		Assert::AreEqual(0.5, trf.dY(), 1e-14);

		std::vector pp = { p1, p2, p3 };
		as.computeAffine(pp);
		Assert::AreEqual(1.0, trf.scale(), 1e-14);
		Assert::AreEqual(0.0, trf.rot(), 1e-14);
		Assert::AreEqual(1.0, trf.dX(), 1e-14);
		Assert::AreEqual(0.5, trf.dY(), 1e-14);
	}

	};
}