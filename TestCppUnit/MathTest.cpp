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

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace MathTest {

	TEST_CLASS(MathTest) {

private:
	bool checkTransform(Affine2D& tf, PointResult& pr) {
		auto p = tf.transform(pr.x, pr.y);
		return std::abs(p.first - pr.x - pr.u) < 1e-10 && std::abs(p.second - pr.y - pr.v) < 1e-10;
	}

public:

	TEST_METHOD(affine) {
		std::vector<PointResult> pr {
			{ 0, 0, 0, 0, 0, 832, -232, -25.7400,  2.5287, PointResultType::SUCCESS_ABSOLUTE_ERR },
			{ 1, 0, 0, 0, 0, 64,   -40, -24.2886,  0.0104, PointResultType::SUCCESS_ABSOLUTE_ERR },
			{ 2, 0, 0, 0, 0, 176,  104, -23.5065, -3.3990, PointResultType::SUCCESS_ABSOLUTE_ERR }
		};

		AffineTransform tf;
		bool hasValue = tf.computeAffine(pr.begin(), 3);
		Assert::IsTrue(hasValue);
		Assert::IsTrue(checkTransform(tf, pr[0]));
		Assert::IsTrue(checkTransform(tf, pr[1]));
		Assert::IsTrue(checkTransform(tf, pr[2]));
	}

	TEST_METHOD(transformDirect) {
		//point sets to test
		std::vector<std::vector<PointResult>> pointSets = {
			{
			{ 0, 0, 0, 0, 0, -4,  2, 0.5, 0.4 },
			{ 1, 0, 0, 0, 0,  1, -4, 0.5, 0.4 },
			{ 2, 0, 0, 0, 0,  3,  4, 0.5, 0.4 },
			{ 1, 0, 0, 0, 0,  2,  2, 0.5, 0.4 },
			},
			{
			{ 0, 0, 0, 0, 0, 3, 3, 0.75, 0.3 },
			{ 1, 0, 0, 0, 0, 4, 3, 0.75, 0.3 },
            },
			{
			{ 0, 0, 0, 0, 0, -1, -1, -0.5,  0.5 },
			{ 1, 0, 0, 0, 0,  1,  1,  0.5, -0.5 },
			},
			{
			{ 0, 0, 0, 0, 0, -2,  1, 1.5, -2.3 },
			{ 1, 0, 0, 0, 0, -2,  3, 1.5, -2.3 },
			{ 2, 0, 0, 0, 0, -2,  7, 1.5, -2.3 },
			{ 1, 0, 0, 0, 0, -2, -4, 1.5, -2.3 },
			},
			{
			{ 0, 0, 0, 0, 0, -2,  1, 15, -23 },
			{ 1, 0, 0, 0, 0, -2,  3, 15, -23 },
			},
		};

		//expected results
		std::vector<Affine2D> resultSet = {
			Affine2D::fromValues(1.0, 0.0, 0.5, 0.4),
			Affine2D::fromValues(1.0, 0.0, 0.75, 0.3),
			Affine2D::fromValues(1.0, 0.5, 0.0, 0.0),
			Affine2D::fromValues(1.0, 0.0, 1.5, -2.3),
			Affine2D::fromValues(1.0, 0.0, 15, -23),
		};

		//checks
		for (int i = 0; i < pointSets.size() && i < resultSet.size(); i++) {
			std::wstring str = L"check " + std::to_wstring(i);
			auto points = pointSets[i];
			AffineSolverSimple trf1(points.size());
			trf1.computeSimilar(points.begin(), points.size());
			Assert::IsTrue(resultSet[i].equals(trf1, 1e-14), str.c_str());

			ThreadPool pool(2);
			AffineSolverFast trf2(pool, points.size());
			trf2.computeSimilar(points.begin(), points.size());
			Assert::IsTrue(resultSet[i].equals(trf2, 1e-14), str.c_str());
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
					int x = distrib(gen);
					int y = distrib(gen);
					double u = distrib(gen) / 50.0;
					double v = distrib(gen) / 50.0;
					points[i] = { i, 0, 0, 0, 0, x, y, u, v };
				}

				//compute transforms
				AffineSolverSimple trf1(points.size());
				trf1.computeSimilar(points.begin(), points.size());

				ThreadPool pool(2);
				AffineSolverFast trf2(pool, points.size());
				trf2.computeSimilar(points.begin(), points.size());

				Assert::IsTrue(trf1.equals(trf2, 1e-12));
			}
		}
	}

	};
}