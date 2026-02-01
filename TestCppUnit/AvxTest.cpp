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
#include "AvxWrapper.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace AvxTest {

	TEST_CLASS(AvxTest) {

public:
	TEST_METHOD(avxRotateConstant) {
		V8d v = iotas.dx8;
		Assert::AreEqual({ 1, 2, 3, 4, 5, 6, 7, 0 }, v.rot<1>().vector());
		Assert::AreEqual({ 2, 3, 4, 5, 6, 7, 0, 1 }, v.rot<2>().vector());
		Assert::AreEqual({ 5, 6, 7, 0, 1, 2, 3, 4 }, v.rot<5>().vector());
		Assert::AreEqual(v.vector(), v.rot<0>().vector());
	}

	TEST_METHOD(avxRotate1) {
		V16f v = iotas.fx16;
		Assert::AreEqual({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }, v.rot(0).vector());
		Assert::AreEqual({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0 }, v.rot(1).vector());
		Assert::AreEqual({ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1 }, v.rot(2).vector());
		Assert::AreEqual({ 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4 }, v.rot(5).vector());
		Assert::AreEqual({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0 }, v.rot(17).vector());
		Assert::AreEqual({ 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 }, v.rot(-1).vector());
		Assert::AreEqual({ 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 }, v.rot(-2).vector());
		Assert::AreEqual({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }, v.rot(-16).vector());
		Assert::AreEqual(v.rot(-1).vector(), v.rot(-17).vector());
		Assert::AreEqual(v.rot<5>().vector(), v.rot(5).vector());
	}

	TEST_METHOD(avxRotate2) {
		V8d v = iotas.dx8;
		Assert::AreEqual({ 0, 1, 2, 3, 4, 5, 6, 7 }, v.rot(0).vector());
		Assert::AreEqual({ 1, 2, 3, 4, 5, 6, 7, 0 }, v.rot(1).vector());
		Assert::AreEqual({ 6, 7, 0, 1, 2, 3, 4, 5 }, v.rot(-2).vector());
	}

	TEST_METHOD(avxRotate3) {
		V4f v = iotas.fx16;
		Assert::AreEqual({ 0, 1, 2, 3 }, v.rot(0).vector());
		Assert::AreEqual({ 1, 2, 3, 0 }, v.rot(1).vector());
		Assert::AreEqual({ 2, 3, 0, 1 }, v.rot(2).vector());
		Assert::AreEqual({ 3, 0, 1, 2 }, v.rot(3).vector());
		Assert::AreEqual({ 0, 1, 2, 3 }, v.rot(4).vector());
		Assert::AreEqual({ 0, 1, 2, 3 }, v.rot(-4).vector());
	}

	TEST_METHOD(avxRotate4) {
		V8f v = iotas.fx16;
		Assert::AreEqual({ 1, 2, 3, 4, 5, 6, 7, 0 }, v.rot<1>().vector());
		Assert::AreEqual({ 2, 3, 4, 5, 6, 7, 0, 1 }, v.rot<2>().vector());
		Assert::AreEqual({ 5, 6, 7, 0, 1, 2, 3, 4 }, v.rot<5>().vector());
		Assert::AreEqual(v.vector(), v.rot<0>().vector());
	}

	};
}