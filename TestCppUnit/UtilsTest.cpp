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
#include "Image.hpp"
#include "Utils.hpp"
#include "Util.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace MiscTest {

	TEST_CLASS(UtilsTest) {

public:

	TEST_METHOD(convertNV12) {
		int w = 400;
		int h = 200;
		int stride = w + 12;
		ImageYuv yuv(h, w, stride);
		yuv.setColor(Color::yuv(50, 10, 20));
		ImageNV12 nv12(h, w, stride);
		yuv.toNV12(nv12);

		ImageYuv dest(h, w, stride);
		nv12.toYuv(dest);

		Assert::AreEqual(yuv, dest, L"different images");
	}

	TEST_METHOD(base64_test1) {
		std::vector<unsigned char> data = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
		std::string str = util::base64_encode(data);
		std::vector<unsigned char> dataCheck = util::base64_decode(str);
		Assert::IsTrue(data == dataCheck);
	}

	};
}
