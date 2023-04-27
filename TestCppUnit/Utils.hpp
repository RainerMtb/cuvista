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

#include "AffineTransform.hpp"
#include "CoreData.cuh"

namespace Microsoft::VisualStudio::CppUnitTestFramework {

	static std::wstring toWString(const std::string& str) {
		return std::wstring(str.cbegin(), str.cend());
	}

	static std::wstring toWString(const std::stringstream& buf) {
		return toWString(buf.str());
	}

	static std::wstring toWString(const Matd& mat) {
		return mat.toWString();
	}

	template <> static std::wstring ToString(const PointResult& res) {
		return toWString("ix0=" + std::to_string(res.ix0) + ", iy0=" + std::to_string(res.iy0) + ", u=" + std::to_string(res.u) + ", v=" + std::to_string(res.v));
	}

	template <> static std::wstring ToString(const AffineTransform& res) {
		return res.toWString();
	}

	template <> static std::wstring ToString(const Mat<double>& mat) {
		return mat.toWString();
	}

	template <> static std::wstring ToString(const Mat<float>& mat) {
		return mat.toWString();
	}

	template <> static std::wstring ToString(const ImageYuv& im) {
		return L"image w=" + std::to_wstring(im.w) + L" h=" + std::to_wstring(im.h);
	}
}
