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

#include <array>

namespace im {

	int alignValue(int numToRound, int base);

	double sqr(double d);

	void yuv_to_rgb(unsigned char y, unsigned char u, unsigned char v, unsigned char* r, unsigned char* g, unsigned char* b);

	void yuv_to_rgb(float y, float u, float v, unsigned char* r, unsigned char* g, unsigned char* b);

	void rgb_to_yuv(unsigned char r, unsigned char g, unsigned char b, unsigned char* y, unsigned char* u, unsigned char* v);

	void rgb_to_yuv(unsigned char r, unsigned char g, unsigned char b, float* y, float* u, float* v);

	void hsv_to_rgb(double h, double s, double v, unsigned char* out_r, unsigned char* out_g, unsigned char* out_b);

	template <class T, int SIZ = 3> class ColorBase {

	public:
		std::array<T, SIZ> colors;
		double alpha = 1.0;
	};

	class ColorNorm : public ColorBase<float> {
	public:
		static ColorNorm WHITE;
		static ColorNorm BLACK;
	};

	class ImageColor : public ColorBase<unsigned char> {};

	class ColorRgb;

	class ColorYuv : public ImageColor {
	public:
		static ColorYuv WHITE;
		static ColorYuv BLACK;
		static ColorYuv GRAY;

		ColorRgb toRgb() const;

		unsigned char y() const { return colors[0]; }

		unsigned char u() const { return colors[1]; }

		unsigned char v() const { return colors[2]; }
	};

	class ColorRgb : public ImageColor {
	public:
		ColorYuv toYuv() const;

		ColorNorm toNormalized() const;

		unsigned char r() const { return colors[0]; }

		unsigned char g() const { return colors[1]; }

		unsigned char b() const { return colors[2]; }
	};

	class ColorBgr : public ImageColor {
	public:
		static ColorBgr RED;
		static ColorBgr GREEN;
		static ColorBgr WHITE;
		static ColorBgr BLACK;
		static ColorBgr BLUE;
	};

	class ColorRGBA : public ColorBase<unsigned char, 4> {
	public:
		static ColorRGBA BLACK;
	};
}