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

//store color information in RGB array with alpha value
class Color {

private:
	std::array<int, 4> colorData;
	double alpha;

	Color(int r, int g, int b, double alpha = 1.0);

public:
	Color();

	static Color WHITE;
	static Color BLACK;
	static Color GRAY;
	static Color RED;
	static Color GREEN;
	static Color BLUE;
	static Color MAGENTA;
	static Color YELLOW;
	static Color CYAN;
	static Color BLACK_SEMI;

	static Color rgbDouble(double red, double green, double blue);
	static Color rgb(int red, int green, int blue);
	static Color rgba(int red, int green, int blue, double alpha = 1.0);
	static Color web(const std::string& webColor);
	static Color yuv(unsigned char y, unsigned char u, unsigned char v);
	static Color hsv(double h, double s, double v);

	std::vector<unsigned char> getYUV() const;
	void toYUVfloat(float* y, float* u, float* v) const;

	void setAlpha(double alpha);
	double getAlpha() const;

	int getChannel(size_t index) const;
};

namespace im {

	void yuv_to_rgb(unsigned char y, unsigned char u, unsigned char v, unsigned char* r, unsigned char* g, unsigned char* b);

	void yuv_to_rgb(float y, float u, float v, unsigned char* r, unsigned char* g, unsigned char* b);

	void rgb_to_yuv(unsigned char r, unsigned char g, unsigned char b, unsigned char* y, unsigned char* u, unsigned char* v);

	void rgb_to_yuv(unsigned char r, unsigned char g, unsigned char b, float* y, float* u, float* v);

	void hsv_to_rgb(double h, double s, double v, unsigned char* out_r, unsigned char* out_g, unsigned char* out_b);
}