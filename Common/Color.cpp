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

#include <cassert>
#include <algorithm>
#include <cmath>
#include <regex>
#include "Color.hpp"

Color::Color() {
	colorData[0] = -1;
	colorData[1] = -1;
	colorData[2] = -1;
	alpha = -1.0;
}

Color::Color(int r, int g, int b, double alpha) {
	colorData[0] = r;
	colorData[1] = g;
	colorData[2] = b;
	this->alpha = alpha;
}

Color Color::WHITE =   Color(255, 255, 255);
Color Color::BLACK =   Color(  0,   0,   0);
Color Color::GRAY =    Color(128, 128, 128);
Color Color::RED =     Color(255,   0,   0);
Color Color::GREEN =   Color(  0, 255,   0);
Color Color::BLUE =    Color(  0,   0, 255);
Color Color::MAGENTA = Color(255,   0, 255);
Color Color::YELLOW =  Color(255, 255,   0);
Color Color::CYAN =    Color(  0, 255, 255);

Color Color::BLACK_SEMI = Color(0, 0, 0, 0.7);

Color Color::web(const std::string& webColor) {
	Color color;
	std::smatch matcher;
	if (std::regex_match(webColor, matcher, std::regex("#([[:xdigit:]]{2})([[:xdigit:]]{2})([[:xdigit:]]{2})$"))) {
		for (int i = 0; i < 3; i++) {
			color.colorData[i] = std::stoi(matcher[i].str(), nullptr, 16);
		}
	}
	return color;
}

Color Color::rgbDouble(double red, double green, double blue) {
	return Color((int) (red * 255.0), (int) (green * 255.0), (int) (blue * 255.0));
}

Color Color::rgb(int red, int green, int blue) {
	return Color(red, green, blue);
}

Color Color::rgba(int red, int green, int blue, double alpha) {
	return Color(red, green, blue, alpha);
}

Color Color::yuv(unsigned char y, unsigned char u, unsigned char v) {
	unsigned char r, g, b;
	im::yuv_to_rgb(y, u, v, &r, &g, &b);
	return Color(r, g, b);
}

Color Color::hsv(double h, double s, double v) {
	unsigned char r, g, b;
	im::hsv_to_rgb(h, s, v, &r, &g, &b);
	return Color(r, g, b);
}

std::vector<unsigned char> Color::getRGB() const {
	return { (unsigned char) colorData[0], (unsigned char) colorData[1], (unsigned char) colorData[2] };
}

std::vector<unsigned char> Color::getYUV() const {
	unsigned char y, u, v;
	im::rgb_to_yuv(colorData[0], colorData[1], colorData[2], &y, &u, &v);
	return { y, u, v };
}
	
std::array<float, 3> Color::getYUVfloats() const {
	float y, u, v;
	im::rgb_to_yuv(colorData[0], colorData[1], colorData[2], &y, &u, &v);
	return { y, u, v };
}

unsigned char Color::getRed() const { return (unsigned char) colorData[0]; }
unsigned char Color::getGreen() const { return (unsigned char) colorData[1]; }
unsigned char Color::getBlue() const { return (unsigned char) colorData[2]; }

double Color::getAlpha() const { return alpha; }

void Color::setAlpha(double alpha) { this->alpha = alpha; }

int Color::getRGBchannel(size_t index) const {
	assert(index >= 0 && index < 4 && "invalid index");
	return colorData[index];
}


	//----------------------------------------------------------------------------------------------------

namespace im {

	static void yuv_to_rgb_func(float yf, float uf, float vf, unsigned char* r, unsigned char* g, unsigned char* b) {
		*r = (unsigned char) std::rint(std::clamp(yf + (1.370705f * (vf - 128.0f)), 0.0f, 255.0f));
		*g = (unsigned char) std::rint(std::clamp(yf - (0.337633f * (uf - 128.0f)) - (0.698001f * (vf - 128.0f)), 0.0f, 255.0f));
		*b = (unsigned char) std::rint(std::clamp(yf + (1.732446f * (uf - 128.0f)), 0.0f, 255.0f));
	}

	void yuv_to_rgb(unsigned char y, unsigned char u, unsigned char v, unsigned char* r, unsigned char* g, unsigned char* b) {
		yuv_to_rgb_func(y, u, v, r, g, b);
	}

	void yuv_to_rgb(float y, float u, float v, unsigned char* r, unsigned char* g, unsigned char* b) {
		yuv_to_rgb_func(y * 255.0f, u * 255.0f, v * 255.0f, r, g, b);
	}

	void rgb_to_yuv(unsigned char r, unsigned char g, unsigned char b, unsigned char* y, unsigned char* u, unsigned char* v) {
		*y = (unsigned char) (0.257 * r + 0.504 * g + 0.098 * b + 16.0);
		*u = (unsigned char) (-0.148 * r - 0.291 * g + 0.439 * b + 128.0);
		*v = (unsigned char) (0.439 * r - 0.368 * g - 0.071 * b + 128.0);
	}

	void rgb_to_yuv(unsigned char r, unsigned char g, unsigned char b, float* y, float* u, float* v) {
		*y = (0.257f * r + 0.504f * g + 0.098f * b + 16.0f) / 255.0f;
		*u = (-0.148f * r - 0.291f * g + 0.439f * b + 128.0f) / 255.0f;
		*v = (0.439f * r - 0.368f * g - 0.071f * b + 128.0f) / 255.0f;
	}

	//input h [0..360], s [0..1], v [0..1]
	//output r, g, b [0..255]
	void hsv_to_rgb(double h, double s, double v, unsigned char* out_r, unsigned char* out_g, unsigned char* out_b) {
		double r, g, b;

		int i = (int) (h / 60);
		double f = h / 60.0 - i;
		double p = v * (1.0 - s);
		double q = v * (1.0 - s * f);
		double t = v * (1.0 - s * (1.0 - f));

		switch (i) {
		case 0:
			r = v; g = t; b = p;
			break;
		case 1:
			r = q; g = v; b = p;
			break;
		case 2:
			r = p; g = v; b = t;
			break;
		case 3:
			r = p; g = q; b = v;
			break;
		case 4:
			r = t; g = p; b = v;
			break;
		default:
			r = v; g = p; b = q;
			break;
		}

		*out_r = (unsigned char) (r * 255.0);
		*out_g = (unsigned char) (g * 255.0);
		*out_b = (unsigned char) (b * 255.0);
	}
}
