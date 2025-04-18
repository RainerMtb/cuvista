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

namespace im {

	ColorNorm ColorNorm::WHITE = { 0.0f, 0.5f, 0.5f };
	ColorNorm ColorNorm::BLACK = { 1.0f, 0.5f, 0.5f };

	ColorYuv ColorYuv::BLACK =   { 0,   128, 128 };
	ColorYuv ColorYuv::WHITE =   { 255, 128, 128 };
	ColorYuv ColorYuv::GRAY =    { 128, 128, 128 };

	ColorBgr ColorBgr::RED =     {   0,   0, 255 };
	ColorBgr ColorBgr::GREEN =   {   0, 255,   0 };
	ColorBgr ColorBgr::WHITE =   { 255, 255, 255 };
	ColorBgr ColorBgr::BLACK =   {   0,   0,   0 };
	ColorBgr ColorBgr::BLUE =    { 255,   0,   0 }; 
	ColorBgr ColorBgr::CYAN =    { 255, 255,   0 };
	ColorBgr ColorBgr::MAGENTA = { 255,   0, 255 }; 
	ColorBgr ColorBgr::YELLOW =  {   0, 255, 255 }; 

	ColorRGBA ColorRGBA::BLACK = {   0,   0,   0, 255 };
	ColorRGBA ColorRGBA::WHITE = { 255, 255, 255, 255 };
	ColorRGBA ColorRGBA::GRAY =  { 128, 128, 128, 255 };

	template <class T> void ColorBase<T>::setColors(const std::string& webColor, std::array<int, 3> index) {
		std::smatch matcher;
		if (std::regex_match(webColor, matcher, std::regex("#([[:xdigit:]]{2})([[:xdigit:]]{2})([[:xdigit:]]{2})$"))) {
			for (int i = 0; i < 3; i++) {
				int group = index[i] + 1;
				colors[i] = (unsigned char) std::stoi(matcher[group].str(), nullptr, 16);
			}
		}
	}

	ColorRgb ColorRgb::webColor(const std::string& webColor) {
		ColorRgb out;
		out.setColors(webColor, { 0, 1, 2 });
		return out;
	}

	ColorBgr ColorBgr::webColor(const std::string& webColor) {
		ColorBgr out;
		out.setColors(webColor, { 2, 1, 0 });
		return out;
	}

	unsigned char ColorYuv::y() const { return colors[0]; }
	unsigned char ColorYuv::u() const { return colors[1]; }
	unsigned char ColorYuv::v() const { return colors[2]; }

	unsigned char ColorRgb::r() const { return colors[0]; }
	unsigned char ColorRgb::g() const { return colors[1]; }
	unsigned char ColorRgb::b() const { return colors[2]; }

	unsigned char ColorBgr::r() const { return colors[2]; }
	unsigned char ColorBgr::g() const { return colors[1]; }
	unsigned char ColorBgr::b() const { return colors[0]; }

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

	ColorYuv ColorRgb::toYuv() const {
		ColorYuv out {};
		rgb_to_yuv(colors[0], colors[1], colors[2], &out.colors[0], &out.colors[1], &out.colors[2]);
		return out;
	}

	ColorNorm ColorRgb::toNormalized() const {
		ColorYuv yuv = toYuv();
		return { yuv.colors[0] / 255.0f, yuv.colors[1] / 255.0f, yuv.colors[2] / 255.0f };
	}

	ColorRgb ColorYuv::toRgb() const {
		ColorRgb out {};
		yuv_to_rgb(colors[0], colors[1], colors[2], &out.colors[0], &out.colors[1], &out.colors[2]);
		return out;
	}
}
