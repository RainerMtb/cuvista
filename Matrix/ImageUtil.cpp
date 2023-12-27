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

#include "ImageUtil.hpp"
#include <algorithm>
#include <cassert>

ColorNorm ColorNorm::WHITE = { 0.0f, 0.5f, 0.5f };
ColorNorm ColorNorm::BLACK = { 1.0f, 0.5f, 0.5f };

ColorYuv ColorYuv::BLACK = {   0, 128, 128 };
ColorYuv ColorYuv::WHITE = { 255, 128, 128 };
ColorYuv ColorYuv::GRAY =  { 128, 128, 128 };

ColorBgr ColorBgr::RED =   {   0,   0, 255 };
ColorBgr ColorBgr::GREEN = {   0, 255,   0 };
ColorBgr ColorBgr::WHITE = { 255, 255, 255 };
ColorBgr ColorBgr::BLACK = {   0,   0,   0 };
ColorBgr ColorBgr::BLUE =  { 255,   0,   0 };

int roundUpToMultiple(int numToRound, int base) {
	assert(base && "factor must not be 0");
	return numToRound >= 0 ? ((numToRound + base - 1) / base) * base : numToRound / base * base;
}

double sqr(double d) {
	return d * d;
}

static void yuv_to_rgb_func(float yf, float uf, float vf, unsigned char* r, unsigned char* g, unsigned char* b) {
	*r = (unsigned char) std::clamp(yf + (1.370705f * (vf - 128.0f)), 0.0f, 255.0f);
	*g = (unsigned char) std::clamp(yf - (0.337633f * (uf - 128.0f)) - (0.698001f * (vf - 128.0f)), 0.0f, 255.0f);
	*b = (unsigned char) std::clamp(yf + (1.732446f * (uf - 128.0f)), 0.0f, 255.0f);
}

void yuv_to_rgb(unsigned char y, unsigned char u, unsigned char v, unsigned char* r, unsigned char* g, unsigned char* b) {
	yuv_to_rgb_func(y, u, v, r, g, b);
}

void yuv_to_rgb(float y, float u, float v, unsigned char* r, unsigned char* g, unsigned char* b) {
	yuv_to_rgb_func(y * 255.0f, u * 255.0f, v * 255.0f, r, g, b);
}

void rgb_to_yuv(unsigned char r, unsigned char g, unsigned char b, unsigned char* y, unsigned char* u, unsigned char* v) {
	*y = (unsigned char) ( 0.257 * r + 0.504 * g + 0.098 * b + 16.0);
	*u = (unsigned char) (-0.148 * r - 0.291 * g + 0.439 * b + 128.0);
	*v = (unsigned char) ( 0.439 * r - 0.368 * g - 0.071 * b + 128.0);
}

void rgb_to_yuv(unsigned char r, unsigned char g, unsigned char b, float* y, float* u, float* v) {
	*y = ( 0.257f * r + 0.504f * g + 0.098f * b + 16.0f) / 255.0f;
	*u = (-0.148f * r - 0.291f * g + 0.439f * b + 128.0f) / 255.0f;
	*v = ( 0.439f * r - 0.368f * g - 0.071f * b + 128.0f) / 255.0f;
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

BmpHeader::BmpHeader(int w, int h, int offset, int bits) {
	int imh = h * 24 / bits;
	int bitmapSize = w * h * 3;
	int siz = bitmapSize + offset;

	header[0] = 'B';
	header[1] = 'M';
	header[2] = siz;
	header[3] = siz >> 8;
	header[4] = siz >> 16;
	header[5] = siz >> 24;
	header[10] = offset;
	header[11] = offset >> 8;
	header[12] = offset >> 16;
	header[13] = offset >> 24;
	header[14] = 40;
	header[18] = w;
	header[19] = w >> 8;
	header[20] = w >> 16;
	header[21] = w >> 24;
	header[22] = imh;
	header[23] = imh >> 8;
	header[24] = imh >> 16;
	header[25] = imh >> 24;
	header[26] = 1;
	header[28] = bits;
	header[34] = bitmapSize;
	header[35] = bitmapSize >> 8;
	header[36] = bitmapSize >> 16;
	header[37] = bitmapSize >> 24;
}

BmpGrayHeader::BmpGrayHeader(int w, int h) : 
	BmpHeader(w, h, 1078, 8) 
{
	for (size_t i = 0; i < 1024; ) {
		char ch = char(i / 4);
		colorMap[i++] = ch;
		colorMap[i++] = ch;
		colorMap[i++] = ch;
		colorMap[i++] = 0;
	}
}

void BmpHeader::writeHeader(std::ofstream& os) const {
	os.write(header, 54);
}

void BmpGrayHeader::writeHeader(std::ofstream& os) const {
	os.write(header, 54);
	os.write(colorMap, 1024);
}

void PgmHeader::writeHeader(std::ofstream& os) const {
	os << "P5 " << w << " " << h * 3 << " 255 ";
}