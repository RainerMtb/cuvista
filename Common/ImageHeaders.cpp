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

#include "ImageHeaders.hpp"
#include <algorithm>
#include <cassert>

im::BmpHeader::BmpHeader(int w, int h, int offset, int bits) {
	int bytes = bits / 8;
	int bitmapSize = w * h * bytes;
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
	header[22] = h;
	header[23] = h >> 8;
	header[24] = h >> 16;
	header[25] = h >> 24;
	header[26] = 1;
	header[28] = bits;
	header[34] = bitmapSize;
	header[35] = bitmapSize >> 8;
	header[36] = bitmapSize >> 16;
	header[37] = bitmapSize >> 24;
}

im::BmpColorHeader::BmpColorHeader(int w, int h) :
	BmpHeader(w, h, 54, 24) {}

im::BmpGrayHeader::BmpGrayHeader(int w, int h) :
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

void im::BmpHeader::writeHeader(std::ofstream& os) const {
	os.write(header, 54);
}

void im::BmpGrayHeader::writeHeader(std::ofstream& os) const {
	os.write(header, 54);
	os.write(colorMap, 1024);
}

void im::PgmHeader::writeHeader(std::ofstream& os) const {
	os << "P5 " << w << " " << h * 3 << " 255 ";
}

void im::PpmHeader::writeHeader(std::ofstream& os) const {
	os << "P6 " << w << " " << h << " 255 ";
}