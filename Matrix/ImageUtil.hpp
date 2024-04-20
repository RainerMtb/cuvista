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

#include <fstream>
#include <string>
#include "CharMap.hpp"
#include "Color.hpp"

//-----------------------------------------
//Bitmap image headers
//-----------------------------------------

class ImageHeader {

protected:
	char header[54] = { 0 };

	virtual void writeHeader(std::ofstream& os) const {}
};

class BmpHeader : public ImageHeader {

protected:
	BmpHeader(int w, int h, int offset, int bits);

public:
	virtual void writeHeader(std::ofstream& os) const override;
};

class BmpColorHeader : public BmpHeader {

public:
	BmpColorHeader(int w, int h) : 
		BmpHeader(w, h, 54, 24) {}
};

class BmpGrayHeader : public ImageHeader {

protected:
	char colorMap[1024] = { 0 };

public:
	BmpGrayHeader(int w, int h);

	virtual void writeHeader(std::ofstream& os) const override;
};

class PgmHeader : public ImageHeader {

protected:
	int h, w;

public:
	PgmHeader(int w, int h) : 
		h { h }, 
		w { w } {}

	void writeHeader(std::ofstream& os) const override;
};
