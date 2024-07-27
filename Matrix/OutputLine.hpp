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

#include <string>

class OutputLine {
public:
	std::string mStr, mNumberStr;
	size_t mDecimalPos = 0;

	//append string to line
	void add(std::string& content) {
		mStr += content;
	}

	//append char to line
	void add(char content) {
		mStr += content;
	}

	//pad with blanks to given length
	void pad(size_t length) {
		while (mStr.length() < length) add(' ');
	}

	//setup number
	void setNumStr(std::string str, size_t* posDecimal) {
		mNumberStr = str;
		mDecimalPos = str.find('.');
		if (mDecimalPos > *posDecimal) *posDecimal = mDecimalPos;
	}

	//put in number and set length
	void addValue(size_t posDecimal, size_t* maxlen) {
		pad(mStr.length() + posDecimal - mDecimalPos);
		mStr += mNumberStr;
		if (mStr.length() > *maxlen) *maxlen = mStr.length();
	}
};