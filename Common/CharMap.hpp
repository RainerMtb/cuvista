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

#include <map>

//character definitions for stamping text onto image
//bitmap of 8 rows, 5 columns
const std::map<uint16_t, uint64_t> charMap = {
	{ '\0', 0b11111'11111'11111'11111'11111'11111'11111'00000 },
	{ '0',  0b01110'10001'10011'10101'11001'10001'01110'00000 },
	{ '1',  0b00100'01100'00100'00100'00100'00100'01110'00000 },
	{ '2',  0b01110'10001'00001'00010'00100'01000'11111'00000 },
	{ '3',  0b11111'00010'00100'00010'00001'10001'01110'00000 },
	{ '4',  0b00010'00110'01010'10010'11111'00010'00010'00000 },
	{ '5',  0b11111'10000'11110'00001'00001'10001'01110'00000 },
	{ '6',  0b00110'01000'10000'11110'10001'10001'01110'00000 },
	{ '7',  0b11111'00001'00010'00100'01000'01000'01000'00000 },
	{ '8',  0b01110'10001'10001'01110'10001'10001'01110'00000 },
	{ '9',  0b01110'10001'10001'01111'00001'00010'01100'00000 },
	{ ' ',  0b00000'00000'00000'00000'00000'00000'00000'00000 },
	{ 'a',  0b00000'00000'01110'00001'01111'10001'01111'00000 },
	{ 'b',  0b10000'10000'10110'11001'10001'10001'11110'00000 },
	{ 'c',  0b00000'00000'01110'10000'10000'10001'01110'00000 },
	{ 'd',  0b00001'00001'01101'10011'10001'10001'01111'00000 },
	{ 'e',  0b00000'00000'01110'10001'11111'10000'01110'00000 },
	{ 'f',  0b00110'01001'01000'11100'01000'01000'01000'00000 },
	{ 'g',  0b00000'00000'01111'10001'10001'01111'00001'01110 },
	{ 'h',  0b10000'10000'10110'11001'10001'10001'10001'00000 },
	{ 'i',  0b00100'00000'01100'00100'00100'00100'01110'00000 },
	{ 'j',  0b00010'00000'00110'00010'00010'10010'01100'00000 },
	{ 'k',  0b10000'10000'10010'10100'11000'10100'10010'00000 },
	{ 'l',  0b01100'00100'00100'00100'00100'00100'01110'00000 },
	{ 'm',  0b00000'00000'11010'10101'10101'10001'10001'00000 },
	{ 'n',  0b00000'00000'10110'11001'10001'10001'10001'00000 },
	{ 'o',  0b00000'00000'01110'10001'10001'10001'01110'00000 },
	{ 'p',  0b00000'00000'11110'10001'10001'11110'10000'10000 },
	{ 'q',  0b00000'00000'01101'10011'10001'01111'00001'00001 },
	{ 'r',  0b00000'00000'10110'11001'10000'10000'10000'00000 },
	{ 's',  0b00000'00000'01110'10000'01110'00001'11110'00000 },
	{ 't',  0b01000'01000'11100'01000'01000'01001'00110'00000 },
	{ 'u',  0b00000'00000'10001'10001'10001'10011'01101'00000 },
	{ 'v',  0b00000'00000'10001'10001'10001'01010'00100'00000 },
	{ 'w',  0b00000'00000'10001'10001'10101'10101'01010'00000 },
	{ 'x',  0b00000'00000'10001'01010'00100'01010'10001'00000 },
	{ 'y',  0b00000'00000'10001'10001'01111'00001'00010'01100 },
	{ 'z',  0b00000'00000'11111'00010'00100'01000'11111'00000 },
	{ 'A',  0b01110'10001'10001'10001'11111'10001'10001'00000 },
	{ 'B',  0b11110'10001'10001'11110'10001'10001'11110'00000 },
	{ 'C',  0b01110'10001'10000'10000'10000'10001'01110'00000 },
	{ 'D',  0b11100'10010'10001'10001'10001'10010'11100'00000 },
	{ 'E',  0b11111'10000'10000'11110'10000'10000'11111'00000 },
	{ 'F',  0b11111'10000'10000'11110'10000'10000'10000'00000 },
	{ 'G',  0b01110'10001'10000'10111'10001'10001'01111'00000 },
	{ 'H',  0b10001'10001'10001'11111'10001'10001'10001'00000 },
	{ 'I',  0b01110'00100'00100'00100'00100'00100'01110'00000 },
	{ 'J',  0b00111'00010'00010'00010'00010'10010'01100'00000 },
	{ 'K',  0b10001'10010'10100'11000'10100'10010'10001'00000 },
	{ 'L',  0b10000'10000'10000'10000'10000'10000'11111'00000 },
	{ 'M',  0b10001'11011'10101'10101'10001'10001'10001'00000 },
	{ 'N',  0b10001'10001'11001'10101'10011'10001'10001'00000 },
	{ 'O',  0b01110'10001'10001'10001'10001'10001'01110'00000 },
	{ 'P',  0b11110'10001'10001'11110'10000'10000'10000'00000 },
	{ 'Q',  0b01110'10001'10001'10001'10101'10010'01101'00000 },
	{ 'R',  0b11110'10001'10001'11110'10100'10010'10001'00000 },
	{ 'S',  0b01111'10000'10000'01110'00001'00001'11110'00000 },
	{ 'T',  0b11111'00100'00100'00100'00100'00100'00100'00000 },
	{ 'U',  0b10001'10001'10001'10001'10001'10001'01110'00000 },
	{ 'V',  0b10001'10001'10001'10001'10001'01010'00100'00000 },
	{ 'W',  0b10001'10001'10001'10101'10101'10101'01010'00000 },
	{ 'X',  0b10001'10001'01010'00100'01010'10001'10001'00000 },
	{ 'Y',  0b10001'10001'10001'01010'00100'00100'00100'00000 },
	{ 'Z',  0b11111'00001'00010'00100'01000'10000'11111'00000 },
	{ 0xE4,  0b01010'00000'01110'00001'01111'10001'01111'00000 }, //ae
	{ 0xF6,  0b01010'00000'01110'10001'10001'10001'01110'00000 }, //oe
	{ 0xFC,  0b01010'00000'10001'10001'10001'10011'01101'00000 }, //ue
	{ 0xDF,  0b01110'10001'10001'10110'10001'10001'10110'00000 }, //szlig
	{ '%',  0b11001'11001'00010'00100'01000'10011'10011'00000 },
	{ ',',  0b00000'00000'00000'00000'00000'00100'00100'01000 },
	{ '.',  0b00000'00000'00000'00000'00000'01100'01100'00000 },
	{ '(',  0b00001'00010'00010'00010'00010'00010'00001'00000 },
	{ ')',  0b10000'01000'01000'01000'01000'01000'10000'00000 },
	{ '/',  0b00000'00000'00001'00010'00100'01000'10000'00000 },
	{ '=',  0b00000'00000'11111'00000'11111'00000'00000'00000 },
	{ '-',  0b00000'00000'00000'01111'00000'00000'00000'00000 },
};