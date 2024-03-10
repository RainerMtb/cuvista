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

#include "RandomSource.hpp"

PseudoRandomSource::result_type PseudoRandomSource::operator () () {
	//compute output value
	uint16_t out = (~rngData[idx1] << 8) | (rngData[idx2]);

	//advance state
	idx1++;
	if (idx1 == 256) {
		idx1 = 0;
		idx2++;
	}
	if (idx2 == 256) {
		idx2 = 0;
	}

	//return result
	return out;
}
