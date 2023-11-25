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

inline std::string kernelCompute = R"(
#pragma OPENCL FP_CONTRACT OFF

struct ArrayIndex {
	int r, c;
};

//map from thread index to S upper triangle
__constant struct ArrayIndex sidx[] = {
	{0,0}, {0,1}, {0,2}, {0,3}, {0,4}, {0,5},
	       {1,1}, {1,2}, {1,3}, {1,4}, {1,5},
	              {2,2}, {2,3}, {2,4}, {2,5},
	                     {3,3}, {3,4}, {3,5},
	                            {4,4}, {4,5},
	                                   {5,5},
};

//result type of one computed point
char FAIL_SINGULAR = -3;
char FAIL_ITERATIONS = -2;
char FAIL_ETA_NAN = -1;
char RUNNING = 0;
char SUCCESS_ABSOLUTE_ERR = 1;
char SUCCESS_STABLE_ITER = 2;

//result of one computed point in a frame
struct PointResult {
	double u, v;
	int idx, ix0, iy0;
	int px, py;
	int xm, ym;
	char result;
};

__kernel void compute(__read_only image2d_t Yp, __read_only image2d_t Y, __global struct PointResult* results) {
	uint ix0 = get_group_id(0);
	uint iy0 = get_group_id(1);
	uint blockIndex = iy0 * get_num_groups(0) + ix0;
	
	if (get_local_id(0) == 0 && get_local_id(1) == 0) {
		results[blockIndex].idx = blockIndex;
	}
}
)";
