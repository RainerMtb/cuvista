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

#include "AffineSolverDirect.hpp"

void AffineSolverDirect::initParam(std::span<PointBase> points) {
	m = points.size() * 2;
	dy = points[0].y;     //represents a2, needs to be smallest value
	dx = points[1].x;     //represents a3
	double nn = 0;        //int could overflow
	s0 = 0;               //in docs s1
	s1 = 0;               //in docs s2

	double* p = M.addr(4, 0);   //5th row holds adjusted x and y values
	double* q = M.addr(5, 0);   //6th row holds adjusted b values

	//accumulate s0, s1, nn and adjust coords
	auto it = points.begin();
	for (size_t idx = 0; idx < m; ) {
		double x = it->x - dx;
		p[idx] = x;
		s0 += x;
		nn += x * x;
		q[idx] = it->x + it->u - dx;
		idx++;

		double y = it->y - dy;
		p[idx] = y;
		s1 += y;
		nn += y * y;
		q[idx] = it->y + it->v - dy;
		idx++;

		it++;
	}

	//compute parameters
	double sign0 = p[0] < 0 ? -1.0 : 1.0;
	n = std::sqrt(nn) * sign0;
	b = p[0] + n;
	double sign2 = sign0 * n * b < p[3] * s1 ? -1.0 : 1.0;
	double t = sign2 * std::sqrt(points.size() * nn - s0 * s0 - s1 * s1);
	double z = b * (n + t) - p[3] * s1;
	e = n / t;
	f = s1 / (b * t);

	double sn = s0 + n;
	g = sn / (b * t);
	h = (p[3] / b * (sn * sn + s1 * s1) - s1 * (n + t)) / (t * z);
	j = sn * (t + n) / (t * z);
	k = p[3] * n * sn / (t * z);

	rd[0] = n;
	rd[1] = -n;
	rd[2] = t / n;
	rd[3] = t / n;
};
