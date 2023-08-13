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

inline std::string luinvFunction = R"(
void luinv(double** Apiv, double* A0, double* temp, double* Ainv, int s, int r, int ci, int cols) {
	//step 1. decompose matrix A0
	if (r < s && ci == 0) {
		Apiv[r] = A0 + 1ull * r * s;

		//iterate j over columns
		for (int j = 0; j < s; j++) {
			for (int i = 0; i < j; i++) {
				if (r > i) {
					Apiv[r][j] -= Apiv[r][i] * Apiv[i][j];
				}
			}
			temp[r] = fabs(Apiv[r][j]);

			if (r == 0) {
				int p = j;
				for (int i = j + 1; i < s; i++) {
					if (temp[i] > temp[p]) p = i;
				}
				double* dp = Apiv[p];
				Apiv[p] = Apiv[j];
				Apiv[j] = dp;
			}

			if (r > j) {
				Apiv[r][j] /= Apiv[j][j];
			}
		}
	}

	//step 2. solve decomposed matrix against identity mat
	if (r < s && ci < cols) {
		for (int c = ci; c < s; c += cols) {
			Ainv[r * s + c] = 0.0;
		}
		if (ci == 0) {
			Ainv[1ull * r * s + (Apiv[r] - A0) / s] = 1.0;
		}

		//forward substitution
		for (int k = 0; k < s; k++) {
			for (int c = ci; c < s; c += cols) {
				if (r > k) Ainv[r * s + c] -= Ainv[k * s + c] * Apiv[r][k];
			}
		}

		//backwards substitution
		for (int k = s - 1; k >= 0; k--) {
			double ap = Apiv[k][k];
			for (int c = ci; c < s; c += cols) {
				double ai = Ainv[k * s + c];
				if (r == k) Ainv[r * s + c] /= ap;
				if (r < k) Ainv[r * s + c] -= ai / ap * Apiv[r][k];
			}
		}
	}
}
)";

inline std::string luinvTestKernel = R"(
__kernel void luinvTest(__global double* input, __global double* outAinv, int s, __local double* temp) {
	int c = get_global_id(0) / s;
	int cols = get_global_size(0) / s;
	int r = get_global_id(0) - c * s;

	double** Apiv = (double**) (temp + s);
	luinv(Apiv, input, temp, outAinv, s, r, c, cols);
}
)";