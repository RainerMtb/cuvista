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

#include "cuDecompose.cuh"
#include <stdio.h>

__device__ void printPointers(double* mat, int s) {
	if (cu::firstThread()) {
		printf("A:    ");
		for (int i = 0; i < s; i++) printf("%llp ", mat + 1ull * i * s);
		printf("\n");
	}
}

__device__ void printPointers(double** mat, int s) {
	if (cu::firstThread()) {
		printf("A:    ");
		for (int i = 0; i < s; i++) printf("%llp ", mat[i]);
		printf("\n");
	}
}

__device__ void printmat(double** data, int h, int w, const char* title = "") {
	if (cu::firstThread()) {
		printf("-- mat %s =\n", title);
		for (int r = 0; r < h; r++) {
			printf("[");
			for (int c = 0; c < w; c++) {
				printf("%.17g ", data[r][c]);
			}
			printf("]\n");
		}
		printf("-- mat end --\n");
	}
}

__device__ void printmat(double** data, int s, const char* title = "") {
	printmat(data, s, s, title);
}

__device__ void printmat(double* data, int h, int w, const char* title = "") {
	if (cu::firstThread()) {
		double** rows = new double* [h];
		for (int i = 0; i < h; i++) rows[i] = data + 1ull * w * i;
		printmat(rows, h, w, title);
		delete[] rows;
	}
}

__device__ void printmat(double* data, int s, const char* title = "") {
	printmat(data, s, s, title);
}

__device__ void luinv(double** Apiv, double* A0, double* temp, double* Ainv, int s, int r, int ci, int cols) {
	//step 1. decompose matrix A0
	if (r < s && ci == 0) {
		Apiv[r] = A0 + 1ull * r * s;

		//iterate j over columns
		for (int j = 0; j < s; j++) {
			// Apply previous transformations, compute dot products
			for (int i = 0; i < j; i++) {
				if (r > i) {
					Apiv[r][j] -= Apiv[r][i] * Apiv[i][j];
				}
			}

			// Find pivot and set new row pointers
			//store abs values in temp for column
			temp[r] = fabs(Apiv[r][j]);

			if (r == 0) {
				//find max value in temp south of diagonal
				int p = j;
				for (int i = j + 1; i < s; i++) {
					if (temp[i] > temp[p]) p = i;
				}
				//exchange row pointers
				double* dp = Apiv[p];
				Apiv[p] = Apiv[j];
				Apiv[j] = dp;
			}

			// Compute multipliers
			if (r > j) Apiv[r][j] /= Apiv[j][j];
		}
	}

	//step 2. solve decomposed matrix against identity mat
	if (r < s && ci < cols) {
		//form identity matrix, take into account reordered rows in Apiv
		for (int c = ci; c < s; c += cols) {
			Ainv[r * s + c] = 0.0;
		}
		if (ci == 0) {
			int pivr = (Apiv[r] - A0) / s;
			Ainv[1ull * r * s + pivr] = 1.0;
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

__device__ double& norm1(const double* mat, int m, int n, double* temp) {
	int i = threadIdx.y;
	int k = threadIdx.x;
	assert(m > 0 && n > 0 && "invalid matrix dimension");

	if (i < n && k == 0) {
		temp[i] = fabs(mat[i]); //take values from first row
		for (int y = 1; y < m; y++) {
			temp[i] += fabs(mat[y * n + i]);
		}
	}
	if (i == 0 && k == 0) {
		for (int x = 1; x < n; x++) {
			double val = temp[x];
			if (isnan(val) || val > temp[0]) temp[0] = val; //find max value, check nan beforehand
		}
	}
	return temp[0];
}