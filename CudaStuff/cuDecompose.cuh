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

#include "cuUtil.cuh"

//print as adresses
__device__ void printPointers(double* mat, int s);

//print as adresses
__device__ void printPointers(double** mat, int s);

//print square matrix given index into rows
__device__ void printmat(double** data, int s, const char* title);

//print square matrix given 1-D array
__device__ void printmat(double* data, int s, const char* title);

//compute inverse from already provided decomposition
__device__ void luinv(double** Apiv, double* A0, double* temp, double* Ainv, int s, int r, int ci, int cols);

//compute 1-norm from matrix given as 1-D array
__device__ double& norm1(const double* mat, int m, int n, double* temp);
