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
__constant char FAIL_SINGULAR = -3;
__constant char FAIL_ITERATIONS = -2;
__constant char FAIL_ETA_NAN = -1;
__constant char RUNNING = 0;
__constant char SUCCESS_ABSOLUTE_ERR = 1;
__constant char SUCCESS_STABLE_ITER = 2;

//result of one computed point in a frame
//definition must match host code
struct PointResult {
	double u, v;
	int xm, ym;
	int idx, ix0, iy0;
	char result;
	int zp;
	int direction;
    char computed; //do not use bool
};

//core data in opencl structure
//definition must match host code
struct KernelData {
	double compMaxTol, deps, dmin, dmax, dnan;
	int w, h, ir, iw, zMin, zMax, compMaxIter, pyramidRowCount;
};

//initial values for wp
__constant double wp0[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
//initial values for eta
__constant double eta0[] = { 0, 0, 1, 0, 0, 1 };

//-------------------------------
//------ compute kernel ---------
//-------------------------------
__kernel void compute(long frameIndex, __read_only image2d_array_depth_t Y, 
			__global struct PointResult* results, __constant struct KernelData* d_core, 
			__local double* shd, int rowStart) {
	const int ix0 = get_group_id(0);
	const int iy0 = get_group_id(1) + rowStart;
	const int direction = (ix0 % 2) ^ (iy0 % 2);
	const long idx0 = (frameIndex - direction) % get_image_array_size(Y);
	const long idx1 = (frameIndex - direction + 1) % get_image_array_size(Y);

	const int blockIndex = iy0 * get_num_groups(0) + ix0;
	const int ci = get_local_id(1);
	const int cols = get_local_size(1);
	const int r = get_local_id(0);
	const int tidx = r * cols + ci;

	int ir = d_core->ir;
	int iw = d_core->iw;

	double* ptr = shd;
	double* sd = ptr;		ptr += 6 * iw * iw;
	double* delta = ptr;	ptr += 1 * iw * iw;
	double* s = ptr;		ptr += 36;
	double* g = ptr;		ptr += 36;
	double* wp = ptr;		ptr += 9;
	double* dwp = ptr;		ptr += 9;
	double* b = ptr;        ptr += 6;
	double* eta = ptr;      ptr += 6;
	double* temp = ptr;     ptr += 6;
	double** Apiv = (double**) ptr;

	//init wp and dwp to identitiy
	if (r < 3 && ci < 3) {
		dwp[r * 3 + ci] = wp[r * 3 + ci] = wp0[r * 3 + ci];
	}

	//center point of image patch in this block
	int ym = iy0 + ir + 1;
	int xm = ix0 + ir + 1;
	char result = RUNNING;

	int z = d_core->zMax;
	int rowOffset = d_core->pyramidRowCount - (d_core->h >> z);

	double err = 0.0;
	for (; z >= d_core->zMin && result >= RUNNING; z--) {
		int wz = d_core->w >> z;
		int hz = d_core->h >> z;

		if (r < iw) {
			for (int c = ci; c < iw; c += cols) {
				int ix = xm - ir + r;
				int iy = ym - ir + c + rowOffset;
				double x = read_imagef(Y, (int4)(ix + 1, iy, idx1, 0)) / 2 - read_imagef(Y, (int4)(ix - 1, iy, idx1, 0)) / 2;
				double y = read_imagef(Y, (int4)(ix, iy + 1, idx1, 0)) / 2 - read_imagef(Y, (int4)(ix, iy - 1, idx1, 0)) / 2;
				int idx = r * iw + c;
				sd[idx] = x;
				idx += iw * iw;
				sd[idx] = y;
				idx += iw * iw;
				sd[idx] = x * (r - ir);
				idx += iw * iw;
				sd[idx] = y * (r - ir);
				idx += iw * iw;
				sd[idx] = x * (c - ir);
				idx += iw * iw;
				sd[idx] = y * (c - ir);
			}
		}

		//S = sd * sd' [6 x 6]
		if (tidx < 21) {
			const struct ArrayIndex ai = sidx[tidx];
			double sval = 0.0;
			for (int i = 0; i < iw * iw; i++) {
				sval += sd[ai.r * iw * iw + i] * sd[ai.c * iw * iw + i];
			}
			//copy symmetric value
			s[ai.c * 6 + ai.r] = s[ai.r * 6 + ai.c] = sval;
		}

		//compute norm before starting inverse, s will be overwritten
		double ns = norm1(s, 6, 6, temp);
		//if (frameIndex == 1 && ix0 == 29 && iy0 == 2) printf("ocl %d %d %.14f\n", r, ci, ns);

		//compute inverse
		luinv(Apiv, s, temp, g, 6, r, ci, cols);

		//compute reciprocal condition, see if result is valid
		double ng = norm1(g, 6, 6, temp);
		double rcond = 1 / (ns * ng);
		result = (isnan(rcond) || rcond < d_core->deps) ? FAIL_SINGULAR : RUNNING;

		//init loop limit counter
		int iter = 0;
		//init error measure to stop loop
		double bestErr = d_core->dmax;
		//main loop to find transformation of image patch
		while (result == RUNNING) {
			//interpolate image patch
			if (r < iw) {
				for (int c = ci; c < iw; c += cols) {
					double ix = xm + (c - ir) * wp[0] + (r - ir) * wp[3] + wp[2];
					double iy = ym + (c - ir) * wp[1] + (r - ir) * wp[4] + wp[5];

					//compute difference between image patches
					//store delta in transposed order [c * iw + r]
					if (ix < 0.0 || ix > wz - 1.0 || iy < 0.0 || iy > hz - 1.0) {
						delta[c * iw + r] = d_core->dnan;

					} else {
						double im = read_imagef(Y, (int4)(xm - ir + c, rowOffset + ym + r - ir, idx1, 0));

						double flx = floor(ix), fly = floor(iy);
						double dx = ix - flx, dy = iy - fly;
						int x0 = (int) flx, y0 = (int) fly;

						double f00 = read_imagef(Y, (int4)(x0,     rowOffset + y0,     idx0, 0));
						double f01 = read_imagef(Y, (int4)(x0 + 1, rowOffset + y0,     idx0, 0));
						double f10 = read_imagef(Y, (int4)(x0,     rowOffset + y0 + 1, idx0, 0));
						double f11 = read_imagef(Y, (int4)(x0 + 1, rowOffset + y0 + 1, idx0, 0));
						double jm = (1.0 - dx) * (1.0 - dy) * f00 + (1.0 - dx) * dy * f10 + dx * (1.0 - dy) * f01 + dx * dy * f11;

						delta[c * iw + r] = im - jm;
					}
				}
			}

			//eta [6 x 1]
			if (r < 6 && ci == 0) {
				//init eta to [0 0 1 0 0 1]
				eta[r] = eta0[r];
				//init b to [0 0 0 0 0 0]
				double bval = 0.0;
				//sd * delta_flat
				for (double* sdptr = sd + r * iw * iw, *deltaptr = delta; deltaptr != delta + iw * iw; sdptr++, deltaptr++) {
					bval += (*sdptr) * (*deltaptr);
				}
				b[r] = bval;
				//g * (sd * delta)
				for (double* gptr = g + r * 6, *bptr = b; bptr != b + 6; gptr++, bptr++) {
					eta[r] += (*gptr) * (*bptr);
				}
			}
			work_group_barrier(CLK_LOCAL_MEM_FENCE);

			//update transform matrix
			if (r < 2 && ci == 0) {
				//update wp to dwp
				dwp[r * 3]     = wp[r * 3] * eta[2] + wp[r * 3 + 1] * eta[4];
				dwp[r * 3 + 1] = wp[r * 3] * eta[3] + wp[r * 3 + 1] * eta[5];
				dwp[r * 3 + 2] = wp[r * 3] * eta[0] + wp[r * 3 + 1] * eta[1] + wp[r * 3 + 2];

				//update wp
				wp[r * 3]     = dwp[r * 3];
				wp[r * 3 + 1] = dwp[r * 3 + 1];
				wp[r * 3 + 2] = dwp[r * 3 + 2];
			}
			work_group_barrier(CLK_LOCAL_MEM_FENCE);

			//analyse result, decide on continuing loop
			err = eta[0] * eta[0] + eta[1] * eta[1];
			if (isnan(err)) result = FAIL_ETA_NAN; //leave loop with fail message FAIL_ETA_NAN
			if (err < d_core->compMaxTol) result = SUCCESS_ABSOLUTE_ERR; //leave loop with success SUCCESS_ABSOLUTE_ERR
			if (fabs(err - bestErr) / bestErr < d_core->compMaxTol * d_core->compMaxTol) result = SUCCESS_STABLE_ITER; //SUCCESS_STABLE_ITER
			bestErr = min(err, bestErr);
			iter++;
			if (iter == d_core->compMaxIter && result == RUNNING) result = FAIL_ITERATIONS; //leave with fail
		}

		//displacement * 2 for next level
		if (r == 0 && ci == 0) wp[2] *= 2.0;
		if (r == 1 && ci == 0) wp[5] *= 2.0;

		//center of integration window on next level
		xm *= 2;
		ym *= 2;

		//new texture row offset
		int delta = d_core->h >> (z - 1);
		rowOffset -= delta;

		//if (frameIndex == 1 && ix0 == 29 && iy0 == 2 && r == 0 && ci == 0) printf("ocl %d %.14f\n", z, wp[5]);
	}

	//first thread writes into result structure
	if (get_local_linear_id() == 0) {
		double u = wp[2];
		double v = wp[5];
		int zp = z;

		while (z < 0) { xm /= 2; ym /= 2; u /= 2; v /= 2; z++; }
		while (z > 0) { xm *= 2; ym *= 2; u *= 2; v *= 2; z--; }

		results[blockIndex].idx = blockIndex;
		results[blockIndex].ix0 = ix0;
		results[blockIndex].iy0 = iy0;
		results[blockIndex].xm = xm;
		results[blockIndex].ym = ym;
		results[blockIndex].u = u;
		results[blockIndex].v = v;
		results[blockIndex].result = result;
		results[blockIndex].zp = zp;
		results[blockIndex].direction = direction;
		results[blockIndex].computed = 1;
	}
}
)";
