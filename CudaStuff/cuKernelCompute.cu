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

#include "cuDeshaker.cuh"

struct ArrayIndex {
	int r, c;
};

//map from thread index to S upper triangle
__constant__ ArrayIndex sidx[] = {
	{0,0}, {0,1}, {0,2}, {0,3}, {0,4}, {0,5},
	       {1,1}, {1,2}, {1,3}, {1,4}, {1,5},
	              {2,2}, {2,3}, {2,4}, {2,5},
	                     {3,3}, {3,4}, {3,5},
	                            {4,4}, {4,5},
	                                   {5,5},
};

//initial values for wp
__constant__ double wp0[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
//initial values for eta
__constant__ double eta0[] = { 0, 0, 1, 0, 0, 1 };

//forward declaration to prevent underline
template<class T> __device__ T tex2D(cudaTextureObject_t tex, float x, float y);

extern __constant__ CoreData d_core;

//compute value for sd matrix directly
__device__ double sdf(int r, int c1, int c2, int y0, int x0, cudaTextureObject_t tex) {
	int idx = r / 2;
	int dy = r % 2;
	int dx = 1 - dy;
	double val = tex2D<float>(tex, x0 + c1 + dx, y0 + c2 + dy) / 2 - tex2D<float>(tex, x0 + c1 - dx, y0 + c2 - dy) / 2;
	int f[] = { 1, c1 - d_core.ir, c2 - d_core.ir };
	return val * f[idx];
}

//compute displacement
//one cuda block works one point in the image using one warp
__global__ void kernelCompute(ComputeTextures tex, CudaPointResult* results, ComputeKernelParam param) {
	int ix0 = blockIdx.x;
	int iy0 = blockIdx.y;
	int blockIndex = iy0 * gridDim.x + ix0;
	if (*param.d_interrupt || results[blockIndex].computed) return;

	int64_t timeStart = cu::globaltimer();
	int direction = (ix0 % 2) ^ (iy0 % 2);
	cudaTextureObject_t pyr0 = tex.Y[direction];
	cudaTextureObject_t pyr1 = tex.Y[1 - direction];

	int ir = d_core.ir;
	int iw = d_core.iw;

	//allocate individual variables in shared memory
	extern __shared__ double shd[];
	double* ptr = shd;
	double* sd = ptr;		ptr += 6 * iw * iw;  // 6 x iw*iw
	double* delta = ptr;	ptr += 1 * iw * iw;  // iw x iw
	double* s = ptr;		ptr += 36;           // 6 x 6
	double* g = ptr;		ptr += 36;           // 6 x 6
	double* wp = ptr;		ptr += 9;            // 3 x 3
	double* dwp = ptr;		ptr += 9;            // 3 x 3
	double* b = ptr;        ptr += 6;            // 6 doubles
	double* eta = ptr;      ptr += 6;            // 6 doubles
	double* temp = ptr;     ptr += 6;            // 6 doubles
	double** Apiv = (double**) (ptr);            // 6 double pointers to rows in LU decomposition

	const int ci = threadIdx.x;	    //column into image
	const int cols = blockDim.x;	//columns that can be addressed in one warp
	const int r = threadIdx.y;		//row into image
	const int tidx = r * cols + ci;

	//init wp and dwp to identitiy
	if (r < 3 && ci < 3) {
		dwp[r * 3 + ci] = wp[r * 3 + ci] = wp0[r * 3 + ci];
	}

	//center point of image patch in this block
	int ym = iy0 + ir + 1;
	int xm = ix0 + ir + 1;
	PointResultType result = PointResultType::RUNNING;

	//pyramid level to start at
	int z = d_core.zMax;
	//offset in rows to current pyramid level as texture spans one full pyramid
	int rowOffset = d_core.pyramidRowCount;

	double err = 0.0;
	for (; z >= d_core.zMin && result >= PointResultType::RUNNING; z--) {
		//dimensions for current pyramid level
		int wz = d_core.w >> z;
		int hz = d_core.h >> z;
		rowOffset -= (d_core.h >> z);

		//build sd matrix [6 x iw*iw]
		if (r < iw) {
			int ix = xm - ir + r;
			for (int c = ci; c < iw; c += cols) {
				int iy = ym - ir + c + rowOffset;
				double x = tex2D<float>(pyr0, ix + 1, iy) / 2 - tex2D<float>(pyr0, ix - 1, iy) / 2;
				double y = tex2D<float>(pyr0, ix, iy + 1) / 2 - tex2D<float>(pyr0, ix, iy - 1) / 2;
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
		//if (param.frameIdx == 1 && ix0 == 63 && iy0 == 1 && cu::firstThread()) cu::storeDebugData(param.debugData, param.debugDataSize, 6, 49, sd);

		//S = sd * sd' [6 x 6]
		//compute upper triangle and mirror value to write all values for S
		if (tidx < 21) {
			const ArrayIndex& ai = sidx[tidx]; //the value to compute in S
			double sval = 0.0;
			for (int i = 0; i < iw * iw; i++) {
				sval += sd[ai.r * iw * iw + i] * sd[ai.c * iw * iw + i];
			}
			//copy symmetric value
			s[ai.c * 6 + ai.r] = s[ai.r * 6 + ai.c] = sval;
		}
		//if (param.frameIdx == 1 && ix0 == 63 && iy0 == 1 && cu::firstThread()) cu::storeDebugData(param.debugData, param.debugDataSize, 6, 6, s);
		//if (param.frameIdx == 1 && ix0 == 29 && iy0 == 2) printf("cuda %d %.14f\n", tidx, s[tidx]);

		//compute norm before starting inverse, s will be overwritten
		double ns = norm1(s, 6, 6, temp);
		//invert sd -> g [6 x 6]
		luinv(Apiv, s, temp, g, 6, r, ci, cols);
		//compute reciprocal condition, see if result is valid
		double ng = norm1(g, 6, 6, temp);
		double rcond = 1 / (ns * ng);
		result = (isnan(rcond) || rcond < d_core.deps) ? PointResultType::FAIL_SINGULAR : PointResultType::RUNNING;
		//if (param.frameIdx == 1 && ix0 == 97 && iy0 == 4 && cu::firstThread()) printf("cuda %d %.14f\n", z, rcond);

		//init loop limit counter
		int iter = 0;
		//init error measure to stop loop
		double bestErr = d_core.dmax;
		//main loop to find transformation of image patch
		while (result == PointResultType::RUNNING) {
			//interpolate image patch
			if (r < iw) {
				for (int c = ci; c < iw; c += cols) {
					double ix = xm + (c - ir) * wp[0] + (r - ir) * wp[3] + wp[2];
					double iy = ym + (c - ir) * wp[1] + (r - ir) * wp[4] + wp[5];

					//compute difference between image patches
					//store delta in transposed order [c * iw + r]
					if (ix < 0.0 || ix > wz - 1.0 || iy < 0.0 || iy > hz - 1.0) {
						delta[c * iw + r] = d_core.dnan;

					} else {
						double im = tex2D<float>(pyr0, xm - ir + c, rowOffset + ym + r - ir);

						double flx = floor(ix), fly = floor(iy);
						double dx = ix - flx, dy = iy - fly;
						int x0 = (int) flx, y0 = (int) fly;

						double f00 = tex2D<float>(pyr1, x0, rowOffset + y0);
						double f01 = tex2D<float>(pyr1, x0 + 1, rowOffset + y0);
						double f10 = tex2D<float>(pyr1, x0, rowOffset + y0 + 1);
						double f11 = tex2D<float>(pyr1, x0 + 1, rowOffset + y0 + 1);
						double jm = (1.0 - dx) * (1.0 - dy) * f00 + (1.0 - dx) * dy * f10 + dx * (1.0 - dy) * f01 + dx * dy * f11;

						delta[c * iw + r] = im - jm;
					}
				}
			}

			//eta = g.times(sd.times(delta.flatToCol())) [6 x 1]
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
			//if (param.frameIdx == 1 && ix0 == 63 && iy0 == 1 && cu::firstThread()) cu::storeDebugData(param.debugData, param.debugDataSize, 3, 3, wp);

			//analyse result, decide on continuing loop
			err = eta[0] * eta[0] + eta[1] * eta[1];
			if (isnan(err)) result = PointResultType::FAIL_ETA_NAN; //leave loop with fail message FAIL_ETA_NAN
			if (err < d_core.COMP_MAX_TOL) result = PointResultType::SUCCESS_ABSOLUTE_ERR; //leave loop with success SUCCESS_ABSOLUTE_ERR
			if (fabs(err - bestErr) / bestErr < d_core.COMP_MAX_TOL * d_core.COMP_MAX_TOL) result = PointResultType::SUCCESS_STABLE_ITER;
			bestErr = min(err, bestErr);
			iter++;
			if (iter == d_core.COMP_MAX_ITER && result == PointResultType::RUNNING) result = PointResultType::FAIL_ITERATIONS; //leave with fail
		}

		//displacement * 2 for next level
		if (r == 0 && ci == 0) wp[2] *= 2.0;
		if (r == 1 && ci == 0) wp[5] *= 2.0;

		//center of integration window on next level
		xm *= 2;
		ym *= 2;

		//if (param.frameIdx == 1 && ix0 == 63 && iy0 == 1 && cu::firstThread()) cu::storeDebugData(param.debugData, param.debugDataSize, 6, 1, eta);
		//if (param.frameIdx == 1 && ix0 == 97 && iy0 == 4 && cu::firstThread()) printf("cuda %d %.14f\n", z, wp[5]);
	}

	if (cu::firstThread()) {
		//final displacement vector
		double u = wp[2];
		double v = wp[5];
		int zp = z;

		//bring values to level 0
		while (z < 0) { xm /= 2; ym /= 2; u /= 2; v /= 2; z++; }
		while (z > 0) { xm *= 2; ym *= 2; u *= 2; v *= 2; z--; }

		//store results object
		int64_t timeStop = cu::globaltimer();
		results[blockIndex] = { timeStart, timeStop, u, v, xm, ym, blockIndex, ix0, iy0, result, zp, direction, true };
	}
}

void kernelComputeCall(ComputeKernelParam param, ComputeTextures& tex, CudaPointResult* d_results) {
	dim3 blk(param.blk.x, param.blk.y);
	dim3 thr(param.thr.x, param.thr.y);
	kernelCompute<<<blk, thr, param.shdBytes, param.stream>>> (tex, d_results, param);
}
