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

using uint = unsigned int;

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

//list of possible results of one compute iteration
__constant__ PointResultType resultTypes[] = {
	PointResultType::RUNNING, 
	PointResultType::FAIL_ETA_NAN, 
	PointResultType::SUCCESS_ABSOLUTE_ERR, 
	PointResultType::SUCCESS_STABLE_ITER,
	PointResultType::FAIL_ITERATIONS,
};

//initial values for wp
__constant__ double wp0[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
//initial values for eta
__constant__ double eta0[] = { 0, 0, 1, 0, 0, 1 };

//parameter structure
__constant__ CoreData d_core;

//forward declaration to prevent underline
template<class T> __device__ T tex2D(cudaTextureObject_t tex, float x, float y);

//compute displacement
//one cuda block works one point in the image using one warp
__global__ void kernelCompute(ComputeTextures tex, PointResult* results, ComputeKernelParam param) {
	uint ix0 = blockIdx.x;
	uint iy0 = blockIdx.y;
	uint blockIndex = iy0 * gridDim.x + ix0;
	if (*param.d_interrupt || param.d_computed[blockIndex]) return;
	param.kernelTimestamps[blockIndex].start();

	int& ir = d_core.ir;
	int& iw = d_core.iw;

	//allocate individual variables in shared memory
	extern __shared__ double shd[];
	double* ptr = shd;
	double* sd = ptr;		ptr += 6 * iw * iw;  // 6 x iw*iw
	double* s = ptr;		ptr += 36;           // 6 x 6
	double* g = ptr;		ptr += 36;           // 6 x 6
	double* delta = ptr;	ptr += iw * iw;      // iw x iw
	double* wp = ptr;		ptr += 9;            // 3 x 3
	double* dwp = ptr;		ptr += 9;            // 3 x 3
	double* b = ptr;        ptr += 6;            // 6 doubles
	double* eta = ptr;      ptr += 6;            //eta 6 doubles
	double* temp = ptr;     ptr += 6;            //temp 6 doubles
	//array of double pointers to get rows in LU decomposition
	double** Apiv = (double**) (ptr);

	const int ci = threadIdx.x;	    //column into image
	const int cols = blockDim.x;	//columns that can be addressed in one warp
	const int r = threadIdx.y;		//row into image
	const int tidx = threadIdx.y * blockDim.x + threadIdx.x;

	//init wp and dwp to identitiy
	if (r < 3 && ci < 3) {
		dwp[r * 3 + ci] = wp[r * 3 + ci] = wp0[r * 3 + ci];
	}

	//center point of image patch in this block
	int ym = iy0 + ir;
	int xm = ix0 + ir;
	PointResultType result = PointResultType::RUNNING;

	//pyramid level to start at
	int z = d_core.zMax;
	//offset in rows to current pyramid level as texture spans one full pyramid
	int rowOffset = d_core.pyramidRowCount - (d_core.h >> z);

	for (; z >= d_core.zMin && result >= PointResultType::RUNNING; z--) {
		//dimensions for current pyramid level
		int wz = d_core.w >> z;
		int hz = d_core.h >> z;

		//build sd matrix [6 x iw*iw]
		if (r < iw) {
			for (int c = ci; c < iw; c += cols) {
				double x = tex2D<float>(tex.DXprev, xm - ir + r, rowOffset + ym - ir + c);
				double y = tex2D<float>(tex.DYprev, xm - ir + r, rowOffset + ym - ir + c);
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
		//if (frameIdx == 1 && ix0 == 63 && iy0 == 1 && cu::firstThread()) cu::storeDebugData(debugData, 6, 49, sd);

		//S = sd * sd' [6 x 6]
		//compute upper triangle and mirror value to write all values for S
		if (tidx < 21) {
			ArrayIndex ai = sidx[tidx]; //the value to compute in S
			double sval = 0.0;
			for (int i = 0; i < iw * iw; i++) {
				sval += sd[ai.r * iw * iw + i] * sd[ai.c * iw * iw + i];
			}
			//copy symmetric value
			s[ai.c * 6 + ai.r] = s[ai.r * 6 + ai.c] = sval;
		}
		//if (frameIdx == 1 && ix0 == 20 && iy0 == 20 && cu::firstThread()) cu::storeDebugData(debugData, 6, 6, s);

		//compute norm before starting inverse, s will be overwritten
		double ns = norm1(s, 6, 6, temp);
		//invert sd -> g [6 x 6]
		luinv(Apiv, s, temp, g, 6, r, ci, cols);
		//compute reciprocal condition, see if result is valid
		double ng = norm1(g, 6, 6, temp);
		double rcond = 1 / (ns * ng);
		result = (isnan(rcond) || rcond < d_core.deps) ? PointResultType::FAIL_SINGULAR : PointResultType::RUNNING;

		//if (frameIdx == 1 && ix0 == 63 && iy0 == 1 && cu::firstThread()) cu::storeDebugData(debugData, 6, 6, g);

		//init loop limit counter
		int iter = 0;
		//init error measure to stop loop
		double bestErr = d_core.dmax;
		//main loop to find transformation of image patch
		while (result == PointResultType::RUNNING) {
			//interpolate image patch
			if (r < iw) {
				for (int c = ci; c < iw; c += cols) {
					int x = c - ir;
					double ix = xm + x * wp[0] + (r - ir) * wp[3] + wp[2];
					double iy = ym + x * wp[1] + (r - ir) * wp[4] + wp[5];

					//compute difference between image patches
					//store delta in transposed order [c * iw + r]
					if (ix < 0.0 || ix > wz - 1.0 || iy < 0.0 || iy > hz - 1.0) {
						delta[c * iw + r] = d_core.dnan;

					} else {
						double im = tex2D<float>(tex.Yprev, xm - ir + c, rowOffset + ym + r - ir);

						double flx = floor(ix), fly = floor(iy);
						double dx = ix - flx, dy = iy - fly;
						int x0 = (int) flx, y0 = (int) fly;

						double f00 = tex2D<float>(tex.Ycur, x0, rowOffset + y0);
						double f01 = tex2D<float>(tex.Ycur, x0 + 1, rowOffset + y0);
						double f10 = tex2D<float>(tex.Ycur, x0, rowOffset + y0 + 1);
						double f11 = tex2D<float>(tex.Ycur, x0 + 1, rowOffset + y0 + 1);
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
				dwp[r * 3 + 0] = wp[r * 3] * eta[2] + wp[r * 3 + 1] * eta[4];
				dwp[r * 3 + 1] = wp[r * 3] * eta[3] + wp[r * 3 + 1] * eta[5];
				dwp[r * 3 + 2] = wp[r * 3] * eta[0] + wp[r * 3 + 1] * eta[1] + wp[r * 3 + 2];

				//update wp
				wp[r * 3 + 0] = dwp[r * 3];
				wp[r * 3 + 1] = dwp[r * 3 + 1];
				wp[r * 3 + 2] = dwp[r * 3 + 2];
			}
			//if (frameIdx == 1 && ix0 == 27 && iy0 == 1 && cu::firstThread()) cu::storeDebugData(debugData, 3, 3, wp); //--------------------------

			//analyse result, decide on continuing loop
			double err = eta[0] * eta[0] + eta[1] * eta[1];
			int typeIndex = 0;
			typeIndex += (int) isnan(err) * 1; //leave loop with fail message FAIL_ETA_NAN
			typeIndex += (int) (err < d_core.compMaxTol) * 2; //leave loop with success SUCCESS_ABSOLUTE_ERR
			typeIndex += (int) (fabs(err - bestErr) / bestErr < d_core.compMaxTol * d_core.compMaxTol) * 3; //SUCCESS_STABLE_ITER
			result = resultTypes[typeIndex];

			bestErr = min(err, bestErr);
			iter++;

			//typeIndex = (int) (iter == d_core.compMaxIter && result == PointResultType::RUNNING) * 4; //FAIL_ITERATIONS
			//result = resultTypes[typeIndex];
			if (iter == d_core.compMaxIter && result == PointResultType::RUNNING) {
				result = PointResultType::FAIL_ITERATIONS; //leave with fail
			}
		}
		//if (frameIdx == 1 && ix0 == 63 && iy0 == 1 && cu::firstThread()) cu::storeDebugData(debugData, 6, 1, eta);

		//displacement * 2 for next level
		if (r == 0 && ci == 0) wp[2] *= 2.0;
		if (r == 1 && ci == 0) wp[5] *= 2.0;

		//center of integration window on next level
		xm *= 2;
		ym *= 2;

		//new texture row offset
		int delta = d_core.h >> (z - 1);
		rowOffset -= delta;
	}

	if (cu::firstThread()) {
		//final displacement vector
		double u = wp[2];
		double v = wp[5];
		//bring values to level 0
		while (z < 0) { xm /= 2; ym /= 2; u /= 2; v /= 2; z++; }
		while (z > 0) { xm *= 2; ym *= 2; u *= 2; v *= 2; z--; }
		//index into results array
		size_t idx = iy0 * gridDim.x + ix0;
		//store results object
		results[idx] = { idx, ix0, iy0, xm, ym, xm - d_core.w / 2, ym - d_core.h / 2, u, v, result };
	}

	param.kernelTimestamps[blockIndex].stop();
	param.d_computed[blockIndex] = 1;
}

void kernelComputeCall(ComputeKernelParam param, ComputeTextures& tex, PointResult* d_results) {
	dim3 blk(param.blk.x, param.blk.y);
	dim3 thr(param.thr.x, param.thr.y);
	kernelCompute<<<blk, thr, param.shdBytes, param.stream>>> (tex, d_results, param);
}

void computeInit(const CoreData& core) {
	//copy core struct to device
	const void* ptr = &d_core;
	cudaMemcpyToSymbol(ptr, &core, sizeof(core));
}
