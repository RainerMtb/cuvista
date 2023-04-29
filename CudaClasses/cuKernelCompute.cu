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

__constant__ double wp0[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
__constant__ double eta0[] = { 0, 0, 1, 0, 0, 1 };
__constant__ CoreData d_core;

template<class T> __device__ T tex2D(cudaTextureObject_t tex, float x, float y);

//compute displacement
//one block works one point in the image
__global__ void kernelCompute(ComputeTextures tex, PointResult* results, int64_t frameIdx, cu::DebugData debugData) {
	//size_t i = (blockIdx.y * gridDim.x + blockIdx.x) * timerCount; KERNEL_TIME(timestamps, i++);
	int& ir = d_core.ir;
	int& iw = d_core.iw;

	//allocate individual variables in shared memory
	extern __shared__ double shd[];
	double* ptr = shd;
	double* sd = ptr;		ptr += 6 * iw * iw; // 6 x iw*iw
	double* s = ptr;		ptr += 36; // 6 x 6
	double* g = ptr;		ptr += 36; // 6 x 6
	double* im = ptr;		ptr += iw * iw; // iw x iw
	double* jm = ptr;		ptr += iw * iw; // iw x iw
	double* delta = ptr;	ptr += iw * iw; // iw x iw
	double* wp = ptr;		ptr += 9; // 3 x 3
	double* dwp = ptr;		ptr += 9; // 3 x 3
	double* b = ptr;        ptr += 6; //b 6 doubles
	double* eta = ptr;      ptr += 6; //eta 6 doubles
	double* temp = ptr;     ptr += 6; //temp 6 doubles
	//array of double pointers to get rows in LU decomposition
	double** Apiv = (double**) (ptr); ptr += 6;

	const uint ci = threadIdx.x;	//0..warpSize/iw    column into image
	const uint cols = blockDim.x;	//columns that can be addressed in one warp
	const uint r = threadIdx.y;		//0..iw   row into image
	const int rir = r - ir;

	//init wp and dwp to identitiy
	if (r < 3 && ci < 3) {
		dwp[r * 3 + ci] = wp[r * 3 + ci] = wp0[r * 3 + ci];
	}

	uint ix0 = blockIdx.x;
	uint iy0 = blockIdx.y;
	//center point of image patch in this block
	int ym = iy0 + ir;
	int xm = ix0 + ir;
	PointResultType result = PointResultType::RUNNING;

	int z = d_core.zMax;
	int rowOffset = 0; //offset in rows to current pyramid level as texture spans one full pyramid
	int hh = d_core.h;
	for (int i = 0; i < z; i++) {
		rowOffset += hh;
		hh /= 2;
	}
	for (; z >= d_core.zMin && result >= PointResultType::RUNNING; z--) {
		//dimensions for current pyramid level
		int wz = d_core.w >> z;
		int hz = d_core.h >> z;

		if (r < iw) {
			for (int c = ci; c < iw; c += cols) {
				//copy area of interest from previous frame
				im[r * iw + c] = tex2D<float>(tex.Yprev, xm - ir + c, rowOffset + ym + rir);

				//build sd matrix [6 x iw*iw]
				double x = tex2D<float>(tex.DXprev, xm + rir, rowOffset + ym - ir + c);
				double y = tex2D<float>(tex.DYprev, xm + rir, rowOffset + ym - ir + c);
				int idx = r * iw + c;
				sd[idx] = x;				
				idx += iw * iw;
				sd[idx] = y;				
				idx += iw * iw;
				sd[idx] = x * rir;			
				idx += iw * iw;
				sd[idx] = y * rir;			
				idx += iw * iw;
				sd[idx] = x * (c - ir);		
				idx += iw * iw;
				sd[idx] = y * (c - ir);
			}
		}
		//if (frameIdx == 1 && ix0 == 63 && iy0 == 1 && cu::firstThread()) cu::storeDebugData(debugData, 6, 49, sd);

		//S = sd * sd' [6 x 6]
		if (r < 6) {
			for (int c = r + ci; c < 6; c += cols) {
				//compute only upper triangle
				s[r * 6 + c] = 0.0;
				for (int i = 0; i < iw * iw; i++) {
					s[r * 6 + c] += sd[r * iw * iw + i] * sd[c * iw * iw + i];
				}
				//copy symmetric value
				s[c * 6 + r] = s[r * 6 + c];
			}
		}
		//if (frameIdx == 1 && ix0 == 63 && iy0 == 1 && cu::firstThread()) cu::storeDebugData(debugData, 6, 6, s);
		//KERNEL_TIME(timestamps, i++);

		//compute norm before starting inverse, s will be overwritten
		double ns = norm1(s, 6, 6, temp);
		//invert sd -> g [6 x 6]
		luinv(Apiv, s, temp, g, 6, r, ci, cols);
		//compute reciprocal condition, see if result is valid
		double ng = norm1(g, 6, 6, temp);
		double rcond = 1 / (ns * ng);
		result = (isnan(rcond) || rcond < d_core.deps) ? PointResultType::FAIL_SINGULAR : PointResultType::RUNNING;

		//if (frameIdx == 1 && ix0 == 63 && iy0 == 1 && cu::firstThread()) cu::storeDebugData(debugData, 6, 6, g); //----------------------------
		//KERNEL_TIME(timestamps, i++);

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
					double ix = xm + x * wp[0] + rir * wp[3] + wp[2];
					double iy = ym + x * wp[1] + rir * wp[4] + wp[5];

					if (ix < 0.0 || ix > wz - 1.0 || iy < 0.0 || iy > hz - 1.0) {
						jm[r * iw + c] = d_core.dnan;

					} else {
						double flx = floor(ix), fly = floor(iy);
						double dx = ix - flx, dy = iy - fly;
						int x0 = (int) flx, y0 = (int) fly;

						double f00 = tex2D<float>(tex.Ycur, x0, rowOffset + y0);
						double f01 = tex2D<float>(tex.Ycur, x0 + 1, rowOffset + y0);
						double f10 = tex2D<float>(tex.Ycur, x0, rowOffset + y0 + 1);
						double f11 = tex2D<float>(tex.Ycur, x0 + 1, rowOffset + y0 + 1);
						jm[r * iw + c] = (1.0 - dx) * (1.0 - dy) * f00 + (1.0 - dx) * dy * f10 + dx * (1.0 - dy) * f01 + dx * dy * f11;
					}

				}
			}

			//compute difference between image patches
			if (r < iw) {
				for (int c = ci; c < iw; c += cols) {
					delta[c * iw + r] = im[r * iw + c] - jm[r * iw + c]; //store delta in transposed order
				}
			}

			//eta = g.times(sd.times(delta.flatToCol())) [6 x 1]
			if (r < 6 && ci == 0) {
				//init eta to [0 0 1 0 0 1]
				eta[r] = eta0[r];
				//init b to [0 0 0 0 0 0]
				b[r] = 0.0;
				//sd * delta
				for (double* sdptr = sd + r * iw * iw, *deltaptr = delta; deltaptr != delta + iw * iw; sdptr++, deltaptr++) {
					b[r] += (*sdptr) * (*deltaptr);
				}
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
			if (isnan(err)) result = PointResultType::FAIL_ETA_NAN; //leave loop with fail
			if (err < d_core.compMaxTol) result = PointResultType::SUCCESS_ABSOLUTE_ERR; //leave loop with success
			if (fabs(err - bestErr) / bestErr < d_core.compMaxTol * d_core.compMaxTol) result = PointResultType::SUCCESS_STABLE_ITER; //leave with success
			if (err < bestErr) bestErr = err;
			iter++;
			if (iter == d_core.compMaxIter && result == PointResultType::RUNNING) result = PointResultType::FAIL_ITERATIONS; //leave with fail
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
}

void kernelComputeCall(kernelParam param, ComputeTextures& tex, PointResult* d_results, int64_t frameIdx, cu::DebugData debugData) {
	kernelCompute <<<param.blk, param.thr, param.shdBytes, param.stream>>> (tex, d_results, frameIdx, debugData);
}

void computeInit(const CoreData& core) {
	//copy core struct to device
	const void* ptr = &d_core;
	cudaMemcpyToSymbol(ptr, &core, sizeof(core));
}
