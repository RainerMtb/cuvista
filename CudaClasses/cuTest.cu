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

#include "cuTest.cuh"

//kernel for testing lu decomposition in single warp
__global__ void inv_kernel(double* input, double* outAinv, size_t s) {
	int c = threadIdx.x / s;
	int cols = blockDim.x / s;
	int r = threadIdx.x - c * s;

	extern __shared__ double mat[];
	double* ptr = mat;
	double* temp = ptr; ptr += s;
	double** Apiv = (double**) ptr;

	luinv(Apiv, input, temp, outAinv, s, r, c, cols);
}

//kernel for testing lu decomposition in multiple warps
__global__ void inv_kernel_parallel(double* input, double* output, size_t s) {
	extern __shared__ double mat[];
	double* temp = mat;
	double** Apiv = (double**) (mat + s);

	input += blockIdx.x * s * s;
	output += blockIdx.x * s * s;
	luinv(Apiv, input, temp, output, s, threadIdx.y, threadIdx.x, blockDim.x);
}

__global__ void norm1_kernel(double* input, double* result, int s) {
	extern __shared__ double temp[];
	*result = norm1(input, s, s, temp);
}

template<class T> __device__ T tex1D(cudaTextureObject_t tex, float x);

//kernel to actually read from texture
__global__ void kernelTex1(cudaTextureObject_t in, float dx, float* out) {
	*out = tex1D<float>(in, dx);
}


//---------------------------------
// host code for kernel callers
//---------------------------------

//test lu decomposition
bool cutest::cudaInv(double* input, double* inv, size_t s) {
	double* d_input;
	double* d_inv;
	size_t siz = sizeof(double) * s * s;
	cudaMalloc(&d_input, siz);
	cudaMalloc(&d_inv, siz);
	cudaMemcpy(d_input, input, siz, cudaMemcpyDefault);

	int warpSize;
	cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, 0);
	dim3 thr(warpSize, 1, 1);
	inv_kernel <<<1, thr, s * 2 * sizeof(double)>>> (d_input, d_inv, s);

	cudaMemcpy(inv, d_inv, siz, cudaMemcpyDefault);
	cudaFree(d_input);
	cudaFree(d_inv);

	cudaError_t status = cudaGetLastError();
	if (status != 0) printf("cuda error #2: %s\n", cudaGetErrorString(status));
	return status == 0;
}

bool cutest::invParallel_6(double* input, double* inv, int count) {
	double* d_input;
	double* d_inv;
	int s = 6;
	int siz = sizeof(double) * s * s * count;
	cudaMalloc(&d_input, siz);
	cudaMalloc(&d_inv, siz);
	cudaMemcpy(d_input, input, siz, cudaMemcpyDefault);

	//dim3 thr(s / 2, s);
	dim3 thr(4, 7);
	dim3 blk(count, 1);
	inv_kernel_parallel <<<blk, thr, s * 2 * sizeof(double)>>> (d_input, d_inv, s);

	cudaMemcpy(inv, d_inv, siz, cudaMemcpyDefault);
	cudaFree(d_input);
	cudaFree(d_inv);

	cudaError_t status = cudaGetLastError();
	if (status != 0) printf("cuda error #2: %s\n", cudaGetErrorString(status));
	return status == 0;
}

double cutest::norm1(double* input, size_t s, int threads) {
	double* d_input;
	double* d_result;
	size_t siz = sizeof(double) * s * s;
	cudaMalloc(&d_input, siz);
	cudaMalloc(&d_result, sizeof(double));
	cudaMemcpy(d_input, input, siz, cudaMemcpyDefault);

	dim3 thr(1, threads);
	int si = (int) s;
	norm1_kernel <<<1, thr, s * sizeof(double)>>> (d_input, d_result, si);

	double result;
	cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDefault);
	cudaFree(d_input);
	cudaFree(d_result);
	cudaError_t status = cudaGetLastError();
	if (status != 0) printf("cuda error #2: %s\n", cudaGetErrorString(status));
	return result;
}

float cutest::textureInterpolation(float f0, float f1, float dx) {
	//set up texture
	float data[] = { f0, f1 };
	size_t siz = 2 * sizeof(float);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray_t d_array;
	cudaMallocArray(&d_array, &channelDesc, 2, 1);
	cudaMemcpy2DToArray(d_array, 0, 0, data, siz, siz, 1, cudaMemcpyDefault);

	cudaResourceDesc resDesc {};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = d_array;

	// Specify texture object parameters
	cudaTextureDesc texDesc {};
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = false;

	// Create texture object
	cudaTextureObject_t texObj;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

	//read from texture
	float* d_result;
	cudaMalloc(&d_result, sizeof(float));
	kernelTex1 <<<1, 1>>> (texObj, dx + 0.5f, d_result);
	float result;
	cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDefault);
	cudaFree(d_result);
	return result;
}
