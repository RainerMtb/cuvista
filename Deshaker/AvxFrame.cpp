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

#include <immintrin.h>
#include "AvxFrame.hpp"
#include "Util.hpp"

std::ostream& operator << (std::ostream& os, __m512 v) {
	for (int i = 0; i < 16; i++) os << v.m512_f32[i] << " ";
	return os;
}

std::ostream& operator << (std::ostream& os, __m256 v) {
	for (int i = 0; i < 8; i++) os << v.m256_f32[i] << " ";
	return os;
}

AvxFrame::AvxFrame(MainData& data, MovieReader& reader, MovieWriter& writer) :
	MovieFrame(data, reader, writer),
	walign{ alignValue(data.w, 16) },
	mYUV(data.bufferCount),
	mPyr(data.pyramidCount, AvxMatFloat(data.pyramidRowCount, data.w, 0.0f)), 
	mYuvPlane(data.h, walign, 0.0f),
	mPrevOut({ AvxMatFloat(data.h, walign, data.bgcol_yuv.colors[0]), 
		AvxMatFloat(data.h, walign, data.bgcol_yuv.colors[1]), 
		AvxMatFloat(data.h, walign, data.bgcol_yuv.colors[2]) }),
	mOutBuffer({ AvxMatFloat(data.h, walign, data.bgcol_yuv.colors[0]),
		AvxMatFloat(data.h, walign, data.bgcol_yuv.colors[1]),
		AvxMatFloat(data.h, walign, data.bgcol_yuv.colors[2]) }),
	mFilterResult(data.h, walign),
	mFilterBuffer(data.w, walign) {}

std::string AvxFrame::getClassName() const {
	return "AVX 512: " + getCpuName();
}

std::string AvxFrame::getClassId() const {
	return "AVX 512";
}

void AvxFrame::inputData() {
	size_t idx = mBufferFrame.index % mYUV.size();
	mYUV[idx] = mBufferFrame;
}

float AvxFrame::sum(__m512 a, int from, int to) {
	float sum = 0.0f;
	for (int i = from; i < to; i++) sum += a.m512_f32[i];
	return sum;
}

float AvxFrame::sum(__m256 a, int from, int to) {
	float sum = 0.0f;
	for (int i = from; i < to; i++) sum += a.m256_f32[i];
	return sum;
}

void AvxFrame::filter(const float* srcptr, int h, int w, int stride, float* destptr, int destStride, __m512 ignore) {
	__m256 k = _mm256_setr_ps(0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f, 0.0f, 0.0f, 0.0f);
	int ks = 5;
	int s = 2;
	for (int r = 0; r < h; r++) {
		const float* row = srcptr + r * stride;
		float* dest = destptr + r;

		__m256 input;
		input = _mm256_setr_ps(row[0], row[0], row[0], row[1], row[2], 0, 0, 0);
		*dest = sum(_mm256_mul_ps(input, k), 0, ks);
		dest += destStride;

		input = _mm256_setr_ps(row[0], row[0], row[1], row[2], row[3], 0, 0, 0);
		*dest = sum(_mm256_mul_ps(input, k), 0, ks);
		dest += destStride;

		for (int c = s; c < w - s; c++) {
			input = _mm256_loadu_ps(row + c - s);
			*dest = sum(_mm256_mul_ps(input, k), 0, ks);
			dest += destStride;
		}

		input = _mm256_setr_ps(row[w - 4], row[w - 3], row[w - 2], row[w - 1], row[w - 1], 0, 0, 0);
		*dest = sum(_mm256_mul_ps(input, k), 0, ks);
		dest += destStride;

		input = _mm256_setr_ps(row[w - 3], row[w - 2], row[w - 1], row[w - 1], row[w - 1], 0, 0, 0);
		*dest = sum(_mm256_mul_ps(input, k), 0, ks);
		dest += destStride;
	}
}


//__m512i offset = _mm512_set1_epi32(3);
//__m512i idxmin = _mm512_set1_epi32(0);
//__m512i idxmax = _mm512_set1_epi32(15);
//
//float* AvxFrame::filterTriple(__m512i index, __m512 input, __m512 k, float* dest, int destStride) {
//	__m512 a = _mm512_permutexvar_ps(index, input);
//	__m512 result = _mm512_mul_ps(a, k);
//	*dest = sum(result, 0, 5);
//	dest += destStride;
//	*dest = sum(result, 5, 10);
//	dest += destStride;
//	*dest = sum(result, 10, 15);
//	return dest + destStride;
//}
//
//float* AvxFrame::filterVector(__m512i index, __m512 input, __m512 k, float* dest, int destStride) {
//	float* ptr = dest;
//	for (int i = 0; i < 4; i++) {
//		__m512i vindex = _mm512_max_epi32(index, idxmin);
//		vindex = _mm512_min_epi32(vindex, idxmax);
//		ptr = filterTriple(vindex, input, k, ptr, destStride);
//		index = _mm512_add_epi32(index, offset);
//	}
//	return ptr;
//}
//
//void AvxFrame::filter(const float* srcptr, int h, int w, int stride, float* destptr, int destStride, __m512 k) {
//	for (int r = 0; r < h; r++) {
//		const float* row = srcptr + r * stride;
//		float* dest = destptr + r;
//		__m512 input;
//		__m512i index;
//
//		input = _mm512_loadu_ps(row);
//		index = _mm512_setr_epi32(-2, -1, 0, 1, 2, -1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0);
//		dest = filterVector(index, input, k, dest, destStride);
//
//		for (int c = 10; c < w - 16; c += 12) {
//			input = _mm512_loadu_ps(row + c);
//			index = _mm512_setr_epi32(0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 0);
//			dest = filterVector(index, input, k, dest, destStride);
//		}
//
//		input = _mm512_loadu_ps(row + w - 16);
//		index = _mm512_setr_epi32(2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 0);
//		dest = destptr + r + destStride * (w - 12);
//		filterVector(index, input, k, dest, destStride);
//	}
//}

void AvxFrame::downsample(const float* srcptr, int h, int w, int stride, float* destptr, int destStride) {
	const int mask = 0b1011'0001;
	for (int r = 0; r < h - 1; r += 2) {
		for (int c = 0; c < w; c += 16) {
			const float* src = srcptr + r * stride + c;
			__m512 a = _mm512_loadu_ps(src);
			__m512 b = _mm512_loadu_ps(src + stride);
			__m512 f = _mm512_set1_ps(0.25f);

			a = _mm512_mul_ps(a, f);
			b = _mm512_mul_ps(b, f);
			__m512 result = _mm512_add_ps(a, b);
			__m512 ap = _mm512_permute_ps(a, mask);
			result = _mm512_add_ps(result, ap);
			__m512 bp = _mm512_permute_ps(b, mask);
			result = _mm512_add_ps(result, bp);
			__m512i index = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14);
			result = _mm512_permutexvar_ps(index, result);

			float* dest = destptr + r / 2 * destStride + c / 2;
			_mm512_mask_storeu_ps(dest, 0xFF, result);
		}
	}
}

//void AvxFrame::downsample(const float* srcptr, int h, int w, int stride, float* destptr, int destStride) {
//	const __m512i selector = _mm512_setr_epi32(0, 1, 0xF0, 0xF1, 2, 3, 0xF2, 0xF3, 4, 5, 0xF4, 0xF5, 6, 7, 0xF6, 0xF7);
//	for (int r = 0; r < h - 1; r += 2) {
//		for (int c = 0; c < w; c += 8) {
//			const float* src = srcptr + r * stride + c;
//			__m512 a = _mm512_maskz_loadu_ps(0xFF, src);
//			__m512 b = _mm512_maskz_loadu_ps(0xFF, src + stride);
//			__m512 input = _mm512_permutex2var_ps(a, selector, b);
//
//			__m512 f = _mm512_set1_ps(0.25f);
//			__m512 result = _mm512_mul_ps(input, f);
//
//			float* dest = destptr + r / 2 * destStride + c / 2;
//			dest[0] = result.m512_f32[0] + result.m512_f32[2] + result.m512_f32[1] + result.m512_f32[3];
//			dest[1] = result.m512_f32[4] + result.m512_f32[6] + result.m512_f32[5] + result.m512_f32[7];
//			dest[2] = result.m512_f32[8] + result.m512_f32[10] + result.m512_f32[9] + result.m512_f32[11];
//			dest[3] = result.m512_f32[12] + result.m512_f32[14] + result.m512_f32[13] + result.m512_f32[15];
//
//			//float b = _mm512_mask_reduce_add_ps(0b0000'0000'0000'1111, result); //will add in the order (0 + 2) + (1 + 3)
//		}
//	}
//}

void AvxFrame::interpolate(const AvxMatFloat& src, int h, int w, __m256 x, __m256 y, float* dest) {
	__m256 ps_zero = _mm256_set1_ps(0.0f);
	__m256 ps_one = _mm256_set1_ps(1.0f);
	__m256i epi32_one = _mm256_set1_epi32(1);
	__m256i idx;

	__mmask8 mask = 0xFF;
	__m256 check;
	check = _mm256_set1_ps(0.0f);
	mask &= _mm256_cmp_ps_mask(x, check, 13); //greater equal
	mask &= _mm256_cmp_ps_mask(y, check, 13); //greater equal
	check = _mm256_set1_ps(float(w - 1));
	mask &= _mm256_cmp_ps_mask(x, check, 2); //less equal
	check = _mm256_set1_ps(float(h - 1));
	mask &= _mm256_cmp_ps_mask(y, check, 2); //less equal

	__m256 flx = _mm256_floor_ps(x);
	__m256 fly = _mm256_floor_ps(y);
	__m256 dx = _mm256_sub_ps(x, flx);
	__m256 dy = _mm256_sub_ps(y, fly);
	__m256 dx1 = _mm256_sub_ps(ps_one, dx);
	__m256 dy1 = _mm256_sub_ps(ps_one, dy);

	__m256i ix = _mm256_cvtps_epi32(flx);
	__m256i iy = _mm256_cvtps_epi32(fly);
	__m256i stride = _mm256_set1_epi32(src.w());
	idx = _mm256_mullo_epi32(stride, iy);
	idx = _mm256_add_epi32(idx, ix);
	__m256 f00 = _mm256_mmask_i32gather_ps(ps_zero, mask, idx, src.data(), 4);

	__mmask8 maskdx = _mm256_cmp_ps_mask(dx, ps_zero, 4); //not equal
	__m256i ix1 = _mm256_mask_add_epi32(ix, maskdx, ix, epi32_one);
	idx = _mm256_mullo_epi32(stride, iy);
	idx = _mm256_add_epi32(idx, ix1);
	__m256 f01 = _mm256_mmask_i32gather_ps(ps_zero, mask, idx, src.data(), 4);

	__mmask8 maskdy = _mm256_cmp_ps_mask(dy, ps_zero, 4); //not equal
	__m256i iy1 = _mm256_mask_add_epi32(iy, maskdy, iy, epi32_one);
	idx = _mm256_mullo_epi32(stride, iy1);
	idx = _mm256_add_epi32(idx, ix);
	__m256 f10 = _mm256_mmask_i32gather_ps(ps_zero, mask, idx, src.data(), 4);

	__mmask8 maskdxdy = maskdx & maskdy;
	__m256i ix11 = _mm256_mask_add_epi32(ix, maskdxdy, ix, epi32_one);
	__m256i iy11 = _mm256_mask_add_epi32(iy, maskdxdy, iy, epi32_one);
	idx = _mm256_mullo_epi32(stride, iy11);
	idx = _mm256_add_epi32(idx, ix11);
	__m256 f11 = _mm256_mmask_i32gather_ps(ps_zero, mask, idx, src.data(), 4);

	__m256 r00, r01, r10, r11;
	r00 = _mm256_mul_ps(dx1, dy1);
	r00 = _mm256_mul_ps(r00, f00);
	r10 = _mm256_mul_ps(dx1, dy);
	r10 = _mm256_mul_ps(r10, f10);
	r01 = _mm256_mul_ps(dx, dy1);
	r01 = _mm256_mul_ps(r01, f01);
	r11 = _mm256_mul_ps(dx, dy);
	r11 = _mm256_mul_ps(r11, f11);

	__m256 result = r00;
	result = _mm256_add_ps(result, r10);
	result = _mm256_add_ps(result, r01);
	result = _mm256_add_ps(result, r11);

	_mm256_mask_storeu_ps(dest, mask, result);
}

void AvxFrame::yuvToFloat(const ImageYuv& yuv, size_t plane, AvxMatFloat& dest) {
	constexpr float f = 1.0f / 255.0f;
	const __m512 f16 = _mm512_set1_ps(f);
	for (int r = 0; r < mData.h; r++) {
		int c = 0;
		//handle blocks of 16 pixels
		for (; c < mData.w / 16 * 16; c += 16) {
			__m128i epi8 = _mm_loadu_epi8(yuv.addr(plane, r, c));
			__m512i epi32 = _mm512_cvtepu8_epi32(epi8);
			__m512 ps = _mm512_cvtepi32_ps(epi32);
			__m512 result = _mm512_mul_ps(ps, f16);
			_mm512_storeu_ps(dest.row(r) + c, result);
		}
		//image width may not align to 16, handle trailing pixels individually
		for (; c < mData.w; c++) {
			dest.at(r, c) = yuv.at(0, r, c) * f;
		}
	}
}

void AvxFrame::createPyramid(int64_t frameIndex) {
	//util::ConsoleTimer ic("avx pyramid");
	size_t pyrIdx = frameIndex % mPyr.size();
	AvxMatFloat& Y = mPyr[pyrIdx];

	//fill topmost level of pyramid
	size_t yuvIdx = frameIndex % mYUV.size();
	ImageYuv& yuv = mYUV[yuvIdx];
	yuvToFloat(yuv, 0, Y);

	//create pyramid levels below by downsampling level above
	int r = 0;
	int h = mData.h;
	int w = mData.w;
	for (size_t z = 1; z < mData.pyramidLevels; z++) {
		filter(Y.row(r), h, w, mData.w, mFilterBuffer.data(), mFilterBuffer.w(), filterKernels[0]);
		filter(mFilterBuffer.data(), w, h, mFilterBuffer.w(), mFilterResult.data(), mFilterResult.w(), filterKernels[0]);
		r += h;
		downsample(mFilterResult.data(), h, w, mFilterResult.w(), Y.row(r), Y.w());
		h /= 2;
		w /= 2;
		//if (z == 1) std::printf("avx %.14f\n", Y.at(r + 100, 100));
		//if (z == 1) mFilterBuffer.saveAsBinary("f:/filterAvx.dat");
	}
}

void AvxFrame::computeStart(int64_t frameIndex) {}

void AvxFrame::computeTerminate(int64_t frameIndex) {
	//TODO
}

void AvxFrame::outputData(const AffineTransform& trf, OutputContext outCtx) {
	size_t yuvidx = trf.frameIndex % mYUV.size();
	const ImageYuv& input = mYUV[yuvidx];
	assert(input.index == trf.frameIndex && "invalid frame index");
	int h = mData.h;
	int w = walign;
	std::array<double, 6> arr = trf.toArray();
	__m512d m00 = _mm512_set1_pd(arr[0]);
	__m512d m01 = _mm512_set1_pd(arr[1]);
	__m512d m02 = _mm512_set1_pd(arr[2]);
	__m512d m10 = _mm512_set1_pd(arr[3]);
	__m512d m11 = _mm512_set1_pd(arr[4]);
	__m512d m12 = _mm512_set1_pd(arr[5]);

	for (size_t z = 0; z < 3; z++) {
		yuvToFloat(input, z, mYuvPlane);

		for (int r = 0; r < h; r++) {
			for (int c = 0; c < w; c += 8) {
				__m256 bg = (mData.bgmode == BackgroundMode::COLOR ? _mm256_set1_ps(mData.bgcol_yuv.colors[z]) : _mm256_loadu_ps(mPrevOut[z].addr(r, c)));
				__m512d x = _mm512_setr_pd(c, c + 1, c + 2, c + 3, c + 4, c + 5, c + 6, c + 7);
				__m512d y = _mm512_set1_pd(r);

				__m512d xx = m02;
				xx = _mm512_fmadd_pd(y, m01, xx);
				xx = _mm512_fmadd_pd(x, m00, xx);
				__m256 xxps = _mm512_cvtpd_ps(xx);

				__m512d yy = m12;
				yy = _mm512_fmadd_pd(y, m11, yy);
				yy = _mm512_fmadd_pd(x, m10, yy);
				__m256 yyps = _mm512_cvtpd_ps(yy);

				interpolate(mYuvPlane, mData.h, mData.w, xxps, yyps, mOutBuffer[z].addr(r, c));
			}
		}
	}
}

Matf AvxFrame::getTransformedOutput() const {
	return Matf::concatVert(mOutBuffer[0], mOutBuffer[1], mOutBuffer[2]);
}

void AvxFrame::getTransformedOutput(int64_t frameIndex, ImagePPM& image) {
	//TODO
}

Matf AvxFrame::getPyramid(size_t idx) const {
	return mPyr[idx];
}

void AvxFrame::getInputFrame(int64_t frameIndex, ImagePPM& image) {
	size_t idx = frameIndex % mYUV.size();
	mYUV[idx].toPPM(image, mPool);
}

ImageYuv AvxFrame::getInput(int64_t index) const {
	return mYUV[index % mYUV.size()];
}
