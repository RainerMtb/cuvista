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

 std::ostream& operator << (std::ostream& os, __m512d v) {
 	for (int i = 0; i < 8; i++) os << v.m512d_f64[i] << " ";
 	return os;
 }
 
 std::ostream& operator << (std::ostream& os, __m256 v) {
 	for (int i = 0; i < 8; i++) os << v.m256_f32[i] << " ";
 	return os;
 }

AvxFrame::AvxFrame(MainData& data, MovieReader& reader, MovieWriter& writer) :
	MovieFrame(data, reader, writer),
	pitch { alignValue(data.w, walign) } 
{
	mYUV.resize(data.bufferCount);
	mPyr.assign(data.pyramidCount, AvxMatFloat(data.pyramidRowCount, data.w, 0.0f));
	mYuvPlane = AvxMatFloat(data.h, pitch, 0.0f);

	mWarped.push_back(AvxMatFloat(data.h, pitch, data.bgcol_yuv.colors[0]));
	mWarped.push_back(AvxMatFloat(data.h, pitch, data.bgcol_yuv.colors[1]));
	mWarped.push_back(AvxMatFloat(data.h, pitch, data.bgcol_yuv.colors[2]));

	mFilterResult = AvxMatFloat(data.h, pitch);
	mFilterBuffer = AvxMatFloat(data.w, pitch);
}

std::string AvxFrame::getClassName() const {
	return "AVX 512: " + mData.getCpuName();
}

std::string AvxFrame::getClassId() const {
	return "AVX 512";
}

void AvxFrame::inputData() {
	size_t idx = mBufferFrame.index % mYUV.size();
	mYUV[idx] = mBufferFrame;
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
		filter(Y, r, h, w, mFilterBuffer, filterKernels[0]);
		mFilterResult.fill(0.0f);
		filter(mFilterBuffer, 0, w, h, mFilterResult, filterKernels[0]);
		r += h;
		downsample(mFilterResult.data(), h, w, mFilterResult.w(), Y.row(r), Y.w());
		h /= 2;
		w /= 2;
		//if (z == 1) std::printf("avx %.14f\n", Y.at(r + 100, 100));
		//if (z == 3) mFilterBuffer.saveAsBinary("f:/filterAvx.dat");
	}
}

void AvxFrame::outputData(const AffineTransform& trf, OutputContext outCtx) {
	util::ConsoleTimer ic("avx output");
	size_t yuvidx = trf.frameIndex % mYUV.size();
	const ImageYuv& input = mYUV[yuvidx];
	assert(input.index == trf.frameIndex && "invalid frame index");

	//for planes Y, U, V
	for (size_t z = 0; z < 3; z++) {
		yuvToFloat(input, z, mYuvPlane);
		if (mData.bgmode == BackgroundMode::COLOR) mWarped[z].fill(mData.bgcol_yuv.colors[z]);
		warpBack(trf, mYuvPlane, mWarped[z]);
		filter(mWarped[z], 0, mData.h, mData.w, mFilterBuffer, filterKernels[z]);
		filter(mFilterBuffer, 0, mData.w, mData.h, mFilterResult, filterKernels[z]);
		unsharpAndWrite(mWarped[z], mFilterResult, mData.unsharp[z], outCtx.outputFrame, z);
	}

	outCtx.outputFrame->index = trf.frameIndex;
	//outCtx.outputFrame->saveAsBMP("f:/out.bmp");

	//copy input if requested
	if (outCtx.requestInput) {
		*outCtx.inputFrame = input;
		outCtx.inputFrame->index = trf.frameIndex;
	}
	//send to cuda for encoding if requested
	if (outCtx.encodeCuda) {
		static std::vector<unsigned char> nv12(outCtx.cudaPitch * mData.h * 3 / 2);
		outCtx.outputFrame->toNV12(nv12, outCtx.cudaPitch);
		encodeNvData(nv12, outCtx.cudaNv12ptr);
	}
}

Matf AvxFrame::getTransformedOutput() const {
	return Matf::concatVert(mWarped[0], mWarped[1], mWarped[2]);
}

void AvxFrame::getTransformedOutput(int64_t frameIndex, ImagePPM& image) {
	yuvToRgb(mWarped[0].data(), mWarped[1].data(), mWarped[2].data(), mData.h, mData.w, pitch, image);
}

Matf AvxFrame::getPyramid(size_t idx) const {
	return mPyr[idx];
}

void AvxFrame::getInput(int64_t frameIndex, ImagePPM& image) {
	size_t idx = frameIndex % mYUV.size();
	ImageYuv& yuv = mYUV[idx];
	yuvToRgb(yuv.plane(0), yuv.plane(1), yuv.plane(2), mData.h, mData.w, yuv.stride, image);
}

ImageYuv AvxFrame::getInput(int64_t index) const {
	return mYUV[index % mYUV.size()];
}


//----------------------------------------------------
// ----- IMPLEMENTATION DETAILS ----------------------
//----------------------------------------------------


float* AvxFrame::filterTriple(__m512i index, Vecf input, Vecf k, float* dest, int destStride) {
	Vecf a = _mm512_permutexvar_ps(index, input.a);
	Vecf result = a * k;
	*dest = result.sum(0, 5);
	dest += destStride;
	*dest = result.sum(5, 10);
	dest += destStride;
	*dest = result.sum(10, 15);
	return dest + destStride;
}

float* AvxFrame::filterVector(__m512i index, Vecf input, Vecf k, float* dest, int destStride) {
	__m512i offset = _mm512_set1_epi32(3);
	__m512i idxmin = _mm512_set1_epi32(0);
	__m512i idxmax = _mm512_set1_epi32(15);
	float* ptr = dest;
	for (int i = 0; i < 4; i++) {
		__m512i vindex = _mm512_max_epi32(index, idxmin);
		vindex = _mm512_min_epi32(vindex, idxmax);
		ptr = filterTriple(vindex, input, k, ptr, destStride);
		index = _mm512_add_epi32(index, offset);
	}
	return ptr;
}

//float sum(__m256 v, int from, int to) {
//	float s = 0.0f;
//	for (int i = from; i < to; i++) s += v.m256_f32[i];
//	return s;
//}
//
//void AvxFrame::filter(const AvxMatFloat& src, int r0, int h, int w, AvxMatFloat& destMat, Vecf ignore) {
//	__m256 k = _mm256_setr_ps(0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f, 0.0f, 0.0f, 0.0f);
//	int ks = 5;
//	int s = 2;
//	int destStride = destMat.w();
//
//	for (int r = 0; r < h; r++) {
//		const float* row = src.addr(r, 0);
//		float* dest = destMat.addr(0, r);
//
//		__m256 input;
//		input = _mm256_setr_ps(row[0], row[0], row[0], row[1], row[2], 0, 0, 0);
//		*dest = sum(_mm256_mul_ps(input, k), 0, ks);
//		dest += destStride;
//
//		input = _mm256_setr_ps(row[0], row[0], row[1], row[2], row[3], 0, 0, 0);
//		*dest = sum(_mm256_mul_ps(input, k), 0, ks);
//		dest += destStride;
//
//		for (int c = s; c < w - s; c++) {
//			input = _mm256_loadu_ps(row + c - s);
//			*dest = sum(_mm256_mul_ps(input, k), 0, ks);
//			dest += destStride;
//		}
//
//		input = _mm256_setr_ps(row[w - 4], row[w - 3], row[w - 2], row[w - 1], row[w - 1], 0, 0, 0);
//		*dest = sum(_mm256_mul_ps(input, k), 0, ks);
//		dest += destStride;
//
//		input = _mm256_setr_ps(row[w - 3], row[w - 2], row[w - 1], row[w - 1], row[w - 1], 0, 0, 0);
//		*dest = sum(_mm256_mul_ps(input, k), 0, ks);
//		dest += destStride;
//	}
//}

void AvxFrame::filter(const AvxMatFloat& src, int r0, int h, int w, AvxMatFloat& destMat, Vecf k) {
	for (int r = 0; r < h; r++) {
		const float* row = src.addr(r0 + r, 0);
		float* dest = destMat.addr(0, r);
		__m512 input;
		__m512i index;

		input = _mm512_loadu_ps(row);
		index = _mm512_setr_epi32(-2, -1, 0, 1, 2, -1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0);
		dest = filterVector(index, input, k, dest, destMat.w());

		for (int c = 10; c < w - 16; c += 12) {
			input = _mm512_loadu_ps(row + c);
			index = _mm512_setr_epi32(0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 0);
			dest = filterVector(index, input, k, dest, destMat.w());
		}

		input = _mm512_loadu_ps(row + w - 16);
		index = _mm512_setr_epi32(2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 0);
		dest = destMat.addr(w - 12, r);
		filterVector(index, input, k, dest, destMat.w());
	}
}

void AvxFrame::downsample(const float* srcptr, int h, int w, int stride, float* destptr, int destStride) {
	__m512i idx1 = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
	__m512i idx2 = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
	__m512 f = _mm512_set1_ps(0.5f);

	for (int r = 0; r < h - 1; r += 2) {
		for (int c = 0; c < w; c += 32) {
			const float* src = srcptr + r * stride + c;
			__m512 x1 = _mm512_loadu_ps(src);
			__m512 x2 = _mm512_loadu_ps(src + 16);
			__m512 y1 = _mm512_loadu_ps(src + stride);
			__m512 y2 = _mm512_loadu_ps(src + stride + 16);

			__m512 f00 = _mm512_permutex2var_ps(x1, idx1, x2);
			__m512 f01 = _mm512_permutex2var_ps(x1, idx2, x2);
			__m512 f10 = _mm512_permutex2var_ps(y1, idx1, y2);
			__m512 f11 = _mm512_permutex2var_ps(y1, idx2, y2);
			Vecf result = interpolate(f00, f10, f01, f11, f, f, f, f);

			float* dest = destptr + r / 2 * destStride + c / 2;
			result.storeu(dest);
		}
	}
}

void AvxFrame::yuvToFloat(const ImageYuv& yuv, size_t plane, AvxMatFloat& dest) {
	constexpr float f = 1.0f / 255.0f;
	for (int r = 0; r < mData.h; r++) {
		int c = 0;
		//handle blocks of 16 pixels
		for (; c < mData.w / 16 * 16; c += 16) {
			Vecf result = yuv.addr(plane, r, c);
			result = result * f;
			result.storeu(dest.row(r) + c);
		}
		//image width may not align to 16, handle trailing pixels individually
		for (; c < mData.w; c++) {
			dest.at(r, c) = yuv.at(0, r, c) * f;
		}
	}
}

void AvxFrame::computeStart(int64_t frameIndex) {}

void AvxFrame::computeTerminate(int64_t frameIndex) {
	//TODO
}

//compute bilinear interpolation
//result = (1 - dx) * (1 - dy) * f00 + (1 - dx) * dy * f10 + dx * (1 - dy) * f01 + dx * dy * f11;
Vecf AvxFrame::interpolate(Vecf f00, Vecf f10, Vecf f01, Vecf f11, Vecf dx, Vecf dy) {
	Vecf one = 1.0f;
	return interpolate(f00, f10, f01, f11, dx, dy, one - dx, one - dy);
}

//compute bilinear interpolation
Vecf AvxFrame::interpolate(Vecf f00, Vecf f10, Vecf f01, Vecf f11, Vecf dx, Vecf dy, Vecf dx1, Vecf dy1) {
	return dx1 * dy1 * f00 + dx1 * dy * f10 + dx * dy1 * f01 + dx * dy * f11;
}

std::pair<__m512d, __m512d> AvxFrame::transform(__m512d x, __m512d y, __m512d m00, __m512d m01, __m512d m02, __m512d m10, __m512d m11, __m512d m12) {
	__m512d xx = m02;
	xx = _mm512_fmadd_pd(y, m01, xx);
	xx = _mm512_fmadd_pd(x, m00, xx);
	__m512d yy = m12;
	yy = _mm512_fmadd_pd(y, m11, yy);
	yy = _mm512_fmadd_pd(x, m10, yy);
	return { xx, yy };
}

void AvxFrame::warpBack(const AffineTransform& trf, const AvxMatFloat& input, AvxMatFloat& dest) {
	//transform parameters
	__m512d m00 = _mm512_set1_pd(trf.arrayValue(0));
	__m512d m01 = _mm512_set1_pd(trf.arrayValue(1));
	__m512d m02 = _mm512_set1_pd(trf.arrayValue(2));
	__m512d m10 = _mm512_set1_pd(trf.arrayValue(3));
	__m512d m11 = _mm512_set1_pd(trf.arrayValue(4));
	__m512d m12 = _mm512_set1_pd(trf.arrayValue(5));

	__m512d offset = _mm512_set1_pd(8);
	__m512i selector = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);

	for (int r = 0; r < mData.h; r++) {
		__m512d iy = _mm512_set1_pd(r);
		__m512d ix = _mm512_setr_pd(0, 1, 2, 3, 4, 5, 6, 7);
		for (int c = 0; c < pitch; c += 16) {

			auto t1 = transform(ix, iy, m00, m01, m02, m10, m11, m12);
			ix = _mm512_add_pd(ix, offset);
			__m512 xx1 = _mm512_castps256_ps512(_mm512_cvtpd_ps(t1.first));
			__m512 yy1 = _mm512_castps256_ps512(_mm512_cvtpd_ps(t1.second));

			auto t2 = transform(ix, iy, m00, m01, m02, m10, m11, m12);
			ix = _mm512_add_pd(ix, offset);
			__m512 xx2 = _mm512_castps256_ps512(_mm512_cvtpd_ps(t2.first));
			__m512 yy2 = _mm512_castps256_ps512(_mm512_cvtpd_ps(t2.second));

			__m512 x = _mm512_permutex2var_ps(xx1, selector, xx2);
			__m512 y = _mm512_permutex2var_ps(yy1, selector, yy2);

			__m512 ps_zero = _mm512_set1_ps(0.0f);
			__m512 ps_one = _mm512_set1_ps(1.0f);
			__m512i epi_one = _mm512_set1_epi32(1);
			__m512i epi_stride = _mm512_set1_epi32(input.w());
			__m512i idx;

			//check within image bounds
			__mmask16 mask = 0xFFFF;
			__m512 check;
			check = _mm512_set1_ps(0.0f);
			mask &= _mm512_cmp_ps_mask(x, check, _CMP_GE_OS); //greater equal
			mask &= _mm512_cmp_ps_mask(y, check, _CMP_GE_OS); //greater equal
			check = _mm512_set1_ps(float(input.w() - 1));
			mask &= _mm512_cmp_ps_mask(x, check, _CMP_LE_OS); //less equal
			check = _mm512_set1_ps(float(input.h() - 1));
			mask &= _mm512_cmp_ps_mask(y, check, _CMP_LE_OS); //less equal

			//compute fractions
			__m512 flx = _mm512_floor_ps(x);
			__m512 fly = _mm512_floor_ps(y);
			__m512 dx = _mm512_sub_ps(x, flx);
			__m512 dy = _mm512_sub_ps(y, fly);
			__m512 dx1 = _mm512_sub_ps(ps_one, dx);
			__m512 dy1 = _mm512_sub_ps(ps_one, dy);

			//compute index to load f00
			__m512i ix = _mm512_cvtps_epi32(flx);
			__m512i iy = _mm512_cvtps_epi32(fly);
			idx = _mm512_mullo_epi32(epi_stride, iy);
			idx = _mm512_add_epi32(idx, ix);
			__m512 f00 = _mm512_mask_i32gather_ps(ps_zero, mask, idx, input.data(), 4);

			//compute index to load f01
			__mmask16 maskdx = _mm512_cmp_ps_mask(dx, ps_zero, _CMP_NEQ_OS); //not equal
			__m512i idx2 = _mm512_mask_add_epi32(idx, maskdx, idx, epi_one);
			__m512 f01 = _mm512_mask_i32gather_ps(ps_zero, mask, idx2, input.data(), 4);

			//compute index to load f10
			__mmask16 maskdy = _mm512_cmp_ps_mask(dy, ps_zero, _CMP_NEQ_OS); //not equal
			__m512i idx3 = _mm512_mask_add_epi32(idx, maskdy, idx, epi_stride);
			__m512 f10 = _mm512_mask_i32gather_ps(ps_zero, mask, idx3, input.data(), 4);

			//compute index to load f11
			__mmask16 maskdxdy = maskdx & maskdy;
			__m512i idx4 = _mm512_mask_add_epi32(idx3, maskdxdy, idx3, epi_one);
			__m512 f11 = _mm512_mask_i32gather_ps(ps_zero, mask, idx4, input.data(), 4);

			Vecf result = interpolate(f00, f10, f01, f11, dx, dy, dx1, dy1);
			result.storeu(dest.addr(r, c), mask);
		}
	}
}

void AvxFrame::unsharpAndWrite(const AvxMatFloat& warped, AvxMatFloat& gauss, float unsharp, ImageYuv* dest, size_t z) {
	for (int r = 0; r < mData.h; r++) {
		for (int c = 0; c < mData.w; c += 16) {
			Vecf ps_warped = warped.addr(r, c);
			Vecf ps_gauss = gauss.addr(r, c);
			Vecf ps_unsharped = Vecf::clamp(ps_warped + (ps_warped - ps_gauss) * unsharp, 0.0f, 1.0f) * 255.0f;
			__m512i chars32 = _mm512_cvt_roundps_epi32(ps_unsharped.a, _MM_FROUND_TO_NEAREST_INT);
			__m128i chars8 = _mm512_cvtepi32_epi8(chars32);
			_mm_storeu_epi8(dest->addr(z, r, c), chars8);
			//if (r==755 && c==464) std::printf("avx %.14f %d\n", result.m512_f32[14], chars32.m512i_i32[14]);
		}
	}
}

void AvxFrame::yuvToRgb(const unsigned char* y, const unsigned char* u, const unsigned char* v, int h, int w, int stride, ImagePPM& dest) {
	uint64_t mask = (1LL << 48) - 1;
	for (int r = 0; r < h; r++) {
		int c = 0;
		for (; c < w / 16 * 16; c += 16) {
			Vecf yy = y + r * stride + c;
			Vecf uu = u + r * stride + c;
			Vecf vv = v + r * stride + c;
			__m512i rgb = yuvToRgbPacked(yy, uu, vv);
			_mm512_mask_storeu_epi8(dest.addr(0, r, c), mask, rgb);
		}
		//TODO rest
	}
}

void AvxFrame::yuvToRgb(const float* y, const float* u, const float* v, int h, int w, int stride, ImagePPM& dest) {
	uint64_t mask = (1LL << 48) - 1;
	for (int r = 0; r < h; r++) {
		int c = 0;
		for (; c < w / 16 * 16; c += 16) {
			Vecf yy = y + r * stride + c;
			Vecf uu = u + r * stride + c;
			Vecf vv = v + r * stride + c;
			__m512i rgb = yuvToRgbPacked(yy * 255.0f, uu * 255.0f, vv * 255.0f);
			_mm512_mask_storeu_epi8(dest.addr(0, r, c), mask, rgb);
		}
		//TODO rest
	}
}

__m512i AvxFrame::yuvToRgbPacked(Vecf y, Vecf u, Vecf v) {
	Vecf r = Vecf::clamp(y + ((v - 128.0f) * 1.370705f), 0.0f, 255.0f);
	Vecf g = Vecf::clamp(y - ((u - 128.0f) * 0.337633f) - ((v - 128.0f) * 0.698001f), 0.0f, 255.0f);
	Vecf b = Vecf::clamp(y + ((u - 128.0f) * 1.732446f), 0.0f, 255.0f);

	//convert floats to uint8 stored in 512 bits
	__m512i ir = _mm512_zextsi128_si512(_mm512_cvtepi32_epi8(_mm512_cvtps_epi32(r.a)));
	__m512i ig = _mm512_zextsi128_si512(_mm512_cvtepi32_epi8(_mm512_cvtps_epi32(g.a)));
	__m512i ib = _mm512_zextsi128_si512(_mm512_cvtepi32_epi8(_mm512_cvtps_epi32(b.a)));
	
	//pack into the lower 3/4 of one 512 vector
	__m512i selectorRG = _mm512_setr_epi8(
		 0, 64,  0,  1, 65,  0,  2, 66,  0,  3, 67,  0,  4, 68,  0,  5,
		69,  0,  6, 70,  0,  7, 71,  0,  8, 72,  0,  9, 73,  0, 10, 74,
		 0, 11, 75,  0, 12, 76,  0, 13, 77,  0, 14, 78,  0, 15, 79,  0,
		 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
	);
	__m512i selectorB = _mm512_setr_epi8(
		64,  65,   0,  67,  68,   1,  70,  71,   2,  73,  74,   3,  76,  77,   4,  79,
		80,   5,  82,  83,   6,  85,  86,   7,  88,  89,   8,  91,  92,   9,  94,  95,
		10,  97,  98,  11, 100, 101,  12, 103, 104,  13, 106, 107,  14, 108, 109,  15,
		 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0
	);

	__m512i result;
	result = _mm512_permutex2var_epi8(ir, selectorRG, ig);
	result = _mm512_permutex2var_epi8(ib, selectorB, result);
	return result;
}