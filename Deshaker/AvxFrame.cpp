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

#include "AvxUtil.hpp"
#include "AvxFrame.hpp"
#include "Util.hpp"

 //---------------------------------
 // ---- constructor ---------------
 //---------------------------------

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

	mFilterBuffer = AvxMatFloat(data.w, alignValue(data.h, walign)); //transposed
	mFilterResult = AvxMatFloat(data.h, pitch);

	mOutput.assign(3, AvxMatFloat(data.h, pitch));
}

//---------------------------------
// ---- main functions ------------
//---------------------------------

MovieFrameId AvxFrame::getId() const {
	return { "AVX 512", "AVX 512: " + mData.getCpuName() };
}

void AvxFrame::inputData() {
	size_t idx = mBufferFrame.index % mYUV.size();
	mYUV[idx] = mBufferFrame;
}

void AvxFrame::createPyramid(int64_t frameIndex) {
	//util::ConsoleTimer ic("avx pyramid");
	size_t pyrIdx = frameIndex % mPyr.size();
	AvxMatFloat& Y = mPyr[pyrIdx];
	Y.frameIndex = frameIndex;

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
		//if (z == 1) mFilterBuffer.saveAsBMP("f:/filterAvx.bmp");
	}
}

void AvxFrame::outputData(const AffineTransform& trf) {
	//util::ConsoleTimer ic("avx output");
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
		unsharp(mWarped[z], mFilterResult, mData.unsharp[z], mOutput[z]);
	}
}

void AvxFrame::getOutput(int64_t frameIndex, ImageYuv& image) {
	write(image);
	image.index = frameIndex;
}

void AvxFrame::getOutput(int64_t frameIndex, unsigned char* cudaNv12ptr, int cudaPitch) {
	static std::vector<unsigned char> nv12(cudaPitch * mData.h * 3 / 2);
	write(nv12, cudaPitch);
	encodeNvData(nv12, cudaNv12ptr);
}

Matf AvxFrame::getTransformedOutput() const {
	return Matf::concatVert(mWarped[0].toMatf(), mWarped[1].toMatf(), mWarped[2].toMatf());
}

void AvxFrame::getWarped(int64_t frameIndex, ImagePPM& image) {
	yuvToRgb(mWarped[0].data(), mWarped[1].data(), mWarped[2].data(), mData.h, mData.w, pitch, image);
}

Matf AvxFrame::getPyramid(size_t idx) const {
	return mPyr[idx].toMatfCopy();
}

void AvxFrame::getInput(int64_t frameIndex, ImagePPM& image) {
	size_t idx = frameIndex % mYUV.size();
	ImageYuv& yuv = mYUV[idx];
	yuvToRgb(yuv.plane(0), yuv.plane(1), yuv.plane(2), mData.h, mData.w, yuv.stride, image);
}

void AvxFrame::getInput(int64_t index, ImageYuv& image) const {
	size_t idx = index % mYUV.size();
	mYUV[idx].copyTo(image);
}


//----------------------------------------------------
// ----- IMPLEMENTATION DETAILS ----------------------
//----------------------------------------------------


//void AvxFrame::filter(const AvxMatFloat& src, int r0, int h, int w, AvxMatFloat& dest, std::span<float> kernel) {
//	util::ConsoleTimer ic("avx filter " + std::to_string(w) + "x" + std::to_string(h));
//	VF8 k(kernel.data(), 0b0001'1111);
//	for (int r = 0; r < h; r++) {
//		const float* row = src.addr(r0 + r, 0);
//
//		//write points 0 and 1 with clamped input
//		dest.at(0, r) = VF8(row[0], row[0], row[0], row[1], row[2], 0, 0, 0).mul(k).sum(0, 5);
//		dest.at(1, r) = VF8(row[0], row[0], row[1], row[2], row[3], 0, 0, 0).mul(k).sum(0, 5);
//
//		//loop from points 2 up to w-2
//		for (int c = 0; c < w - 4; c++) {
//			dest.at(c + 2, r) = VF8(row + c).mul(k).sum(0, 5);
//		}
//
//		//write points w-2 and w-1 with clamped input
//		dest.at(w - 2, r) = VF8(row[w - 4], row[w - 3], row[w - 2], row[w - 1], row[w - 1], 0, 0, 0).mul(k).sum(0, 5);
//		dest.at(w - 1, r) = VF8(row[w - 3], row[w - 2], row[w - 1], row[w - 1], row[w - 1], 0, 0, 0).mul(k).sum(0, 5);
//	}
//}

void AvxFrame::filter(std::span<VF16> v, std::span<float> k, AvxMatFloat& dest, int r0, int c0) {
	v[0] = v[0] * k[0] + v[0].rot<1>() * k[1] + v[0].rot<2>() * k[2] + v[0].rot<3>() * k[3] + v[0].rot<4>() * k[4];
	v[1] = v[1] * k[0] + v[1].rot<1>() * k[1] + v[1].rot<2>() * k[2] + v[1].rot<3>() * k[3] + v[1].rot<4>() * k[4];
	v[2] = v[2] * k[0] + v[2].rot<1>() * k[1] + v[2].rot<2>() * k[2] + v[2].rot<3>() * k[3] + v[2].rot<4>() * k[4];
	v[3] = v[3] * k[0] + v[3].rot<1>() * k[1] + v[3].rot<2>() * k[2] + v[3].rot<3>() * k[3] + v[3].rot<4>() * k[4];
	
	Avx::transpose16x4(v);

	_mm_storeu_ps(dest.addr(r0 + 0, c0), _mm512_extractf32x4_ps(v[0], 0));
	_mm_storeu_ps(dest.addr(r0 + 1, c0), _mm512_extractf32x4_ps(v[1], 0));
	_mm_storeu_ps(dest.addr(r0 + 2, c0), _mm512_extractf32x4_ps(v[2], 0));
	_mm_storeu_ps(dest.addr(r0 + 3, c0), _mm512_extractf32x4_ps(v[3], 0));
	_mm_storeu_ps(dest.addr(r0 + 4, c0), _mm512_extractf32x4_ps(v[0], 1));
	_mm_storeu_ps(dest.addr(r0 + 5, c0), _mm512_extractf32x4_ps(v[1], 1));
	_mm_storeu_ps(dest.addr(r0 + 6, c0), _mm512_extractf32x4_ps(v[2], 1));
	_mm_storeu_ps(dest.addr(r0 + 7, c0), _mm512_extractf32x4_ps(v[3], 1));
	_mm_storeu_ps(dest.addr(r0 + 8, c0), _mm512_extractf32x4_ps(v[0], 2));
	_mm_storeu_ps(dest.addr(r0 + 9, c0), _mm512_extractf32x4_ps(v[1], 2));
	_mm_storeu_ps(dest.addr(r0 + 10, c0), _mm512_extractf32x4_ps(v[2], 2));
	_mm_storeu_ps(dest.addr(r0 + 11, c0), _mm512_extractf32x4_ps(v[3], 2));
}

void AvxFrame::filter(const AvxMatFloat& src, int r0, int h, int w, AvxMatFloat& dest, std::span<float> k) {
	//util::ConsoleTimer ic("avx filter " + std::to_string(w) + "x" + std::to_string(h));
	assert(h >= 16 && w >= 16 && "invalid dimensions");

	//always handle block of data of 4 rows of 16 values
	std::vector<VF16> v(4);

	for (int r = 0; ; ) {
		//left edge
		for (int i = 0; i < 4; i++) {
			const float* in = src.addr(r0 + r + i, 0);
			v[i] = VF16(in[0], in[0], in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], in[8], in[9], in[10], in[11], in[12], in[13]);
		}
		filter(v, k, dest, 0, r);

		//main loop
		for (int c = 10; c < w - 15; c += 12) {
			for (int i = 0; i < 4; i++) v[i] = src.addr(r0 + r + i, c);
			filter(v, k, dest, c + 2, r);
		}

		//right edge
		for (int i = 0; i < 4; i++) {
			const float* in = src.addr(r0 + r + i, w - 14);
			v[i] = VF16(in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], in[8], in[9], in[10], in[11], in[12], in[13], in[13], in[13]);
		}
		filter(v, k, dest, w - 12, r);

		//next rows
		if (r == h - 4) break;
		else r = std::min(r + 4, h - 4);
	}
}

void AvxFrame::downsample(const float* srcptr, int h, int w, int stride, float* destptr, int destStride) {
	__m512i idx1 = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
	__m512i idx2 = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
	VF16 f = 0.5f;

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
			VF16 result = interpolate(f00, f10, f01, f11, f, f, f, f);

			float* dest = destptr + r / 2 * destStride + c / 2;
			result.storeu(dest);
		}
	}
}

void AvxFrame::yuvToFloat(const ImageYuv& yuv, size_t plane, AvxMatFloat& dest) {
	//util::ConsoleTimer ic("avx yuv to float");
	constexpr float f = 1.0f / 255.0f;
	for (int r = 0; r < mData.h; r++) {
		int c = 0;
		//handle blocks of 16 pixels
		for (; c < mData.w / 16 * 16; c += 16) {
			VF16 result = yuv.addr(plane, r, c);
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
VF16 AvxFrame::interpolate(VF16 f00, VF16 f10, VF16 f01, VF16 f11, VF16 dx, VF16 dy) {
	VF16 one = 1.0f;
	return interpolate(f00, f10, f01, f11, dx, dy, one - dx, one - dy);
}

//compute bilinear interpolation
VF16 AvxFrame::interpolate(VF16 f00, VF16 f10, VF16 f01, VF16 f11, VF16 dx, VF16 dy, VF16 dx1, VF16 dy1) {
	return dx1 * dy1 * f00 + dx1 * dy * f10 + dx * dy1 * f01 + dx * dy * f11;
}

std::pair<VD8, VD8> AvxFrame::transform(VD8 x, VD8 y, VD8 m00, VD8 m01, VD8 m02, VD8 m10, VD8 m11, VD8 m12) {
	VD8 xx = m02;
	xx = _mm512_fmadd_pd(y, m01, xx);
	xx = _mm512_fmadd_pd(x, m00, xx);
	VD8 yy = m12;
	yy = _mm512_fmadd_pd(y, m11, yy);
	yy = _mm512_fmadd_pd(x, m10, yy);
	return { xx, yy };
}

void AvxFrame::warpBack(const AffineTransform& trf, const AvxMatFloat& input, AvxMatFloat& dest) {
	//util::ConsoleTimer ic("avx warp");
	//transform parameters
	VD8 m00 = trf.arrayValue(0);
	VD8 m01 = trf.arrayValue(1);
	VD8 m02 = trf.arrayValue(2);
	VD8 m10 = trf.arrayValue(3);
	VD8 m11 = trf.arrayValue(4);
	VD8 m12 = trf.arrayValue(5);

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

			//index to load f00
			__m512i ix = _mm512_cvtps_epi32(flx);
			__m512i iy = _mm512_cvtps_epi32(fly);
			idx = _mm512_mullo_epi32(epi_stride, iy);    //idx = stride * row
			idx = _mm512_add_epi32(idx, ix);             //idx += col
			__m512 f00 = _mm512_mask_i32gather_ps(ps_zero, mask, idx, input.data(), 4);

			//index to load f01
			__mmask16 maskdx = _mm512_cmp_ps_mask(dx, ps_zero, _CMP_NEQ_OS); //not equal
			__m512i idx2 = _mm512_mask_add_epi32(idx, maskdx, idx, epi_one);
			__m512 f01 = _mm512_mask_i32gather_ps(ps_zero, mask, idx2, input.data(), 4);

			//index to load f10
			__mmask16 maskdy = _mm512_cmp_ps_mask(dy, ps_zero, _CMP_NEQ_OS); //not equal
			__m512i idx3 = _mm512_mask_add_epi32(idx, maskdy, idx, epi_stride);
			__m512 f10 = _mm512_mask_i32gather_ps(ps_zero, mask, idx3, input.data(), 4);

			//index to load f11
			__m512i idx4 = _mm512_mask_add_epi32(idx3, maskdx, idx3, epi_one);
			__m512 f11 = _mm512_mask_i32gather_ps(ps_zero, mask, idx4, input.data(), 4);

			VF16 result = interpolate(f00, f10, f01, f11, dx, dy);
			result.storeu(dest.addr(r, c), mask);
		}
	}
}

void AvxFrame::unsharp(const AvxMatFloat& warped, AvxMatFloat& gauss, float unsharp, AvxMatFloat& out) {
	//util::ConsoleTimer ic("avx unsharp");
	for (int r = 0; r < mData.h; r++) {
		for (int c = 0; c < mData.w; c += 16) {
			VF16 ps_warped = warped.addr(r, c);
			VF16 ps_gauss = gauss.addr(r, c);
			VF16 ps_unsharped = (ps_warped + (ps_warped - ps_gauss) * unsharp).clamp(0.0f, 1.0f) * 255.0f;
			ps_unsharped.storeu(out.addr(r, c));
		}
	}
}

void AvxFrame::write(ImageYuv& dest) {
	for (int z = 0; z < 3; z++) {
		for (int r = 0; r < mData.h; r++) {
			for (int c = 0; c < mData.w; c += 16) {
				VF16 out = mOutput[z].addr(r, c);
				__m512i chars32 = _mm512_cvt_roundps_epi32(out, _MM_FROUND_TO_NEAREST_INT);
				__m128i chars8 = _mm512_cvtepi32_epi8(chars32);
				_mm_storeu_epi8(dest.addr(z, r, c), chars8);
			}
		}
	}
}

void AvxFrame::write(std::span<unsigned char> nv12, int cudaPitch) {
	//Y-Plane
	for (int r = 0; r < mData.h; r++) {
		unsigned char* dest = nv12.data() + r * cudaPitch;
		for (int c = 0; c < mData.w; c += 16) {
			VF16 out = mOutput[0].addr(r, c);
			__m512i chars32 = _mm512_cvt_roundps_epi32(out, _MM_FROUND_TO_NEAREST_INT);
			__m128i chars8 = _mm512_cvtepi32_epi8(chars32);
			_mm_storeu_epi8(dest + c, chars8);
		}
	}

	//U-V-Planes
	unsigned char* dest = nv12.data() + mData.h * cudaPitch;
	for (int rr = 0; rr < mData.h / 2; rr++) {
		int r = rr * 2;
		__m512i a, b, x, sumU, sumV, sum;

		for (int c = 0; c < mData.w; c += 16) {
			a = _mm512_cvt_roundps_epi32(_mm512_loadu_ps(mOutput[1].addr(r, c)), _MM_FROUND_TO_NEAREST_INT);
			b = _mm512_cvt_roundps_epi32(_mm512_loadu_ps(mOutput[1].addr(r + 1, c)), _MM_FROUND_TO_NEAREST_INT);
			sumU = _mm512_add_epi32(a, b);

			a = _mm512_cvt_roundps_epi32(_mm512_loadu_ps(mOutput[2].addr(r, c)), _MM_FROUND_TO_NEAREST_INT);
			b = _mm512_cvt_roundps_epi32(_mm512_loadu_ps(mOutput[2].addr(r + 1, c)), _MM_FROUND_TO_NEAREST_INT);
			sumV = _mm512_add_epi32(a, b);

			//cross over and combine
			x = _mm512_shuffle_epi32(sumU, 0b11110101);
			x = _mm512_mask_shuffle_epi32(x, 0b10101010'10101010, sumV, 0b10100000);
			//combine without crossing
			sum = _mm512_mask_blend_epi32(0b10101010'10101010, sumU, sumV);
			//add the blocks
			sum = _mm512_add_epi32(sum, x);
			//divide sum by 4
			sum = _mm512_srli_epi32(sum, 2);
			__m128i uv8 = _mm512_cvtepi32_epi8(sum);
			_mm_storeu_epi8(dest + c, uv8);
		}

		dest += cudaPitch;
	}
}

//from uchar yuv to uchar rgb
void AvxFrame::yuvToRgb(const unsigned char* y, const unsigned char* u, const unsigned char* v, int h, int w, int stride, ImagePPM& dest) {
	for (int r = 0; r < h; r++) {
		for (int c = 0; c < w; c += 16) {
			int n = std::min(16, w - c);
			__mmask16 maskLoad = (1LL << n) - 1;
			VF16 yy(y + r * stride + c, maskLoad);
			VF16 uu(u + r * stride + c, maskLoad);
			VF16 vv(v + r * stride + c, maskLoad);
			__m512i rgb = Avx::yuvToRgbPacked(yy, uu, vv);

			__mmask64 mask = (1LL << n * 3) - 1;
			_mm512_mask_storeu_epi8(dest.addr(0, r, c), mask, rgb);
		}
	}
}

//from float yuv to uchar rgb
void AvxFrame::yuvToRgb(const float* y, const float* u, const float* v, int h, int w, int stride, ImagePPM& dest) {
	VF16 ps_255 = 255.0f;
	for (int r = 0; r < h; r++) {
		for (int c = 0; c < w; c += 16) {
			int n = std::min(16, w - c);
			__mmask16 maskLoad = (1LL << n) - 1;
			VF16 ps_y(y + r * stride + c, maskLoad);
			VF16 ps_u(u + r * stride + c, maskLoad);
			VF16 ps_v(v + r * stride + c, maskLoad);
			__m512i rgb = Avx::yuvToRgbPacked(ps_y * ps_255, ps_u * ps_255, ps_v * ps_255);

			__mmask64 mask = (1LL << n * 3) - 1;
			_mm512_mask_storeu_epi8(dest.addr(0, r, c), mask, rgb);
		}
	}
}
