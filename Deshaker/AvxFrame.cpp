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
	pitch { align(data.w, walign) } 
{
	for (int i = 0; i < data.bufferCount; i++) mYUV.emplace_back(data.h, data.w, data.cpupitch);
	mPyr.assign(data.pyramidCount, AvxMatf(data.pyramidRowCount, data.w, 0.0f));
	mYuvPlane = AvxMatf(data.h, pitch, 0.0f);

	mWarped.push_back(AvxMatf(data.h, pitch, data.bgcol_yuv.colors[0]));
	mWarped.push_back(AvxMatf(data.h, pitch, data.bgcol_yuv.colors[1]));
	mWarped.push_back(AvxMatf(data.h, pitch, data.bgcol_yuv.colors[2]));

	mFilterBuffer = AvxMatf(data.w, align(data.h, walign)); //transposed
	mFilterResult = AvxMatf(data.h, pitch);

	mOutput.assign(3, AvxMatf(data.h, pitch));
}

int AvxFrame::align(int base, int alignment) {
	return (base + alignment - 1) / alignment * alignment;
}

//---------------------------------
// ---- main functions ------------
//---------------------------------

MovieFrameId AvxFrame::getId() const {
	return { "AVX 512", "AVX 512: " + mData.getCpuName() };
}

void AvxFrame::inputData() {
	size_t idx = mBufferFrame.index % mYUV.size();
	mBufferFrame.copyTo(mYUV[idx], mPool);
}

void AvxFrame::createPyramid(int64_t frameIndex) {
	//util::ConsoleTimer ic("avx pyramid");
	size_t pyrIdx = frameIndex % mPyr.size();
	AvxMatf& Y = mPyr[pyrIdx];
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

void AvxFrame::getOutput(int64_t frameIndex, ImageRGBA& image) {
	yuvToRgba(mOutput[0].data(), mOutput[1].data(), mOutput[2].data(), mData.h, mData.w, pitch, image);
	image.index = frameIndex;
}

void AvxFrame::getOutput(int64_t frameIndex, unsigned char* cudaNv12ptr, int cudaPitch) {
	static std::vector<unsigned char> nv12(cudaPitch * mData.h * 3 / 2);
	write(nv12, cudaPitch);
	encodeNvData(nv12, cudaNv12ptr);
}

Matf AvxFrame::getTransformedOutput() const {
	return Matf::concatVert(mWarped[0].toMat(), mWarped[1].toMat(), mWarped[2].toMat());
}

void AvxFrame::getWarped(int64_t frameIndex, ImageRGBA& image) {
	yuvToRgba(mWarped[0].data(), mWarped[1].data(), mWarped[2].data(), mData.h, mData.w, pitch, image);
}

Matf AvxFrame::getPyramid(int64_t index) const {
	return mPyr[index].copyToMat();
}

void AvxFrame::getInput(int64_t frameIndex, ImageRGBA& image) {
	size_t idx = frameIndex % mYUV.size();
	ImageYuv& yuv = mYUV[idx];
	yuvToRgba(yuv.plane(0), yuv.plane(1), yuv.plane(2), mData.h, mData.w, yuv.stride, image);
}

void AvxFrame::getInput(int64_t index, ImageYuv& image) const {
	size_t idx = index % mYUV.size();
	mYUV[idx].copyTo(image);
}


//----------------------------------------------------
// ----- IMPLEMENTATION DETAILS ----------------------
//----------------------------------------------------


void AvxFrame::filter(const AvxMatf& src, int r0, int h, int w, AvxMatf& dest, std::span<float> kernel) {
	//util::ConsoleTimer ic("avx filter " + std::to_string(w) + "x" + std::to_string(h));
	VF8 k(kernel.data(), 0b0001'1111);
	for (int r = 0; r < h; r++) mPool.add([&, r] {
		const float* row = src.addr(r0 + r, 0);

		//write points 0 and 1 with clamped input
		dest.at(0, r) = VF8(row[0], row[0], row[0], row[1], row[2], 0, 0, 0).mul(k).sum(0, 5);
		dest.at(1, r) = VF8(row[0], row[0], row[1], row[2], row[3], 0, 0, 0).mul(k).sum(0, 5);

		//loop from points 2 up to w-2
		for (int c = 0; c < w - 4; c++) {
			dest.at(c + 2, r) = VF8(row + c).mul(k).sum(0, 5);
		}

		//write points w-2 and w-1 with clamped input
		dest.at(w - 2, r) = VF8(row[w - 4], row[w - 3], row[w - 2], row[w - 1], row[w - 1], 0, 0, 0).mul(k).sum(0, 5);
		dest.at(w - 1, r) = VF8(row[w - 3], row[w - 2], row[w - 1], row[w - 1], row[w - 1], 0, 0, 0).mul(k).sum(0, 5);
	});
	mPool.wait();
}

//void AvxFrame::filter(std::span<VF16> v, std::span<float> k, AvxMatf& dest, int r0, int c0) {
//	v[0] = v[0] * k[0] + v[0].rot<1>() * k[1] + v[0].rot<2>() * k[2] + v[0].rot<3>() * k[3] + v[0].rot<4>() * k[4];
//	v[1] = v[1] * k[0] + v[1].rot<1>() * k[1] + v[1].rot<2>() * k[2] + v[1].rot<3>() * k[3] + v[1].rot<4>() * k[4];
//	v[2] = v[2] * k[0] + v[2].rot<1>() * k[1] + v[2].rot<2>() * k[2] + v[2].rot<3>() * k[3] + v[2].rot<4>() * k[4];
//	v[3] = v[3] * k[0] + v[3].rot<1>() * k[1] + v[3].rot<2>() * k[2] + v[3].rot<3>() * k[3] + v[3].rot<4>() * k[4];
//	
//	Avx::transpose16x4(v);
//
//	_mm_storeu_ps(dest.addr(r0 + 0, c0), _mm512_extractf32x4_ps(v[0], 0));
//	_mm_storeu_ps(dest.addr(r0 + 1, c0), _mm512_extractf32x4_ps(v[1], 0));
//	_mm_storeu_ps(dest.addr(r0 + 2, c0), _mm512_extractf32x4_ps(v[2], 0));
//	_mm_storeu_ps(dest.addr(r0 + 3, c0), _mm512_extractf32x4_ps(v[3], 0));
//	_mm_storeu_ps(dest.addr(r0 + 4, c0), _mm512_extractf32x4_ps(v[0], 1));
//	_mm_storeu_ps(dest.addr(r0 + 5, c0), _mm512_extractf32x4_ps(v[1], 1));
//	_mm_storeu_ps(dest.addr(r0 + 6, c0), _mm512_extractf32x4_ps(v[2], 1));
//	_mm_storeu_ps(dest.addr(r0 + 7, c0), _mm512_extractf32x4_ps(v[3], 1));
//	_mm_storeu_ps(dest.addr(r0 + 8, c0), _mm512_extractf32x4_ps(v[0], 2));
//	_mm_storeu_ps(dest.addr(r0 + 9, c0), _mm512_extractf32x4_ps(v[1], 2));
//	_mm_storeu_ps(dest.addr(r0 + 10, c0), _mm512_extractf32x4_ps(v[2], 2));
//	_mm_storeu_ps(dest.addr(r0 + 11, c0), _mm512_extractf32x4_ps(v[3], 2));
//}
//
//void AvxFrame::filter(const AvxMatf& src, int r0, int h, int w, AvxMatf& dest, std::span<float> k) {
//	//util::ConsoleTimer ic("avx filter " + std::to_string(w) + "x" + std::to_string(h));
//	assert(h >= 16 && w >= 16 && "invalid dimensions");
//
//	//always handle block of data of 4 rows of 16 values
//	std::vector<VF16> v(4);
//
//	for (int r = 0; ; ) {
//		//left edge
//		for (int i = 0; i < 4; i++) {
//			const float* in = src.addr(r0 + r + i, 0);
//			v[i] = VF16(in[0], in[0], in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], in[8], in[9], in[10], in[11], in[12], in[13]);
//		}
//		filter(v, k, dest, 0, r);
//
//		//main loop
//		for (int c = 10; c < w - 15; c += 12) {
//			for (int i = 0; i < 4; i++) v[i] = src.addr(r0 + r + i, c);
//			filter(v, k, dest, c + 2, r);
//		}
//
//		//right edge
//		for (int i = 0; i < 4; i++) {
//			const float* in = src.addr(r0 + r + i, w - 14);
//			v[i] = VF16(in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], in[8], in[9], in[10], in[11], in[12], in[13], in[13], in[13]);
//		}
//		filter(v, k, dest, w - 12, r);
//
//		//next rows
//		if (r == h - 4) break;
//		else r = std::min(r + 4, h - 4);
//	}
//}

void AvxFrame::downsample(const float* srcptr, int h, int w, int stride, float* destptr, int destStride) {
	__m512i idx1 = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
	__m512i idx2 = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
	VF16 f = 0.5f;

	for (int r = 0; r < h - 1; r += 2) {
		for (int c = 0; c < w; c += 32) {
			const float* src = srcptr + r * stride + c;
			VF16 x1 = _mm512_loadu_ps(src);
			VF16 x2 = _mm512_loadu_ps(src + 16);
			VF16 y1 = _mm512_loadu_ps(src + stride);
			VF16 y2 = _mm512_loadu_ps(src + stride + 16);

			VF16 f00 = _mm512_permutex2var_ps(x1, idx1, x2);
			VF16 f01 = _mm512_permutex2var_ps(x1, idx2, x2);
			VF16 f10 = _mm512_permutex2var_ps(y1, idx1, y2);
			VF16 f11 = _mm512_permutex2var_ps(y1, idx2, y2);
			VF16 result = interpolate(f00, f10, f01, f11, f, f, f, f);

			float* dest = destptr + r / 2 * destStride + c / 2;
			result.storeu(dest);
		}
	}
}

void AvxFrame::yuvToFloat(const ImageYuv& yuv, size_t plane, AvxMatf& dest) {
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
	size_t pyrIdx = frameIndex % mPyr.size();
	size_t pyrIdxPrev = (frameIndex - 1) % mPyr.size();
	AvxMatf& Ycur = mPyr[pyrIdx];
	AvxMatf& Yprev = mPyr[pyrIdxPrev];
	assert(Ycur.frameIndex > 0 && Ycur.frameIndex == Yprev.frameIndex + 1 && "wrong frames to compute");

	for (int threadIdx = 0; threadIdx < mData.cpuThreads; threadIdx++) mPool.add([&, threadIdx] {
		int ir = mData.ir;
		int iw = mData.iw;
		int iww = iw * iw;

		AvxMatd sd(6, iw * iw);
		std::vector<double> eta(6);
		std::vector<double> wp(6);
		std::vector<double> dwp(6);
		__mmask8 maskIW = (1 << iw) - 1;
		__m512i vindexScatter = _mm512_setr_epi64(0, iw, 2ll * iw, 3ll * iw, 4ll * iw, 5ll * iw, 6ll * iw, 7ll * iw);
		__m512i vidxGather = _mm512_setr_epi64(0, iww, 2ll * iww, 3ll * iww, 4ll * iww, 5ll * iww, 0, 0);

		for (int iy0 = threadIdx; iy0 < mData.iyCount; iy0 += mData.cpuThreads) {
			for (int ix0 = 0; ix0 < mData.ixCount; ix0++) {
				wp = { 1, 0, 0, 0, 1, 0 };

				// center of previous integration window
				// one pixel padding around outside for delta
				// changes per z level
				int ym = iy0 + ir + 1;
				int xm = ix0 + ir + 1;
				PointResultType result = PointResultType::RUNNING;
				int z = mData.zMax;
				double err = 0.0;
				int rowOffset = mData.pyramidRowCount;

				for (; z >= mData.zMin && result >= PointResultType::RUNNING; z--) {
					rowOffset -= (mData.h >> z);

					for (int c = 0; c < iw; c++) {
						int iy = ym - ir + c + rowOffset;
						int ix = xm - ir;
						VD8 dx = VF8(Yprev.addr(iy, ix + 1)) / 2 - VF8(Yprev.addr(iy, ix - 1)) / 2;
						VD8 dy = VF8(Yprev.addr(iy + 1, ix)) / 2 - VF8(Yprev.addr(iy - 1, ix)) / 2;
						_mm512_mask_i64scatter_pd(sd.addr(0, c), maskIW, vindexScatter, dx, 8);
						_mm512_mask_i64scatter_pd(sd.addr(1, c), maskIW, vindexScatter, dy, 8);

						const VD8 f23 = VD8(0 - ir, 1 - ir, 2 - ir, 3 - ir, 4 - ir, 5 - ir, 6 - ir, 7 - ir);
						_mm512_mask_i64scatter_pd(sd.addr(2, c), maskIW, vindexScatter, dx * f23, 8);
						_mm512_mask_i64scatter_pd(sd.addr(3, c), maskIW, vindexScatter, dy * f23, 8);

						const VD8 f45 = VD8(c - ir);
						_mm512_mask_i64scatter_pd(sd.addr(4, c), maskIW, vindexScatter, dx * f45, 8);
						_mm512_mask_i64scatter_pd(sd.addr(5, c), maskIW, vindexScatter, dy * f45, 8);
					}
					//if (frameIndex == 1 && ix0 == 63 && iy0 == 1 && z == mData.zMax) sd.toConsole(); //----------------

					//s = sd * sd'
					std::vector<VD8> s(6);
					for (int i = 0; i < iww; i++) {
						VD8 a = _mm512_i64gather_pd(vidxGather, sd.addr(0, i), 8);
						for (int k = 0; k < 6; k++) {
							s[k] += a * sd.at(k, i);
						}
					}
					//if (frameIndex == 1 && ix0 == 63 && iy0 == 1) Avx::toConsole(s); //----------------

					double ns = Avx::norm1(s);
					std::span<VD8> g = s;
					Avx::inv(g);
					double gs = Avx::norm1(g);
					double rcond = 1 / (ns * gs);
					result = (std::isnan(rcond) || rcond < mData.deps) ? PointResultType::FAIL_SINGULAR : PointResultType::RUNNING;
					//if (frameIndex == 1 && ix0 == 75 && iy0 == 10) Avx::toConsole(g); //----------------

					int iter = 0;
					double bestErr = std::numeric_limits<double>::max();
					std::vector<VD8> delta(iw);

					while (result == PointResultType::RUNNING) {
						for (int r = 0; r < iw; r++) {
							//compute coordinate point for interpolation
							VD8 x0 = VD8(0 - ir, 1 - ir, 2 - ir, 3 - ir, 4 - ir, 5 - ir, 6 - ir, 7 - ir);
							VD8 y0 = r - ir;
							VD8 x = VD8(xm) + x0 * wp[0] + y0 * wp[3] + wp[2];
							VD8 y = VD8(ym) + x0 * wp[1] + y0 * wp[4] + wp[5];

							//check image bounds
							__mmask8 mask = 0xFF;
							VD8 checkValue = 0.0;
							mask &= _mm512_cmp_pd_mask(x, checkValue, _CMP_GE_OS); //greater equal
							mask &= _mm512_cmp_pd_mask(y, checkValue, _CMP_GE_OS); //greater equal
							checkValue = (mData.w >> z) - 1.0;
							mask &= _mm512_cmp_pd_mask(x, checkValue, _CMP_LE_OS); //less equal
							checkValue = (mData.h >> z) - 1.0;
							mask &= _mm512_cmp_pd_mask(y, checkValue, _CMP_LE_OS); //less equal

							//compute fractions
							VD8 flx = _mm512_floor_pd(x);
							VD8 fly = _mm512_floor_pd(y);
							VD8 dx = _mm512_sub_pd(x, flx);
							VD8 dy = _mm512_sub_pd(y, fly);

							//prepare values
							VD8 pd_zero = 0.0;
							VF8 ps_zero = 0.0f;
							__m256i epi_one = _mm256_set1_epi32(1);
							__m256i epi_stride = _mm256_set1_epi32(Ycur.w());
							__m256i idx, idx2;

							//index to load f00
							__m256i ix = _mm512_cvtpd_epi32(flx);
							__m256i iy = _mm512_cvtpd_epi32(fly);
							idx = _mm256_mullo_epi32(epi_stride, iy);    //idx = stride * row
							idx = _mm256_add_epi32(idx, ix);             //idx += col
							VD8 f00 = _mm256_mmask_i32gather_ps(ps_zero, mask, idx, Ycur.row(rowOffset), 4);

							//index to load f01
							__mmask8 maskdx = _mm512_cmp_pd_mask(dx, pd_zero, _CMP_NEQ_OS); //not equal
							idx2 = _mm256_mask_add_epi32(idx, maskdx, idx, epi_one);
							VD8 f01 = _mm256_mmask_i32gather_ps(ps_zero, mask, idx2, Ycur.row(rowOffset), 4);

							//index to load f10
							__mmask8 maskdy = _mm512_cmp_pd_mask(dy, pd_zero, _CMP_NEQ_OS); //not equal
							idx2 = _mm256_mask_add_epi32(idx, maskdy, idx, epi_stride);
							VD8 f10 = _mm256_mmask_i32gather_ps(ps_zero, mask, idx2, Ycur.row(rowOffset), 4);

							//index to load f11
							idx2 = _mm256_mask_add_epi32(idx2, maskdx, idx2, epi_one);
							VD8 f11 = _mm256_mmask_i32gather_ps(ps_zero, mask, idx2, Ycur.row(rowOffset), 4);

							//interpolate
							VD8 dx1 = VD8(1.0) - dx;
							VD8 dy1 = VD8(1.0) - dy;
							VD8 jm = dx1 * dy1 * f00 + dx1 * dy * f10 + dx * dy1 * f01 + dx * dy * f11;
							
							//delta
							VF8 im = VF8(Yprev.addr(rowOffset + ym + r - ir, xm - ir), maskIW);
							VD8 nan = mData.dnan;
							delta[r] = _mm512_mask_sub_pd(nan, mask, _mm512_cvtps_pd(im), jm);
						}
						//if (frameIndex == 1 && ix0 == 48 && iy0 == 0 && iter == 1) Avx::toConsole(delta, 18); //----------------

						//eta = g.times(sd.times(delta.flatToCol())) //[6 x 1]
						VD8 b = 0.0;
						for (int c = 0; c < iw; c++) {
							for (int r = 0; r < iw; r++) {
								VD8 val_sd = _mm512_i64gather_pd(vidxGather, sd.addr(0, c * iw + r), 8);
								__m512i vindex = _mm512_set1_epi64(c);
								VD8 val_delta = _mm512_permutexvar_pd(vindex, delta[r]);
								b += val_sd * val_delta;
							}
						}
						eta = { 0, 0, 1, 0, 0, 1 };
						for (int c = 0; c < 6; c++) {
							for (int r = 0; r < 6; r++) {
								eta[r] += g[r][c] * b[c];
							}
						}
						//if (frameIndex == 1 && ix0 == 48 && iy0 == 0 && iter == 1) Avx::toConsole(b, 18);
						//if (frameIndex == 1 && ix0 == 48 && iy0 == 0 && iter == 1) Matd::fromArray(6, 1, eta.data(), false).toConsole("avx", 18);

						dwp[0] = wp[0] * eta[2] + wp[1] * eta[4];
						dwp[1] = wp[0] * eta[3] + wp[1] * eta[5];
						dwp[2] = wp[0] * eta[0] + wp[1] * eta[1] + wp[2];
						dwp[3] = wp[3] * eta[2] + wp[4] * eta[4];
						dwp[4] = wp[3] * eta[3] + wp[4] * eta[5];
						dwp[5] = wp[3] * eta[0] + wp[4] * eta[1] + wp[5];

						wp = dwp;
						//if (frameIndex == 1 && ix0 == 48 && iy0 == 0) Matd::fromArray(2, 3, wp.data(), false).toConsole("avx", 20);

						err = eta[0] * eta[0] + eta[1] * eta[1];
						if (std::isnan(err)) result = PointResultType::FAIL_ETA_NAN;
						if (err < mData.COMP_MAX_TOL) result = PointResultType::SUCCESS_ABSOLUTE_ERR;
						if (std::abs(err - bestErr) / bestErr < mData.COMP_MAX_TOL * mData.COMP_MAX_TOL) result = PointResultType::SUCCESS_STABLE_ITER;
						if (err < bestErr) bestErr = err;
						iter++;
						if (iter == mData.COMP_MAX_ITER && result == PointResultType::RUNNING) result = PointResultType::FAIL_ITERATIONS;
					}

					//center of integration window on next level
					ym *= 2;
					xm *= 2;

					//transformation x 2 for next higher z level
					wp[2] *= 2.0;
					wp[5] *= 2.0;
				}

				//bring values to level 0
				double u = wp[2];
				double v = wp[5];
				int zp = z;

				while (z < 0) {
					xm /= 2; ym /= 2; u /= 2.0; v /= 2.0; z++;
				}
				while (z > 0) {
					xm *= 2; ym *= 2; u *= 2.0; v *= 2.0; z--;
				}

				//transformation for points with respect to center of image and level 0 of pyramid
				int idx = iy0 * mData.ixCount + ix0;
				mResultPoints[idx] = { idx, ix0, iy0, xm, ym, xm - mData.w / 2, ym - mData.h / 2, u, v, result, zp };
			}
		}
	});
	mPool.wait();
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

void AvxFrame::warpBack(const AffineTransform& trf, const AvxMatf& input, AvxMatf& dest) {
	//util::ConsoleTimer ic("avx warp");
	//transform parameters
	VD8 m00 = trf.arrayValue(0);
	VD8 m01 = trf.arrayValue(1);
	VD8 m02 = trf.arrayValue(2);
	VD8 m10 = trf.arrayValue(3);
	VD8 m11 = trf.arrayValue(4);
	VD8 m12 = trf.arrayValue(5);
	VD8 offset = 8;

	for (int r = 0; r < mData.h; r++) mPool.add([&, r] {
		__m512d iy = _mm512_set1_pd(r);
		__m512d ix = _mm512_setr_pd(0, 1, 2, 3, 4, 5, 6, 7);
		for (int c = 0; c < pitch; c += 16) {

			auto t1 = transform(ix, iy, m00, m01, m02, m10, m11, m12);
			ix = _mm512_add_pd(ix, offset);
			auto t2 = transform(ix, iy, m00, m01, m02, m10, m11, m12);
			ix = _mm512_add_pd(ix, offset);
			VF16 xx1 = _mm512_castps256_ps512(_mm512_cvtpd_ps(t1.first));
			VF16 yy1 = _mm512_castps256_ps512(_mm512_cvtpd_ps(t1.second));
			VF16 xx2 = _mm512_castps256_ps512(_mm512_cvtpd_ps(t2.first));
			VF16 yy2 = _mm512_castps256_ps512(_mm512_cvtpd_ps(t2.second));

			VF16 x = _mm512_mask_expand_ps(xx1, 0xFF00, xx2);
			VF16 y = _mm512_mask_expand_ps(yy1, 0xFF00, yy2);

			VF16 ps_zero = _mm512_set1_ps(0.0f);
			__m512i epi_one = _mm512_set1_epi32(1);
			__m512i epi_stride = _mm512_set1_epi32(input.w());
			__m512i idx;

			//check within image bounds
			__mmask16 mask = 0xFFFF;
			VF16 check;
			check = _mm512_set1_ps(0.0f);
			mask &= _mm512_cmp_ps_mask(x, check, _CMP_GE_OS); //greater equal
			mask &= _mm512_cmp_ps_mask(y, check, _CMP_GE_OS); //greater equal
			check = _mm512_set1_ps(float(input.w() - 1));
			mask &= _mm512_cmp_ps_mask(x, check, _CMP_LE_OS); //less equal
			check = _mm512_set1_ps(float(input.h() - 1));
			mask &= _mm512_cmp_ps_mask(y, check, _CMP_LE_OS); //less equal

			//compute fractions
			VF16 flx = _mm512_floor_ps(x);
			VF16 fly = _mm512_floor_ps(y);
			VF16 dx = _mm512_sub_ps(x, flx);
			VF16 dy = _mm512_sub_ps(y, fly);

			//index to load f00
			__m512i ix = _mm512_cvtps_epi32(flx);
			__m512i iy = _mm512_cvtps_epi32(fly);
			idx = _mm512_mullo_epi32(epi_stride, iy);    //idx = stride * row
			idx = _mm512_add_epi32(idx, ix);             //idx += col
			VF16 f00 = _mm512_mask_i32gather_ps(ps_zero, mask, idx, input.data(), 4);

			//index to load f01
			__mmask16 maskdx = _mm512_cmp_ps_mask(dx, ps_zero, _CMP_NEQ_OS); //not equal
			__m512i idx2 = _mm512_mask_add_epi32(idx, maskdx, idx, epi_one);
			VF16 f01 = _mm512_mask_i32gather_ps(ps_zero, mask, idx2, input.data(), 4);

			//index to load f10
			__mmask16 maskdy = _mm512_cmp_ps_mask(dy, ps_zero, _CMP_NEQ_OS); //not equal
			__m512i idx3 = _mm512_mask_add_epi32(idx, maskdy, idx, epi_stride);
			VF16 f10 = _mm512_mask_i32gather_ps(ps_zero, mask, idx3, input.data(), 4);

			//index to load f11
			__m512i idx4 = _mm512_mask_add_epi32(idx3, maskdx, idx3, epi_one);
			VF16 f11 = _mm512_mask_i32gather_ps(ps_zero, mask, idx4, input.data(), 4);

			VF16 result = interpolate(f00, f10, f01, f11, dx, dy);
			result.storeu(dest.addr(r, c), mask);
		}
	});
	mPool.wait();
}

void AvxFrame::unsharp(const AvxMatf& warped, AvxMatf& gauss, float unsharp, AvxMatf& out) {
	//util::ConsoleTimer ic("avx unsharp");
	for (int r = 0; r < mData.h; r++) {
		for (int c = 0; c < mData.w; c += 16) {
			VF16 ps_warped = warped.addr(r, c);
			VF16 ps_gauss = gauss.addr(r, c);
			VF16 ps_unsharped = (ps_warped + (ps_warped - ps_gauss) * unsharp).clamp(0.0f, 1.0f);
			ps_unsharped.storeu(out.addr(r, c));
		}
	}
}

void AvxFrame::write(ImageYuv& dest) {
	for (int z = 0; z < 3; z++) {
		for (int r = 0; r < mData.h; r++) {
			for (int c = 0; c < mData.w; c += 16) {
				VF16 out = mOutput[z].addr(r, c);
				__m512i chars32 = _mm512_cvt_roundps_epi32(out * 255.0f, _MM_FROUND_TO_NEAREST_INT);
				__m128i chars8 = _mm512_cvtepi32_epi8(chars32);
				_mm_storeu_epi8(dest.addr(z, r, c), chars8);
			}
		}
	}
}

void AvxFrame::write(std::span<unsigned char> nv12, int cudaPitch) {
	//util::ConsoleTimer ic("avx write nv12");
	//Y-Plane
	for (int r = 0; r < mData.h; r++) {
		unsigned char* dest = nv12.data() + r * cudaPitch;
		for (int c = 0; c < mData.w; c += 16) {
			VF16 out = mOutput[0].addr(r, c);
			__m512i chars32 = _mm512_cvt_roundps_epi32(out * 255.0f, _MM_FROUND_TO_NEAREST_INT);
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
void AvxFrame::yuvToRgba(const unsigned char* y, const unsigned char* u, const unsigned char* v, int h, int w, int stride, ImageRGBA& dest) {
	for (int r = 0; r < h; r++) {
		for (int c = 0; c < w; c += 4) {
			VF4 vecy = y + r * stride + c;
			VF4 vecu = u + r * stride + c;
			VF4 vecv = v + r * stride + c;
			__m128i rgba = Avx::yuvToRgbaPacked(vecy, vecu, vecv);
			_mm_storeu_epi8(dest.addr(0, r, c), rgba);
		}
	}
}

//from float yuv to uchar rgb
void AvxFrame::yuvToRgba(const float* y, const float* u, const float* v, int h, int w, int stride, ImageRGBA& dest) {
	VF4 f = 255.0f;
	for (int r = 0; r < h; r++) {
		for (int c = 0; c < w; c += 4) {
			VF4 vecy = y + r * stride + c;
			VF4 vecu = u + r * stride + c;
			VF4 vecv = v + r * stride + c;
			__m128i rgba = Avx::yuvToRgbaPacked(vecy * f, vecu * f, vecv * f);
			_mm_storeu_epi8(dest.addr(0, r, c), rgba);
		}
	}
}
