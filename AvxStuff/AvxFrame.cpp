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

#include "AvxFrame.hpp"
#include "AvxUtil.hpp"
#include "cuDeshaker.cuh"

AvxFrame::AvxFrame(CoreData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool) :
	FrameExecutor(data, deviceInfo, frame, pool),
	walign { 32 },
	pitch { util::alignValue(data.w, walign) } 
{
	assert(mDeviceInfo.getType() == DeviceType::AVX && "device type must be AVX here");
	for (int i = 0; i < mData.bufferCount; i++) mYUV.emplace_back(mData.h, mData.w, mData.cpupitch);
	mPyr.assign(mData.pyramidCount, AvxMatf(mData.pyramidRowCount, pitch, 0.0f));
	mYuvPlane = AvxMatf(mData.h, pitch, 0.0f);

	mWarped.push_back(AvxMatf(mData.h, pitch, mData.bgcolorYuv[0]));
	mWarped.push_back(AvxMatf(mData.h, pitch, mData.bgcolorYuv[1]));
	mWarped.push_back(AvxMatf(mData.h, pitch, mData.bgcolorYuv[2]));

	mFilterBuffer = AvxMatf(mData.w, util::alignValue(mData.h, walign)); //transposed
	mFilterResult = AvxMatf(mData.h, pitch);

	mOutput.assign(3, AvxMatf(mData.h, pitch));
}


//---------------------------------
// ---- main functions ------------
//---------------------------------


void AvxFrame::inputData(int64_t frameIdx, const ImageYuv& inputFrame) {
	size_t idx = frameIdx % mYUV.size();
	inputFrame.copyTo(mYUV[idx], mPool);
}

void AvxFrame::createPyramid(int64_t frameIndex, AffineDataFloat trf, bool warp) {
	//util::ConsoleTimer ic("avx pyramid");
	size_t pyrIdx = frameIndex % mPyr.size();
	AvxMatf& Y = mPyr[pyrIdx];
	Y.frameIndex = frameIndex;

	//fill topmost level of pyramid
	size_t yuvIdx = frameIndex % mYUV.size();
	ImageYuv& yuv = mYUV[yuvIdx];
	int h = mData.h;
	int w = mData.w;

	if (warp) {
		//convert to float
		yuvToFloat(yuv, 0, mFilterResult);

		//transform input
		Y.fill(0.0f);
		warpBack(trf, mFilterResult, Y);

	} else {
		//convert to float
		yuvToFloat(yuv, 0, Y);

		//filter first level
		filter(Y, 0, h, w, mFilterBuffer, filterKernels[0]);
		filter(mFilterBuffer, 0, w, h, Y, filterKernels[0]);
	}

	//create pyramid levels below by downsampling level above
	int r = 0;
	for (size_t z = 1; z < mData.pyramidLevels; z++) {
		downsample(Y.row(r), h, w, Y.w(), Y.row(r + h), Y.w());
		r += h;
		h /= 2;
		w /= 2;
		//if (z == 1) std::printf("avx %.14f\n", Y.at(r + 100, 100));
		//if (z == 1) mFilterBuffer.saveAsBMP("f:/filterAvx.bmp");
	}
}

void AvxFrame::outputData(int64_t frameIndex, AffineDataFloat trf) {
	//util::ConsoleTimer ic("avx output");
	size_t yuvidx = frameIndex % mYUV.size();
	const ImageYuv& input = mYUV[yuvidx];

	//for planes Y, U, V
	for (size_t z = 0; z < 3; z++) {
		yuvToFloat(input, z, mYuvPlane);
		if (mData.bgmode == BackgroundMode::COLOR) mWarped[z].fill(mData.bgcolorYuv[z]);
		warpBack(trf, mYuvPlane, mWarped[z]);
		filter(mWarped[z], 0, mData.h, mData.w, mFilterBuffer, filterKernels[z]);
		filter(mFilterBuffer, 0, mData.w, mData.h, mFilterResult, filterKernels[z]);
		unsharp(mWarped[z], mFilterResult, mData.unsharp[z], mOutput[z]);
	}
}

void AvxFrame::getOutputYuv(int64_t frameIndex, ImageYuv& image) const {
	write(image);
	image.setIndex(frameIndex);
}

void AvxFrame::getOutputImage(int64_t frameIndex, ImageBaseRgb& image) const {
	//util::ConsoleTimer ic("avx output rgba");
	yuvToRgb(mOutput[0].data(), mOutput[1].data(), mOutput[2].data(), mData.h, mData.w, pitch, image);
	image.setIndex(frameIndex);
}

void AvxFrame::getOutputNvenc(int64_t frameIndex, ImageNV12& image, unsigned char* cudaNv12ptr) const {
	write(image);
	encodeNvData(image, cudaNv12ptr);
}

Matf AvxFrame::getTransformedOutput() const {
	return Matf::concatVert(mWarped[0].matShare(), mWarped[1].matShare(), mWarped[2].matShare());
}

void AvxFrame::getWarped(int64_t frameIndex, ImageRGBA& image) {
	yuvToRgb(mWarped[0].data(), mWarped[1].data(), mWarped[2].data(), mData.h, mData.w, pitch, image);
}

Matf AvxFrame::getPyramid(int64_t index) const {
	return mPyr[index].matCopy();
}

void AvxFrame::getInput(int64_t frameIndex, ImageRGBA& image) const {
	size_t idx = frameIndex % mYUV.size();
	const ImageYuv& yuv = mYUV[idx];
	yuvToRgb(yuv.plane(0), yuv.plane(1), yuv.plane(2), mData.h, mData.w, yuv.stride, image);
}

void AvxFrame::getInput(int64_t index, ImageYuv& image) const {
	size_t idx = index % mYUV.size();
	mYUV[idx].copyTo(image);
}


//----------------------------------------------------
// ----- IMPLEMENTATION DETAILS ----------------------
//----------------------------------------------------


void AvxFrame::filter(const AvxMatf& src, int r0, int h, int w, AvxMatf& dest, std::span<V16f> ks) {
	//util::ConsoleTimer ic("avx filter " + std::to_string(w) + "x" + std::to_string(h));
	__m512i vw = _mm512_set1_epi32(dest.w());
	__m512i idx = _mm512_loadu_epi32(Iota::i32);
	__m512i vidx = _mm512_mullo_epi32(vw, idx);
	__mmask16 mask = 0x0FFF;

	for (int threadIdx = 0; threadIdx < mData.cpuThreads; threadIdx++) mPool.add([&, threadIdx] {
		V16f x, result;

		for (int r = threadIdx; r < h; r += mData.cpuThreads) {
			const float* row = src.addr(r0 + r, 0);

			//write first points adhering to border
			//first broadcast border point, then overwrite other points
			x = _mm512_mask_loadu_ps(_mm512_set1_ps(row[0]), 0xFFFC, row - 2);
			result = x * ks[0] + x.rot<1>() * ks[1] + x.rot<2>() * ks[2] + x.rot<3>() * ks[3] + x.rot<4>() * ks[4];
			_mm512_mask_i32scatter_ps(dest.addr(0, r), mask, vidx, result, 4);

			//main loop
			for (int c = 12; c < w - 12; c += 12) {
				x = V16f(row + c - 2);
				result = x * ks[0] + x.rot<1>() * ks[1] + x.rot<2>() * ks[2] + x.rot<3>() * ks[3] + x.rot<4>() * ks[4];
				_mm512_mask_i32scatter_ps(dest.addr(c, r), mask, vidx, result, 4);
			}

			//write last points adhering to border
			x = _mm512_mask_loadu_ps(_mm512_set1_ps(row[w - 1]), 0x3FFF, row + w - 14);
			result = x * ks[0] + x.rot<1>() * ks[1] + x.rot<2>() * ks[2] + x.rot<3>() * ks[3] + x.rot<4>() * ks[4];
			_mm512_mask_i32scatter_ps(dest.addr(w - 12, r), mask, vidx, result, 4);
		}
	});
	mPool.wait();
}

void AvxFrame::downsample(const float* srcptr, int h, int w, int stride, float* destptr, int destStride) {
	const __m512i idx1 = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
	const __m512i idx2 = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
	const V16f f = 0.5f;

	for (int r = 0; r < h - 1; r += 2) {
		for (int c = 0; c < w; c += 32) {
			const float* src = srcptr + r * stride + c;
			V16f x1 = src;
			V16f x2 = src + 16;
			V16f y1 = src + stride;
			V16f y2 = src + stride + 16;

			V16f f00 = _mm512_permutex2var_ps(x1, idx1, x2);
			V16f f01 = _mm512_permutex2var_ps(x1, idx2, x2);
			V16f f10 = _mm512_permutex2var_ps(y1, idx1, y2);
			V16f f11 = _mm512_permutex2var_ps(y1, idx2, y2);
			V16f result = interpolate(f00, f10, f01, f11, f, f, f, f);

			float* dest = destptr + r / 2 * destStride + c / 2;
			result.storeu(dest);
		}
	}
}

void AvxFrame::yuvToFloat(const ImageYuv& yuv, size_t plane, AvxMatf& dest) {
	//util::ConsoleTimer ic("avx yuv to float");
	constexpr float f = 1.0f / 255.0f;
	V16f ff = f;
	for (int r = 0; r < mData.h; r++) {
		int c = 0;
		//handle blocks of 16 pixels
		for (; c < mData.w / 16 * 16; c += 16) {
			V16f result = yuv.addr(plane, r, c);
			result = result * ff;
			result.storeu(dest.row(r) + c);
		}
		//image width may not align to 16, handle trailing pixels individually
		for (; c < mData.w; c++) {
			dest.at(r, c) = yuv.at(0, r, c) * f;
		}
	}
}

V8d AvxFrame::sd(int c1, int c2, int y0, int x0, const AvxMatf& Y) {
	__m128i w = _mm_set1_epi32(Y.w());
	__m128i x = _mm_set1_epi32(x0 + c1);
	__m128i y = _mm_set1_epi32(y0 + c2);
	__m128i dx = _mm_setr_epi32(1, -1, 0, 0);
	__m128i dy = _mm_setr_epi32(0, 0, 1, -1);

	__m128i idx = y;
	idx = _mm_add_epi32(idx, dy);
	idx = _mm_mullo_epi32(idx, w);
	idx = _mm_add_epi32(idx, x);
	idx = _mm_add_epi32(idx, dx);
	V4f vf = _mm_i32gather_ps(Y.data(), idx, 4);
	vf /= 2.0f;
	vf = _mm_hsub_ps(vf, vf);
	V8d vd = _mm256_broadcast_f32x2(vf);
	V8d f(1, 1, c1 - mData.ir, c1 - mData.ir, c2 - mData.ir, c2 - mData.ir, 0, 0);
	return vd * f;
}

void AvxFrame::computeStart(int64_t frameIndex, std::vector<PointResult>& results) {}

void AvxFrame::computeTerminate(int64_t frameIndex, std::vector<PointResult>& results) {
	//util::ConsoleTimer ct("avx compute");
	size_t idx0 = frameIndex % mPyr.size();
	size_t idx1 = (frameIndex - 1) % mPyr.size();
	assert(mPyr[idx0].frameIndex > 0 && mPyr[idx0].frameIndex == mPyr[idx1].frameIndex + 1 && "wrong frames to compute");

	for (int threadIdx = 0; threadIdx < mData.cpuThreads; threadIdx++) mPool.add([&, threadIdx] {
		int ir = mData.ir;
		int iw = mData.iw;

		std::vector<double> eta(6);
		std::vector<double> wp(6);
		std::vector<double> dwp(6);
		__mmask8 maskIW = (1 << iw) - 1;
		V8d iota = Iota::d;

		for (int iy0 = threadIdx; iy0 < mData.iyCount; iy0 += mData.cpuThreads) {
			for (int ix0 = 0; ix0 < mData.ixCount; ix0++) {
				wp = { 1, 0, 0, 0, 1, 0 };
				int direction = (ix0 % 2) ^ (iy0 % 2);
				AvxMatf& pyr0 = mPyr[(frameIndex - direction) % mPyr.size()];
				AvxMatf& pyr1 = mPyr[(frameIndex - direction + 1) % mPyr.size()];

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

					//s = sd * sd'
					std::vector<V8d> s(6);
					for (int c1 = 0; c1 < iw; c1++) {
						for (int c2 = 0; c2 < iw; c2++) {
							V8d a = sd(c1, c2, ym - ir + rowOffset, xm - ir, pyr1);
							for (int k = 0; k < 6; k++) {
								s[k] += a * a.broadcast(k);
							}
						}
					}

					//if (frameIndex == 1 && ix0 == 63 && iy0 == 1) avx::toConsole(s); //----------------

					double ns = avx::norm1(s);
					std::span<V8d> g = s;
					std::vector<size_t> piv = { 0, 1, 2, 3, 4, 5 };
					avx::inv(g, piv);
					double gs = avx::norm1(g);
					double rcond = 1 / (ns * gs);
					result = (std::isnan(rcond) || rcond < mData.deps) ? PointResultType::FAIL_SINGULAR : PointResultType::RUNNING;
					//if (frameIndex == 1 && ix0 == 75 && iy0 == 10) avx::toConsole(g); //----------------

					int iter = 0;
					double bestErr = std::numeric_limits<double>::max();
					std::vector<V8d> delta(iw);

					while (result == PointResultType::RUNNING) {
						for (int r = 0; r < iw; r++) {
							//compute coordinate point for interpolation
							V8d x0 = iota - ir;
							V8d y0 = r - ir;
							V8d x = V8d(xm) + x0 * wp[0] + y0 * wp[3] + wp[2];
							V8d y = V8d(ym) + x0 * wp[1] + y0 * wp[4] + wp[5];

							//check image bounds
							__mmask8 mask = 0xFF;
							V8d checkValue = 0.0;
							mask &= _mm512_cmp_pd_mask(x, checkValue, _CMP_GE_OS); //greater equal
							mask &= _mm512_cmp_pd_mask(y, checkValue, _CMP_GE_OS); //greater equal
							checkValue = (mData.w >> z) - 1.0;
							mask &= _mm512_cmp_pd_mask(x, checkValue, _CMP_LE_OS); //less equal
							checkValue = (mData.h >> z) - 1.0;
							mask &= _mm512_cmp_pd_mask(y, checkValue, _CMP_LE_OS); //less equal

							//compute fractions
							V8d flx = _mm512_floor_pd(x);
							V8d fly = _mm512_floor_pd(y);
							V8d dx = x - flx;
							V8d dy = y - fly;

							//prepare values
							V8d pd_zero = 0.0;
							V8f ps_zero = 0.0f;
							__m256i epi_one = _mm256_set1_epi32(1);
							__m256i epi_stride = _mm256_set1_epi32(pyr0.w());
							__m256i idx, idx2;

							//index to load f00
							__m256i ix = _mm512_cvtpd_epi32(flx);
							__m256i iy = _mm512_cvtpd_epi32(fly);
							idx = _mm256_mullo_epi32(epi_stride, iy);    //idx = stride * row
							idx = _mm256_add_epi32(idx, ix);             //idx += col
							V8d f00 = _mm256_mmask_i32gather_ps(ps_zero, mask, idx, pyr0.row(rowOffset), 4);

							//index to load f01
							__mmask8 maskdx = _mm512_cmp_pd_mask(dx, pd_zero, _CMP_NEQ_OS); //not equal
							idx2 = _mm256_mask_add_epi32(idx, maskdx, idx, epi_one);
							V8d f01 = _mm256_mmask_i32gather_ps(ps_zero, mask, idx2, pyr0.row(rowOffset), 4);

							//index to load f10
							__mmask8 maskdy = _mm512_cmp_pd_mask(dy, pd_zero, _CMP_NEQ_OS); //not equal
							idx2 = _mm256_mask_add_epi32(idx, maskdy, idx, epi_stride);
							V8d f10 = _mm256_mmask_i32gather_ps(ps_zero, mask, idx2, pyr0.row(rowOffset), 4);

							//index to load f11
							idx2 = _mm256_mask_add_epi32(idx2, maskdx, idx2, epi_one);
							V8d f11 = _mm256_mmask_i32gather_ps(ps_zero, mask, idx2, pyr0.row(rowOffset), 4);

							//interpolate
							V8d one = 1.0;
							V8d dx1 = one - dx;
							V8d dy1 = one - dy;
							V8d jm = dx1 * dy1 * f00 + dx1 * dy * f10 + dx * dy1 * f01 + dx * dy * f11;

							//delta
							V8f im = V8f(pyr1.addr(rowOffset + ym + r - ir, xm - ir), maskIW);
							V8d nan = mData.dnan;
							delta[r] = _mm512_mask_sub_pd(nan, mask, _mm512_cvtps_pd(im), jm);
						}
						//if (frameIndex == 1 && ix0 == 48 && iy0 == 0 && iter == 1) avx::toConsole(delta, 18); //----------------

						//eta = g.times(sd.times(delta.flatToCol())) //[6 x 1]
						V8d b = 0.0;
						for (int c = 0; c < iw; c++) {
							for (int r = 0; r < iw; r++) {
								V8d vsd = sd(c, r, ym - ir + rowOffset, xm - ir, pyr1);
								V8d vdelta = delta[r].broadcast(c);
								b += vsd * vdelta;
							}
						}
						eta = { 0, 0, 1, 0, 0, 1 };
						for (int c = 0; c < 6; c++) {
							for (int r = 0; r < 6; r++) {
								eta[r] += g[r][c] * b[c];
							}
						}
						//if (frameIndex == 1 && ix0 == 48 && iy0 == 0 && iter == 1) avx::toConsole(b, 18);
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
				double x0 = xm - mData.w / 2.0 + u * direction;
				double y0 = ym - mData.h / 2.0 + v * direction;
				double fdir = 1.0 - 2.0 * direction;
				results[idx] = { idx, ix0, iy0, x0, y0, u * fdir, v * fdir, result, zp, direction };
			}
		}
	});
	mPool.wait();
}

//compute bilinear interpolation
//result = (1 - dx) * (1 - dy) * f00 + (1 - dx) * dy * f10 + dx * (1 - dy) * f01 + dx * dy * f11;
V16f AvxFrame::interpolate(V16f f00, V16f f10, V16f f01, V16f f11, V16f dx, V16f dy) {
	V16f one = 1.0f;
	return interpolate(f00, f10, f01, f11, dx, dy, one - dx, one - dy);
}

//compute bilinear interpolation
V16f AvxFrame::interpolate(V16f f00, V16f f10, V16f f01, V16f f11, V16f dx, V16f dy, V16f dx1, V16f dy1) {
	return dx1 * dy1 * f00 + dx1 * dy * f10 + dx * dy1 * f01 + dx * dy * f11;
}

void AvxFrame::warpBack(const AffineDataFloat& trf, const AvxMatf& input, AvxMatf& dest) {
	//util::ConsoleTimer ic("avx warp");
	//transform parameters
	V16f m00 = trf.m00;
	V16f m01 = trf.m01;
	V16f m02 = trf.m02;
	V16f m10 = trf.m10;
	V16f m11 = trf.m11;
	V16f m12 = trf.m12;
	V16f iota = Iota::f;

	for (int threadIdx = 0; threadIdx < mData.cpuThreads; threadIdx++) mPool.add([&, threadIdx] {
		for (int r = threadIdx; r < mData.h; r += mData.cpuThreads) {
			V16f ix = iota;
			V16f iy = float(r);
			V16f ps_zero = _mm512_set1_ps(0.0f);
			__m512i epi_one = _mm512_set1_epi32(1);
			__m512i epi_stride = _mm512_set1_epi32(input.w());
			__m512i idx;
			
			for (int c = 0; c < pitch; c += 16) {
				//transform
				V16f x = m02;
				x = _mm512_fmadd_ps(iy, m01, x);
				x = _mm512_fmadd_ps(ix, m00, x);
				V16f y = m12;
				y = _mm512_fmadd_ps(iy, m11, y);
				y = _mm512_fmadd_ps(ix, m10, y);
				ix += 16;

				//check within image bounds
				__mmask16 mask = 0xFFFF;
				V16f check;
				check = ps_zero;
				mask &= _mm512_cmp_ps_mask(x, check, _CMP_GE_OS); //greater equal
				mask &= _mm512_cmp_ps_mask(y, check, _CMP_GE_OS); //greater equal
				check = _mm512_set1_ps(float(input.w() - 1));
				mask &= _mm512_cmp_ps_mask(x, check, _CMP_LE_OS); //less equal
				check = _mm512_set1_ps(float(input.h() - 1));
				mask &= _mm512_cmp_ps_mask(y, check, _CMP_LE_OS); //less equal

				//compute fractions
				V16f flx = _mm512_floor_ps(x);
				V16f fly = _mm512_floor_ps(y);
				V16f dx = _mm512_sub_ps(x, flx);
				V16f dy = _mm512_sub_ps(y, fly);

				//index to load f00
				__m512i ix = _mm512_cvtps_epi32(flx);
				__m512i iy = _mm512_cvtps_epi32(fly);
				idx = _mm512_mullo_epi32(epi_stride, iy);    //idx = stride * row
				idx = _mm512_add_epi32(idx, ix);             //idx += col
				V16f f00 = _mm512_mask_i32gather_ps(ps_zero, mask, idx, input.data(), 4);

				//index to load f01
				__mmask16 maskdx = _mm512_cmp_ps_mask(dx, ps_zero, _CMP_NEQ_OS); //not equal
				__m512i idx2 = _mm512_mask_add_epi32(idx, maskdx, idx, epi_one);
				V16f f01 = _mm512_mask_i32gather_ps(ps_zero, mask, idx2, input.data(), 4);

				//index to load f10
				__mmask16 maskdy = _mm512_cmp_ps_mask(dy, ps_zero, _CMP_NEQ_OS); //not equal
				__m512i idx3 = _mm512_mask_add_epi32(idx, maskdy, idx, epi_stride);
				V16f f10 = _mm512_mask_i32gather_ps(ps_zero, mask, idx3, input.data(), 4);

				//index to load f11
				__m512i idx4 = _mm512_mask_add_epi32(idx3, maskdx, idx3, epi_one);
				V16f f11 = _mm512_mask_i32gather_ps(ps_zero, mask, idx4, input.data(), 4);

				V16f result = interpolate(f00, f10, f01, f11, dx, dy);
				result.storeu(dest.addr(r, c), mask);
			}
		}
	});
	mPool.wait();
}

void AvxFrame::unsharp(const AvxMatf& warped, AvxMatf& gauss, float unsharp, AvxMatf& out) {
	//util::ConsoleTimer ic("avx unsharp");
	for (int threadIdx = 0; threadIdx < mData.cpuThreads; threadIdx++) mPool.add([&, threadIdx] {
		V16f zero = 0.0f;
		V16f one = 1.0f;
		for (int r = threadIdx; r < mData.h; r += mData.cpuThreads) {
			for (int c = 0; c < mData.w; c += 16) {
				V16f ps_warped = warped.addr(r, c);
				V16f ps_gauss = gauss.addr(r, c);
				V16f ps_unsharped = (ps_warped + (ps_warped - ps_gauss) * unsharp).clamp(zero, one);
				ps_unsharped.storeu(out.addr(r, c));
			}
		}
	});
	mPool.wait();
}

void AvxFrame::write(ImageYuv& dest) const {
	for (int z = 0; z < 3; z++) {
		for (int r = 0; r < mData.h; r++) {
			for (int c = 0; c < mData.w; c += 16) {
				V16f out = mOutput[z].addr(r, c);
				__m512i chars32 = _mm512_cvt_roundps_epi32(out * 255.0f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
				_mm512_mask_cvtusepi32_storeu_epi8(dest.addr(z, r, c), 0xFFFF, chars32);
			}
		}
	}
}

void AvxFrame::write(ImageNV12& image) const {
	//util::ConsoleTimer ic("avx write nv12");
	//Y-Plane
	for (int r = 0; r < mData.h; r++) {
		unsigned char* dest = image.addr(0, r, 0);
		for (int c = 0; c < mData.w; c += 16) {
			V16f out = mOutput[0].addr(r, c);
			__m512i chars32 = _mm512_cvt_roundps_epi32(out * 255.0f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
			_mm512_mask_cvtusepi32_storeu_epi8(dest + c, 0xFFFF, chars32);
		}
	}

	//U-V-Planes
	unsigned char* dest = image.addr(0, 0, 0) + image.h * image.stride;
	for (int rr = 0; rr < mData.h / 2; rr++) {
		int r = rr * 2;
		__m512i a, b, x, sumU, sumV, sum;
		V16f vec;

		for (int c = 0; c < mData.w; c += 16) {
			vec = mOutput[1].addr(r, c);
			a = _mm512_cvt_roundps_epi32(vec * 255.0f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
			vec = mOutput[1].addr(r + 1, c);
			b = _mm512_cvt_roundps_epi32(vec * 255.0f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
			sumU = _mm512_add_epi32(a, b);

			vec = mOutput[2].addr(r, c);
			a = _mm512_cvt_roundps_epi32(vec * 255.0f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
			vec = mOutput[2].addr(r + 1, c);
			b = _mm512_cvt_roundps_epi32(vec * 255.0f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
			sumV = _mm512_add_epi32(a, b);

			//cross over and combine
			x = _mm512_shuffle_epi32(sumU, _MM_PERM_DDBB);
			x = _mm512_mask_shuffle_epi32(x, 0b10101010'10101010, sumV, _MM_PERM_CCAA);
			//combine without crossing
			sum = _mm512_mask_blend_epi32(0b10101010'10101010, sumU, sumV);
			//add the blocks
			sum = _mm512_add_epi32(sum, x);
			//divide sum by 4
			sum = _mm512_srli_epi32(sum, 2);
			__m128i uv8 = _mm512_cvtepi32_epi8(sum);
			_mm_storeu_epi8(dest + c, uv8);
		}

		dest += image.stride;
	}
}

//from uchar yuv to uchar rgb
void AvxFrame::yuvToRgb(const uchar* y, const uchar* u, const uchar* v, int h, int w, int stride, ImageBaseRgb& dest) const {
	auto vidx = dest.indexRgba();
	V16f factorU = { fu[vidx[0]], fu[vidx[1]], fu[vidx[2]], fu[vidx[3]] };
	V16f factorV = { fv[vidx[0]], fv[vidx[1]], fv[vidx[2]], fv[vidx[3]] };

	for (int r = 0; r < h; r++) {
		for (int c = 0; c < w; c += 4) {
			V4f vecy = y + r * stride + c;
			V4f vecu = u + r * stride + c;
			V4f vecv = v + r * stride + c;
			avx::yuvToRgbaPacked(vecy, vecu, vecv, dest.addr(0, r, c), factorU, factorV);
		}
	}
}

//from float yuv to uchar rgb
void AvxFrame::yuvToRgb(const float* y, const float* u, const float* v, int h, int w, int stride, ImageBaseRgb& dest) const {
	auto vidx = dest.indexRgba();
	V16f factorU = { fu[vidx[0]], fu[vidx[1]], fu[vidx[2]], fu[vidx[3]] };
	V16f factorV = { fv[vidx[0]], fv[vidx[1]], fv[vidx[2]], fv[vidx[3]] };

	V4f f = 255.0f;
	for (int r = 0; r < h; r++) {
		for (int c = 0; c < w; c += 4) {
			V4f vecy = y + r * stride + c;
			V4f vecu = u + r * stride + c;
			V4f vecv = v + r * stride + c;
			avx::yuvToRgbaPacked(vecy * f, vecu * f, vecv * f, dest.addr(0, r, c), factorU, factorV);
		}
	}
}
