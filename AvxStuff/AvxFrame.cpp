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

using namespace avx;

AvxFrame::AvxFrame(CoreData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool) :
	FrameExecutor(data, deviceInfo, frame, pool)
{}

void AvxFrame::init() {
	assert(mDeviceInfo.getType() == DeviceType::AVX && "device type must be AVX here");
	int w = mData.w;
	int h = mData.h;

	mReadBuffer = ImageYuv(h, w, mData.stride);
	for (int i = 0; i < mData.bufferCount; i++) mInput.emplace_back(h, w, mData.stride);
	for (int i = 0; i < mData.pyramidCount; i++) mPyr.emplace_back(mData.pyramidRowCount, mData.stride, 0.0f);

	mBackground = AvxMatf(h, mData.stride4);
	mWarped = AvxMatf(h, mData.stride4);
	mOutput = AvxMatf(h, mData.stride4);

	mFilterBuffer = AvxMatf(w, util::alignValue(h, 64)); //transposed
	mFilterResult = AvxMatf(h, util::alignValue(w, 64));

	mFilterBuffer4 = AvxMatf(w, util::alignValue(h * 4, 64)); //transposed
	mFilterResult4 = AvxMatf(h, util::alignValue(w * 4, 64));

	//fill background color
	for (int r = 0; r < h; r++) {
		for (int c = 0; c < w * 4; c += 4) {
			for (int k = 0; k < 4; k++) {
				mBackground.at(r, c + k) = mData.bgcol4[k];
			}
		}
	}
	mBackground.copyTo(mWarped);

	//allocate storage for luma sum
	luma.resize(mData.h * 16ull);
}


//---------------------------------
// ---- main functions ------------
//---------------------------------

Image8& AvxFrame::inputDestination(int64_t frameIndex) {
	return mReadBuffer;
}

void AvxFrame::inputData(int64_t frameIndex) {
	size_t idx = frameIndex % mInput.size();
	std::swap(mReadBuffer, mInput[idx]);
}

void AvxFrame::createPyramid(int64_t frameIndex, std::span<int> hist, AffineDataFloat trf, bool warp) {
	//util::ConsoleTimer timer("avx pyramid " + std::to_string(frameIndex));
	size_t pyrIdx = frameIndex % mPyr.size();
	AvxMatf& Y = mPyr[pyrIdx];
	Y.frameIndex = frameIndex;

	//fill topmost level of pyramid
	size_t yuvIdx = frameIndex % mInput.size();
	ImageYuv& yuv = mInput[yuvIdx];

	//convert Y to float and sum up luma
	V16f f = 1.0f / 255.0f;

	auto func = [&] (size_t r) {
		uchar* srcPtr = yuv.row(r);
		std::fill(srcPtr + mData.w, srcPtr + mData.stride, 0); //decoder puts random bytes into padding area
		float* destPtr = mFilterResult.addr(r, 0);

		for (int c = 0; c < mData.w; c += 16) {
			__m512i y = _mm512_cvtepu8_epi32(_mm_loadu_epi8(srcPtr + c));
			V16f result = _mm512_cvtepi32_ps(y);
			result *= f;
			result.storeu(destPtr + c);
		}
	};
	mPool.addAndWait(func, 0, mData.h);

	//write first pyramid level
	if (warp) {
		//transform input
		Y.fill(0.0f);
		warpBack1(trf, mFilterResult, Y);

	} else {
		//filter first level
		filter1(mFilterResult, mData.h, mData.w, mFilterBuffer, mFilterKernels[0]);
		filter1(mFilterBuffer, mData.w, mData.h, Y, mFilterKernels[0]);
	}

	//create pyramid levels below by downsampling level above
	int r = 0;
	int hh = mData.h;
	int ww = mData.w;
	for (size_t z = 1; z < mData.pyramidLevels; z++) {
		const float* src = Y.row(r);
		float* dest = Y.row(r + hh);
		r += hh;
		hh /= 2;
		ww /= 2;
		downsample(src, hh, ww, Y.w(), dest, Y.w());
	}

	//create histogram
	std::fill(hist.begin(), hist.end(), 0);
	for (int r = 0; r < mData.h; r++) {
		const uchar* src = yuv.row(r);
		for (int c = 0; c < mData.w; c++) {
			hist[src[c]]++;
		}
	}
}

void AvxFrame::adjustPyramid(int64_t frameIndex, float gamma) {

}

void AvxFrame::outputData(int64_t frameIndex, AffineDataFloat trf) {
	//util::ConsoleTimer ic("avx output");
	size_t idx = frameIndex % mInput.size();
	const ImageYuv& input = mInput[idx];
	assert(input.index == frameIndex && "wrong frame");

	yuvToFloat4(input, mFilterResult4);
	if (mData.bgmode == BackgroundMode::COLOR) mBackground.copyTo(mWarped);
	warpBack4(trf, mFilterResult4, mWarped);
	filter4(mWarped, mData.h, mData.w, mFilterBuffer4);
	filter4(mFilterBuffer4, mData.w, mData.h, mFilterResult4);
	unsharp4(mWarped, mFilterResult4, mOutput);
}

void AvxFrame::getOutput(int64_t frameIndex, Image8& image) const {
	if (image.imageType() == ImageType::VUYX) {
		writeVuyx(image);

	} else if (image.imageType() == ImageType::YUV) {
		writeYuv(image);

	} else if (image.colorBase() == ColorBase::RGB) {
		vuyxToRgba(mOutput, image);
	}
	image.setIndex(frameIndex);
}

bool AvxFrame::getOutput(int64_t frameIndex, Image8& image, int cudaNv12stride, unsigned char* cudaNv12ptr) const {
	writeNV12(image);
	return true;
}

Matf AvxFrame::getTransformedOutput() const {
	return mWarped.matShare().subMat(0, 0, mData.h, mData.w * 4ull);
}

Matf AvxFrame::getPyramid(int64_t frameIndex) const {
	size_t pyrIdx = frameIndex % mData.pyramidCount;
	return mPyr[pyrIdx].matShare().subMat(0, 0, mData.pyramidRowCount, mData.w);
}

void AvxFrame::getInput(int64_t frameIndex, Image8& image) const {
	size_t idx = frameIndex % mInput.size();
	if (image.colorBase() == ColorBase::RGB) {
		yuvToRgba(mInput[idx], image);

	} else if (image.colorBase() == ColorBase::YUV) {
		mInput[idx].convertTo(image);
	}
}


//----------------------------------------------------
// ----- IMPLEMENTATION DETAILS ----------------------
//----------------------------------------------------


static V16f rotsum(V16f x, std::span<V16f> ks) {
	V16f sum = x * ks[0];
	for (int i = 1; i < ks.size(); i++) {
		x = x.rot<1>();
		sum = _mm512_fmadd_ps(x, ks[i], sum);
	}
	return sum;
}

void AvxFrame::filter1(const AvxMatf& src, int h, int w, AvxMatf& dest, std::span<V16f> ks) {
	//util::ConsoleTimer ic("avx filter " + std::to_string(w) + "x" + std::to_string(h));
	__m512i vw = _mm512_set1_epi32(dest.w());
	__m512i idx = _mm512_loadu_epi32(iotas.i32x16);
	__m512i vidx = _mm512_mullo_epi32(vw, idx);
	__mmask16 mask = 0x0FFF;

	for (int threadIdx = 0; threadIdx < mData.cpuThreads; threadIdx++) mPool.add([&, threadIdx] {
		V16f x, result;

		for (int r = threadIdx; r < h; r += mData.cpuThreads) {
			const float* row = src.addr(r, 0);

			//write first points adhering to border
			//first broadcast border point, then overwrite other points
			x = _mm512_mask_loadu_ps(_mm512_set1_ps(row[0]), 0xFFFC, row - 2);
			result = rotsum(x, ks);
			_mm512_mask_i32scatter_ps(dest.addr(0, r), mask, vidx, result, 4);

			//main loop
			for (int c = 12; c < w - 12; c += 12) {
				x = V16f(row + c - 2);
				result = rotsum(x, ks);
				_mm512_mask_i32scatter_ps(dest.addr(c, r), mask, vidx, result, 4);
			}

			//write last points adhering to border
			x = _mm512_mask_loadu_ps(_mm512_set1_ps(row[w - 1]), 0x3FFF, row + w - 14);
			result = rotsum(x, ks);
			_mm512_mask_i32scatter_ps(dest.addr(w - 12, r), mask, vidx, result, 4);
		}
	});
	mPool.wait();
}

static V16f rotsum(V16f a, V16f b, std::span<V16f> ks) {
	V16f x;
	V16f sum = a * ks[0];
	x = _mm512_castsi512_ps(_mm512_alignr_epi32(_mm512_castps_si512(b), _mm512_castps_si512(a), 4));
	sum = _mm512_fmadd_ps(x, ks[1], sum);
	x = _mm512_castsi512_ps(_mm512_alignr_epi32(_mm512_castps_si512(b), _mm512_castps_si512(a), 8));
	sum = _mm512_fmadd_ps(x, ks[2], sum);
	x = _mm512_castsi512_ps(_mm512_alignr_epi32(_mm512_castps_si512(b), _mm512_castps_si512(a), 12));
	sum = _mm512_fmadd_ps(x, ks[3], sum);
	return _mm512_fmadd_ps(b, ks[4], sum);
}

void AvxFrame::filter4(const AvxMatf& src, int h, int w, AvxMatf& dest) {
	__m512i strides = _mm512_set1_epi32(dest.w());
	__m512i idx = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
	__m512i vidx = _mm512_mullo_epi32(strides, idx);
	idx = _mm512_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3);
	vidx = _mm512_add_epi32(vidx, idx);

	auto func = [&] (size_t r) {
		V16f a, b, result;
		const float* row = src.addr(r, 0);

		a = V16f(row[0], row[1], row[2], row[3], row[0], row[1], row[2], row[3], row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]);
		b = V16f(row + 8);
		result = rotsum(a, b, mFilterKernels4);
		_mm512_i32scatter_ps(dest.addr(0, r * 4ull), vidx, result, 4);

		//main loop
		a = V16f(row);
		for (int c = 2; c < w - 2; c += 4) {
			b = V16f(row + c * 4 + 8);
			result = rotsum(a, b, mFilterKernels4);
			_mm512_i32scatter_ps(dest.addr(c, r * 4ull), vidx, result, 4);
			a = b;
		}

		const float* ptr = row + w * 4 - 24;
		a = V16f(ptr);
		b = V16f(ptr[16], ptr[17], ptr[18], ptr[19], ptr[20], ptr[21], ptr[22], ptr[23], ptr[20], ptr[21], ptr[22], ptr[23], ptr[20], ptr[21], ptr[22], ptr[23]);
		result = rotsum(a, b, mFilterKernels4);
		_mm512_i32scatter_ps(dest.addr(w - 4ull, r * 4ull), vidx, result, 4);
	};
	mPool.addAndWait(func, 0, h);
}

void AvxFrame::downsample(const float* srcptr, int h, int w, int stride, float* destptr, int destStride) {
	const __m512i idx1 = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30); //sequence 0, 2, 4, ..., 30
	const __m512i idx2 = _mm512_add_epi32(idx1, _mm512_set1_epi32(1)); // sequence 1, 3, 5, ..., 31
	const V16f f = 0.5f;

	auto func = [&] (size_t r) {
		for (int c = 0; c < w; c += 16) {
			c = std::min(c, w - 16);
			const float* src = srcptr + r * 2 * stride + c * 2;
			V16f x1 = src;
			V16f x2 = src + 16;
			V16f y1 = src + stride;
			V16f y2 = src + stride + 16;

			V16f f00 = _mm512_permutex2var_ps(x1, idx1, x2);
			V16f f01 = _mm512_permutex2var_ps(x1, idx2, x2);
			V16f f10 = _mm512_permutex2var_ps(y1, idx1, y2);
			V16f f11 = _mm512_permutex2var_ps(y1, idx2, y2);
			V16f result = interpolate(f00, f10, f01, f11, f, f, f, f);

			float* dest = destptr + r * destStride + c;
			result.storeu(dest);
		}
	};
	mPool.addAndWait(func, 0, h);
}

void AvxFrame::yuvToFloat4(const ImageYuv& yuv, AvxMatf& dest) {
	//util::ConsoleTimer ic("avx yuv to float");
	V16f f = 1.0f / 255.0f;
	char val = (char) 255;

	auto func = [&] (size_t r) {
		__m128i v = _mm_set1_epi8(val);
		float* destPtr = dest.addr(r, 0);
		for (int c = 0; c < yuv.w(); c += 4) {
			v = _mm_mask_expandloadu_epi8(v, 0x4444, yuv.addr(0, r, c)); //y
			v = _mm_mask_expandloadu_epi8(v, 0x2222, yuv.addr(1, r, c)); //u
			v = _mm_mask_expandloadu_epi8(v, 0x1111, yuv.addr(2, r, c)); //v
			V16f vuyxData = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v));
			vuyxData *= f;
			vuyxData.storeu(destPtr + c * 4);
		}
	};
	mPool.addAndWait(func, 0, yuv.h());
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

void AvxFrame::computeStart(int64_t frameIndex, std::span<PointResult> results) {}

void AvxFrame::computeTerminate(int64_t frameIndex, std::span<PointResult> results) {
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
		V8d iota = iotas.dx8;

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
						if (err < mData.compMaxTol) result = PointResultType::SUCCESS_ABSOLUTE_ERR;
						if (std::abs(err - bestErr) / bestErr < mData.compMaxTol * mData.compMaxTol) result = PointResultType::SUCCESS_STABLE_ITER;
						if (err < bestErr) bestErr = err;
						iter++;
						if (iter == mData.compMaxIter && result == PointResultType::RUNNING) result = PointResultType::FAIL_ITERATIONS;
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
				double length = std::sqrt(u * u + v * v);
				results[idx] = { idx, ix0, iy0, x0, y0, u * fdir, v * fdir, result, zp, direction, length };
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

void AvxFrame::warpBack4(const AffineDataFloat& trf, const AvxMatf& input, AvxMatf& dest) {
	//util::ConsoleTimer ic("avx warp4");
	//transform parameters
	V16f m00 = trf.m00;
	V16f m01 = trf.m01;
	V16f m02 = trf.m02;
	V16f m10 = trf.m10;
	V16f m11 = trf.m11;
	V16f m12 = trf.m12;

	for (int threadIdx = 0; threadIdx < mData.cpuThreads; threadIdx++) mPool.add([&, threadIdx] {
		for (int r = threadIdx; r < mData.h; r += mData.cpuThreads) {
			V16f ixf = iotas.fx16;
			V16f iyf = float(r);
			V16f fstride = float(input.w());
			V16f ps_zero = 0.0f;
			__m512i epi_one = _mm512_set1_epi32(1);
			__m512i epi_four = _mm512_set1_epi32(4);
			__m512i epi_stride = _mm512_set1_epi32(input.w());
			std::array<int, 16> idx00, idx01, idx10, idx11;

			for (int c = 0; c < mData.w; c += 16) {
				//transform
				V16f x = m02;
				x = _mm512_fmadd_ps(iyf, m01, x);
				x = _mm512_fmadd_ps(ixf, m00, x);
				V16f y = m12;
				y = _mm512_fmadd_ps(iyf, m11, y);
				y = _mm512_fmadd_ps(ixf, m10, y);
				ixf += 16;

				//check within image bounds
				__mmask16 mask = 0xFFFF;
				V16f check;
				mask &= _mm512_cmp_ps_mask(x, ps_zero, _CMP_GE_OS); //greater equal
				mask &= _mm512_cmp_ps_mask(y, ps_zero, _CMP_GE_OS); //greater equal
				check = mData.w - 1.0f;
				mask &= _mm512_cmp_ps_mask(x, check, _CMP_LE_OS); //less equal
				check = mData.h - 1.0f;
				mask &= _mm512_cmp_ps_mask(y, check, _CMP_LE_OS); //less equal

				//compute fractions
				V16f flx = _mm512_floor_ps(x);
				V16f fly = _mm512_floor_ps(y);
				V16f dx16 = _mm512_sub_ps(x, flx);
				V16f dy16 = _mm512_sub_ps(y, fly);

				//index to load f00
				V16f idxf = fly * fstride + flx * 4.0f;
				__m512i idx0 = _mm512_cvtps_epi32(idxf);
				_mm512_storeu_epi32(idx00.data(), idx0);

				//index to load f01
				__mmask16 maskdx = _mm512_cmp_ps_mask(dx16, ps_zero, _CMP_NEQ_OS); //not equal
				__m512i idx1 = _mm512_mask_add_epi32(idx0, maskdx, idx0, epi_four);
				_mm512_storeu_epi32(idx01.data(), idx1);

				//index to load f10
				__mmask16 maskdy = _mm512_cmp_ps_mask(dy16, ps_zero, _CMP_NEQ_OS); //not equal
				idx1 = _mm512_mask_add_epi32(idx0, maskdy, idx0, epi_stride);
				_mm512_storeu_epi32(idx10.data(), idx1);

				//index to load f11
				idx1 = _mm512_mask_add_epi32(idx1, maskdx, idx1, epi_four);
				_mm512_storeu_epi32(idx11.data(), idx1);

				__m512i epi_idx = _mm512_setzero_si512();
				for (size_t i = 0; i < 16; i++) {
					if (mask & 1) {
						V4f f00 = input.data() + idx00[i];
						V4f f01 = input.data() + idx01[i];
						V4f f10 = input.data() + idx10[i];
						V4f f11 = input.data() + idx11[i];

						V4f one = 1.0f;
						V4f dx = _mm512_castps512_ps128(_mm512_permutexvar_ps(epi_idx, dx16));
						V4f dy = _mm512_castps512_ps128(_mm512_permutexvar_ps(epi_idx, dy16));
						V4f result = (one - dx) * (one - dy) * f00 + (one - dx) * dy * f10 + dx * (one - dy) * f01 + dx * dy * f11;
						result.storeu(dest.addr(r, (c + i) * 4));
					}
					mask >>= 1;
					epi_idx = _mm512_add_epi32(epi_idx, epi_one);
				}
			}
		}
	});
	mPool.wait();
}

void AvxFrame::warpBack1(const AffineDataFloat& trf, const AvxMatf& input, AvxMatf& dest) {
	//util::ConsoleTimer ic("avx warp");
	//transform parameters
	const V16f m00 = trf.m00;
	const V16f m01 = trf.m01;
	const V16f m02 = trf.m02;
	const V16f m10 = trf.m10;
	const V16f m11 = trf.m11;
	const V16f m12 = trf.m12;

	for (int threadIdx = 0; threadIdx < mData.cpuThreads; threadIdx++) mPool.add([&, threadIdx] {
		for (int r = threadIdx; r < mData.h; r += mData.cpuThreads) {
			V16f ixf = iotas.fx16;
			V16f iyf = float(r);
			V16f ps_zero = _mm512_set1_ps(0.0f);
			V16f fstride = float(input.w());
			__m512i epi_one = _mm512_set1_epi32(1);
			__m512i epi_stride = _mm512_set1_epi32(input.w());
			__m512i idx;
			
			for (int c = 0; c < mData.w; c += 16) {
				//transform
				V16f x = m02;
				x = _mm512_fmadd_ps(iyf, m01, x);
				x = _mm512_fmadd_ps(ixf, m00, x);
				V16f y = m12;
				y = _mm512_fmadd_ps(iyf, m11, y);
				y = _mm512_fmadd_ps(ixf, m10, y);
				ixf += 16;

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
				V16f fidx = fly * fstride + flx;
				idx = _mm512_cvtps_epi32(fidx);
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

void AvxFrame::unsharp4(const AvxMatf& warped, AvxMatf& gauss, AvxMatf& out) {
	//util::ConsoleTimer ic("avx unsharp");
	for (int threadIdx = 0; threadIdx < mData.cpuThreads; threadIdx++) mPool.add([&, threadIdx] {
		V16f unsharp(mData.unsharp4[0], mData.unsharp4[1], mData.unsharp4[2], mData.unsharp4[3]);
		V16f zero = 0.0f;
		V16f one = 1.0f;
		for (int r = threadIdx; r < mData.h; r += mData.cpuThreads) {
			for (int c = 0; c < mData.w * 4; c += 16) {
				V16f ps_warped = warped.addr(r, c);
				V16f ps_gauss = gauss.addr(r, c);
				V16f ps_unsharped = (ps_warped + (ps_warped - ps_gauss) * unsharp).clamp(zero, one);
				ps_unsharped.storeu(out.addr(r, c));
			}
		}
	});
	mPool.wait();
}

void AvxFrame::writeVuyx(Image8& dest) const {
	assert(dest.imageType() == ImageType::VUYX && "invalid image");
	for (int threadIdx = 0; threadIdx < mData.cpuThreads; threadIdx++) mPool.add([&, threadIdx] {
		for (int r = threadIdx; r < mData.h; r += mData.cpuThreads) {
			const float* srcPtr = mOutput.addr(r, 0);
			uchar* destPtr = dest.row(r);
			for (int c = 0; c < mData.w * 4; c += 16) {
				V16f out = srcPtr + c;
				__m512i chars32 = _mm512_cvt_roundps_epi32(out * 255.0f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
				_mm512_mask_cvtusepi32_storeu_epi8(destPtr + c, 0xFFFF, chars32);
			}
		}
	});
	mPool.wait();
}

void AvxFrame::writeYuv(Image8& dest) const {
	assert(dest.imageType() == ImageType::YUV && "invalid image");
	for (int threadIdx = 0; threadIdx < mData.cpuThreads; threadIdx++) mPool.add([&, threadIdx] {
		__m512i idx = _mm512_setr_epi32(2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12, 0, 0, 0, 0);
		for (int r = threadIdx; r < mData.h; r += mData.cpuThreads) {
			const float* srcPtr = mOutput.addr(r, 0);
			for (int c = 0; c < mData.w * 4; c += 16) {
				V16f out = srcPtr + c;
				__m512i chars32 = _mm512_cvt_roundps_epi32(out * 255.0f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
				chars32 = _mm512_permutexvar_epi32(idx, chars32);
				__m128i yuv = _mm512_cvtusepi32_epi8(chars32);

				uchar* ptr = dest.data() + r * dest.stride() + c / 4;
				_mm_mask_storeu_epi8(ptr, 0x000F, yuv);
				ptr += dest.h() * dest.stride() - 4;
				_mm_mask_storeu_epi8(ptr, 0x00F0, yuv);
				ptr += dest.h() * dest.stride() - 4;
				_mm_mask_storeu_epi8(ptr, 0x0F00, yuv);
			}
		}
	});
	mPool.wait();
}

void AvxFrame::writeNV12(Image8& dest) const {
	assert(dest.imageType() == ImageType::NV12 && "invalid image");
	__m512i idx = _mm512_setr_epi32(1, 5, 0, 4, 9, 13, 8, 12, 2, 6, 10, 14, 0, 0, 0, 0);
	for (int r = 0; r < mData.h; r += 2) {
		for (int c = 0; c < mData.w; c += 4) {
			V16f vuyx;
			vuyx = mOutput.addr(r, c * 4);
			__m512i epi0 = _mm512_cvt_roundps_epi32(vuyx * 255.0f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
			epi0 = _mm512_permutexvar_epi32(idx, epi0);
			_mm512_mask_cvtusepi32_storeu_epi8(dest.row(r) + c - 8, 0x0F00, epi0);

			vuyx = mOutput.addr(r + 1, c * 4);
			__m512i epi1 = _mm512_cvt_roundps_epi32(vuyx * 255.0f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
			epi1 = _mm512_permutexvar_epi32(idx, epi1);
			_mm512_mask_cvtusepi32_storeu_epi8(dest.row(r + 1ull) + c - 8, 0x0F00, epi1);
			
			//combine u and v
			__m256i uv0 = _mm512_castsi512_si256(_mm512_add_epi32(epi0, epi1));
			__m256i uv1 = _mm256_permute2f128_si256(uv0, uv0, mask8(1, 0, 0, 0, 1, 0, 0, 0));
			__m256i uv = _mm256_hadd_epi32(uv0, uv1);
			uv = _mm256_srli_epi32(uv, 2);
			_mm256_mask_cvtusepi32_storeu_epi8(dest.row(mData.h + r / 2) + c, 0xF, uv);
		}
	}
}

//from uchar yuv to uchar rgba
void AvxFrame::yuvToRgba(const ImageYuv& yuv, Image8& dest) const {
	assert(dest.colorBase() == ColorBase::RGB && "invalid color format");
	//order of rgb colors
	auto vidx = dest.colorIndex();
	V16f fu = { mFactorU[vidx[0]], mFactorU[vidx[1]], mFactorU[vidx[2]], mFactorU[vidx[3]] };
	V16f fv = { mFactorV[vidx[0]], mFactorV[vidx[1]], mFactorV[vidx[2]], mFactorV[vidx[3]] };

	for (int threadIdx = 0; threadIdx < mData.cpuThreads; threadIdx++) mPool.add([&, threadIdx] {
		for (int r = threadIdx; r < yuv.h(); r += mData.cpuThreads) {
			uchar* destPtr = dest.row(r);
			for (int c = 0; c < yuv.w(); c += 4) {
				__m512 y = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_maskz_expandloadu_epi8(0x1111, yuv.addr(0, r, c))));
				y = _mm512_permute_ps(y, 0);
				__m512 u = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_maskz_expandloadu_epi8(0x1111, yuv.addr(1, r, c))));
				u = _mm512_permute_ps(u, 0);
				__m512 v = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_maskz_expandloadu_epi8(0x1111, yuv.addr(2, r, c))));
				v = _mm512_permute_ps(v, 0);
				avx::yuvToRgbaPacked(y, u, v, destPtr + c * 4, fu, fv);
			}
		}
	});
	mPool.wait();
}


//from float vuyx to uchar rgba
void AvxFrame::vuyxToRgba(const AvxMatf& vuyx, Image8& dest) const {
	assert(dest.colorBase() == ColorBase::RGB && "invalid color format");
	//order of rgb colors
	auto vidx = dest.colorIndex();
	V16f fu = { mFactorU[vidx[0]], mFactorU[vidx[1]], mFactorU[vidx[2]], mFactorU[vidx[3]] };
	V16f fv = { mFactorV[vidx[0]], mFactorV[vidx[1]], mFactorV[vidx[2]], mFactorV[vidx[3]] };
	V16f f = 255.0f;

	for (int threadIdx = 0; threadIdx < mData.cpuThreads; threadIdx++) mPool.add([&, threadIdx] {
		int destW = dest.w() * 4;
		for (int r = threadIdx; r < vuyx.h(); r += mData.cpuThreads) {
			const float* srcPtr = vuyx.addr(r, 0);
			uchar* destPtr = dest.row(r);
			for (int c = 0; c < destW - 16; c += 16) {
				V16f vuyx4 = srcPtr + c;
				V16f y = _mm512_permute_ps(vuyx4, mask8(2, 2, 2, 2));
				V16f u = _mm512_permute_ps(vuyx4, mask8(1, 1, 1, 1));
				V16f v = _mm512_permute_ps(vuyx4, mask8(0, 0, 0, 0));
				avx::yuvToRgbaPacked(y * f, u * f, v * f, destPtr + c, fu, fv);
			}
			V16f vuyx4 = srcPtr + destW - 16;
			V16f y = _mm512_permute_ps(vuyx4, mask8(2, 2, 2, 2));
			V16f u = _mm512_permute_ps(vuyx4, mask8(1, 1, 1, 1));
			V16f v = _mm512_permute_ps(vuyx4, mask8(0, 0, 0, 0));
			avx::yuvToRgbaPacked(y * f, u * f, v * f, destPtr + destW - 16, fu, fv);
		}
	});
	mPool.wait();
}
