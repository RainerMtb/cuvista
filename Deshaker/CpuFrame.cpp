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

#include "CpuFrame.hpp"

struct FilterKernel {
	static const int maxSize = 8;
	int siz;
	float k[maxSize];
};

FilterKernel filterKernels[4] = {
	{5, {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f}},
	{3, {0.25f, 0.5f, 0.25f}},
	{3, {0.25f, 0.5f, 0.25f}},
	{3, {-0.5f, 0.0f, 0.5f}},
};


//constructor
CpuFrame::CpuFrame(MainData& data, MovieReader& reader, MovieWriter& writer) : 
	MovieFrame(data, reader, writer) 
{
	//buffer to hold input frames in yuv format
	mYUV.assign(data.bufferCount, ImageYuv());

	//init cpuFrameItem structures, allocate pyramid
	for (int i = 0; i < data.pyramidCount; i++) mPyr.emplace_back(data);

	//init storage for previous output frame to background colors
	mPrevOut.push_back(Matf::values(data.h, data.w, data.bgcol_yuv.colors[0]));
	mPrevOut.push_back(Matf::values(data.h, data.w, data.bgcol_yuv.colors[1]));
	mPrevOut.push_back(Matf::values(data.h, data.w, data.bgcol_yuv.colors[2]));

	//buffer for output and pyramid creation
	mBuffer.assign(4, Matf::allocate(data.h, data.w));
	mFilterBuffer = Matf::allocate(data.h, data.w);
	mFilterResult = Matf::allocate(data.h, data.w);
	mYuvPlane = Matf::allocate(data.h, data.w);
}

//construct data for one pyramid
CpuFrame::CpuFrameItem::CpuFrameItem(MainData& data) {
	//allocate matrices for pyramid
	for (int z = 0; z <= data.zMax; z++) {
		int hz = data.h >> z;
		int wz = data.w >> z;
		mY.push_back(Matf::allocate(hz, wz));
	}
}

//read input frame and put into buffer
void CpuFrame::inputData() {
	size_t idx = mBufferFrame.index % mYUV.size();
	mYUV[idx] = mBufferFrame;
}

void CpuFrame::createPyramid(int64_t frameIndex) {
	//ConsoleTimer ic("pyramid");
	size_t pyrIdx = frameIndex % mPyr.size();
	CpuFrameItem& frame = mPyr[pyrIdx];
	frame.frameIndex = frameIndex;

	//fill topmost level of pyramid
	size_t yuvIdx = frameIndex % mYUV.size();
	ImageYuv& yuv = mYUV[yuvIdx];
	float f = 1.0f / 255.0f;
	Matf& y0 = frame.mY[0];
	y0.setValues([&] (size_t r, size_t c) { return yuv.at(0, r, c) * f; }, mPool);
	//y0.filter1D(k, y0, mFilterBuffer, mPool);

	//create pyramid levels below by downsampling level above
	for (size_t z = 0; z < mData.zMax; z++) {
		Matf& y = frame.mY[z];
		//gauss filtering
		Matf filterTemp = mFilterBuffer.reuse(y.rows(), y.cols());
		Matf mat = mFilterResult.reuse(y.rows(), y.cols());
		y.filter1D(filterKernels[0].k, filterKernels[0].siz, Matf::Direction::HORIZONTAL, filterTemp, mPool);
		filterTemp.filter1D(filterKernels[0].k, filterKernels[0].siz, Matf::Direction::VERTICAL, mat, mPool);

		//if (z == 0) std::printf("cpu %.14f %.14f %.14f %.14f %.14f\n", y.at(30, 28), y.at(30, 29), y.at(30, 30), y.at(30, 31), y.at(30, 32));
		//if (z == 0) mat.saveAsBinary("f:/buf_c.dat");
		//downsampling
		auto func = [&] (size_t r, size_t c) { return mat.interp2(c * 2, r * 2, 0.5f, 0.5f); };
		Matf& dest = frame.mY[z + 1];
		dest.setArea(func, mPool);
	}
	//if (status.frameInputIndex == 1) frame.Y[1].saveAsBinary("f:/cpu.dat");
}

void CpuFrame::computeStart(int64_t frameIndex) {}

void CpuFrame::computeTerminate(int64_t frameIndex) {
	size_t pyrIdx = frameIndex % mPyr.size();
	size_t pyrIdxPrev = (frameIndex - 1) % mPyr.size();
	CpuFrameItem& frame = mPyr[pyrIdx];
	CpuFrameItem& previous = mPyr[pyrIdxPrev];
	assert(frame.frameIndex > 0 && frame.frameIndex == previous.frameIndex + 1 && "wrong frames to compute");
	//Mat<double>::precision(16);

	for (int threadIdx = 0; threadIdx < mData.cpuThreads; threadIdx++) mPool.add([&, threadIdx] {
		int ir = mData.ir;
		int iw = mData.iw;
		Mat jm = Mat<double>::allocate(iw, iw);
		Mat delta = Mat<double>::allocate(iw, iw);
		Mat sd = Mat<double>::allocate(6, 1ull * iw * iw);
		Mat etaMat = Mat<double>::allocate(6, 1);
		Mat wp = Mat<double>::allocate(3, 3);
		Mat dwp = Mat<double>::allocate(3, 3);
		for (int iy0 = threadIdx; iy0 < mData.iyCount; iy0 += mData.cpuThreads) {
			for (int ix0 = 0; ix0 < mData.ixCount; ix0++) {
				//start with null transform
				wp.setValuesByRow({ 1, 0, 0, 0, 1, 0, 0, 0, 1 });
				dwp.setValuesByRow({ 1, 0, 0, 0, 1, 0, 0, 0, 1 });

				// center of previous integration window
				// one pixel padding around outside for delta
				// changes per z level
				int ym = iy0 + ir + 1;
				int xm = ix0 + ir + 1;
				PointResultType result = PointResultType::RUNNING;
				int z = mData.zMax;
				double err = 0.0;

				for (; z >= mData.zMin && result >= PointResultType::RUNNING; z--) {
					Matf& Y = previous.mY[z];
					SubMat<float> im = Y.subMatShared(ym - ir, xm - ir, iw, iw);

					//affine transform
					for (int r = 0; r < iw; r++) {
						for (int c = 0; c < iw; c++) {
							int iy = ym - ir + c;
							int ix = xm - ir + r;
							double dx = Y.at(iy, ix + 1) / 2 - Y.at(iy, ix - 1) / 2;
							double dy = Y.at(iy + 1, ix) / 2 - Y.at(iy - 1, ix) / 2;
							int idx = r * iw + c;
							sd.at(0, idx) = dx;
							sd.at(1, idx) = dy;
							sd.at(2, idx) = dx * (r - ir);
							sd.at(3, idx) = dy * (r - ir);
							sd.at(4, idx) = dx * (c - ir);
							sd.at(5, idx) = dy * (c - ir);
						}
					}
					//if (frameIndex == 1 && ix0 == 63 && iy0 == 1) sd.toConsole(); //----------------
					Mat s = sd.timesTransposed();
					//if (frameIndex == 1 && ix0 == 63 && iy0 == 1) s.toConsole(); //----------------

					Mat g = s.inv().value();
					double ns = s.norm1();
					double gs = g.norm1();
					double rcond = 1 / (ns * gs); //reciprocal condition number

					//if (frameIndex == 1 && ix0 == 97 && iy0 == 4) std::printf("cpu %d %.14f\n", z, rcond);
					//if (frameIndex == 1 && ix0 == 63 && iy0 == 1) g.toConsole(); //----------------

					result = (std::isnan(rcond) || rcond < mData.deps) ? PointResultType::FAIL_SINGULAR : PointResultType::RUNNING;
					int iter = 0;
					double bestErr = std::numeric_limits<double>::max();
					while (result == PointResultType::RUNNING) {
						//search for selected patch in current frame
						jm.setValues([&] (size_t r, size_t c) {
							double x = (double) (c) -mData.ir;
							double y = (double) (r) -mData.ir;
							double ix = xm + x * wp.at(0, 0) + y * wp.at(1, 0) + wp.at(0, 2);
							double iy = ym + x * wp.at(0, 1) + y * wp.at(1, 1) + wp.at(1, 2);
							return frame.mY[z].interp2(ix, iy).value_or(mData.dnan);
						}
						);

						delta.setValues([&] (size_t r, size_t c) {
							return im.at(r, c) - jm.at(r, c); }
						);

						//eta = g.times(sd.times(delta.flatToCol())) //[6 x 1]
						etaMat.setValuesByRow({ 0, 0, 1, 0, 0, 1 });
						for (int r = 0; r < 6; r++) {
							double b = 0.0;
							for (int idx = 0, cc = 0; cc < mData.iw; cc++) {
								for (int rr = 0; rr < mData.iw; rr++) {
									b += sd.at(r, idx) * delta.at(rr, cc);
									idx++;
								}
							}
							for (int i = 0; i < 6; i++) {
								etaMat[i][0] += g[i][r] * b;
							}
						}
						double* eta = etaMat.data(); //[6 x 1]
						dwp.setValuesByRow(0, 0, 2, 3, { eta[2], eta[3], eta[0], eta[4], eta[5], eta[1] }); //set first 6 values
						wp = wp.times(dwp);
						//if (frameIndex == 1 && ix0 == 27 && iy0 == 1) wp.toConsole("cpu"); //------------------------

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

					//transformation * 2
					wp[0][2] *= 2.0;
					wp[1][2] *= 2.0;

					//if (frameIndex == 1 && ix0 == 63 && iy0 == 1) etaMat.toConsole("cpu"); //------------------------
					//if (frameIndex == 1 && ix0 == 97 && iy0 == 4) std::printf("cpu %d %.14f\n", z, wp[1][2]);
				}
				//bring values to level 0
				double u = wp.at(0, 2);
				double v = wp.at(1, 2);
				int zp = z;

				while (z < 0) {
					xm /= 2; ym /= 2; u /= 2.0; v /= 2.0; z++;
				}
				while (z > 0) {
					xm *= 2; ym *= 2; u *= 2.0; v *= 2.0; z--;
				}

				//transformation for points with respect to center of image and level 0 of pyramid
				int idx = iy0 * mData.ixCount + ix0;
				mResultPoints[idx] = { idx, ix0, iy0, xm, ym, xm - mData.w / 2, ym - mData.h / 2, u, v, result, zp, err };
			}
		}
	});
	mPool.wait();
}

//fractions with 8bit precision used in cuda textures
//float frac(float f) {
//	return std::round(f * 256.0f) / 256.0f;
//}

void CpuFrame::outputData(const AffineTransform& trf, OutputContext outCtx) {
	size_t yuvidx = trf.frameIndex % mYUV.size();
	const ImageYuv& input = mYUV[yuvidx];
	assert(input.index == trf.frameIndex && "invalid frame index");
	for (size_t z = 0; z < 3; z++) {
		float f = 1.0f / 255.0f;
		mYuvPlane.setValues([&] (size_t r, size_t c) { return input.at(z, r, c) * f; }, mPool);
		//transform and evaluate pixels, write to out buffer
		auto func1 = [&] (size_t r, size_t c) {
			float bg = (mData.bgmode == BackgroundMode::COLOR ? mData.bgcol_yuv.colors[z] : mPrevOut[z].at(r, c));
			auto [x, y] = trf.transform(c, r); //pay attention to order of x and y
			return mYuvPlane.interp2(float(x), float(y)).value_or(bg);
		};
		Matf& buf = mBuffer[z];
		buf.setValues(func1, mPool);
		mPrevOut[z].setData(buf);

		//unsharp masking
		//Mat gauss = buf.filter2D(MainData::FILTER[z], &mPool);
		const FilterKernel& k = filterKernels[z];
		buf.filter1D(k.k, k.siz, Matf::Direction::HORIZONTAL, mFilterBuffer, mPool);
		mFilterBuffer.filter1D(k.k, k.siz, Matf::Direction::VERTICAL, mFilterResult, mPool);
		//gauss.saveAsCSV("f:/gauss_cpu.csv");

		//write output
		//define function with respect to row index
		ImageYuv* out = outCtx.outputFrame;
		unsigned char* yuvp = out->plane(z); //one plane for output
		auto func2 = [&] (size_t r) {
			unsigned char* yuvrow = yuvp + r * mData.cpupitch;
			for (size_t c = 0; c < mData.w; c++) {
				float val = buf.at(r, c) + (buf.at(r, c) - mFilterResult.at(r, c)) * mData.unsharp[z];
				val = std::clamp(val, 0.0f, 1.0f);
				yuvrow[c] = (unsigned char) std::round(val * 255);
			}
		};
		//forward to thread pool for iteration there
		mPool.addAndWait(func2, 0, mData.h);
	}
	if (outCtx.requestInput) {
		*outCtx.inputFrame = input;
	}
	outCtx.outputFrame->index = trf.frameIndex;

	//when encoding on gpu is selected
	if (outCtx.encodeCuda) {
		static std::vector<unsigned char> nv12(outCtx.cudaPitch * mData.h * 3 / 2);
		outCtx.outputFrame->toNV12(nv12, outCtx.cudaPitch);
		encodeNvData(nv12, outCtx.cudaNv12ptr);
	}
}

Matf CpuFrame::getTransformedOutput() const {
	return Matf::concatVert(mBuffer[0], mBuffer[1], mBuffer[2]);
}

Matf CpuFrame::getPyramid(size_t idx) const {
	assert(idx < mPyr.size() && "invalid pyramid index");
	Matf out = Matf::zeros(mData.pyramidRowCount, mData.w);
	size_t row = 0;
	for (const Matf& mat : mPyr[idx].mY) {
		out.setArea(row, 0, mat);
		row += mat.rows();
	}
	return out;
}

void CpuFrame::getInputFrame(int64_t frameIndex, ImagePPM& image) {
	size_t idxIn = (frameIndex) % mYUV.size();
	mYUV[idxIn].toPPM(image, mPool);
}

void CpuFrame::getTransformedOutput(int64_t frameIndex, ImagePPM& image) {
	ImageYuvMat(mData.h, mData.w, mBuffer[0], mBuffer[1], mBuffer[2]).toPPM(image, mPool);
}

ImageYuv CpuFrame::getInput(int64_t index) const {
	return mYUV[index % mYUV.size()];
}
