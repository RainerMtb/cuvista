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
#include "MovieFrame.hpp"
#include "SubMat.hpp"
#include "MatrixInverter.hpp"
#include "Util.hpp"
#include "cuDeshaker.cuh"

//constructor
CpuFrame::CpuFrame(MainData& data, DeviceInfoBase& deviceInfo, MovieFrame& frame, ThreadPoolBase& pool) :
	FrameExecutor(data, deviceInfo, frame, pool) 
{
	assert(mDeviceInfo.type == DeviceType::CPU && "device type must be CPU here");

	//buffer to hold input frames in yuv format
	for (int i = 0; i < mData.bufferCount; i++) mYUV.emplace_back(mData.h, mData.w, mData.cpupitch);

	//init pyramid structures
	for (int i = 0; i < mData.pyramidCount; i++) mPyr.emplace_back(mData);

	//init storage for previous output frame to background colors
	mPrevOut.push_back(Matf::values(mData.h, mData.w, mData.bgcol_yuv.colors[0]));
	mPrevOut.push_back(Matf::values(mData.h, mData.w, mData.bgcol_yuv.colors[1]));
	mPrevOut.push_back(Matf::values(mData.h, mData.w, mData.bgcol_yuv.colors[2]));

	//buffer for output and pyramid creation
	mBuffer.assign(3, Matf::allocate(mData.h, mData.w));
	mFilterBuffer = Matf::allocate(mData.h, mData.w);
	mFilterResult = Matf::allocate(mData.h, mData.w);
	mYuvPlane = Matf::allocate(mData.h, mData.w);
	mOutput = ImageYuv(mData.h, mData.w, mData.cpupitch);
}

//construct data for one pyramid
CpuFrame::CpuPyramid::CpuPyramid(CoreData& data) {
	//allocate matrices for pyramid
	for (int z = 0; z <= data.zMax; z++) {
		int hz = data.h >> z;
		int wz = data.w >> z;
		mY.push_back(Matf::allocate(hz, wz));
	}
}

//read input frame and put into buffer
void CpuFrame::inputData(int64_t frameIndex, const ImageYuv& inputFrame) {
	size_t idx = frameIndex % mYUV.size();
	inputFrame.copyTo(mYUV[idx], mPool);
}

void CpuFrame::createPyramid(int64_t frameIndex) {
	//util::ConsoleTimer ic("cpu pyramid");
	size_t pyrIdx = frameIndex % mPyr.size();
	CpuPyramid& frame = mPyr[pyrIdx];
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
		Matf filterTemp = mFilterBuffer.share(y.rows(), y.cols());
		Matf mat = mFilterResult.share(y.rows(), y.cols());
		y.filter1D(filterKernels[0].k, filterKernels[0].siz, Matf::Direction::HORIZONTAL, filterTemp, mPool);
		filterTemp.filter1D(filterKernels[0].k, filterKernels[0].siz, Matf::Direction::VERTICAL, mat, mPool);

		//downsampling
		auto func = [&] (size_t r, size_t c) { return mat.interp2(c * 2, r * 2, 0.5f, 0.5f); };
		Matf& dest = frame.mY[z + 1];
		dest.setArea(func, mPool);
		//if (z == 0) std::printf("cpu %.14f\n", dest.at(100, 100));
		//if (z == 1) mat.saveAsBinary("f:/filterCpu.dat");
	}
	//if (status.frameInputIndex == 1) frame.Y[1].saveAsBinary("f:/cpu.dat");
}

void CpuFrame::computeStart(int64_t frameIndex, std::vector<PointResult>& results) {}

void CpuFrame::computeTerminate(int64_t frameIndex, std::vector<PointResult>& results) {
	size_t pyrIdx = frameIndex % mPyr.size();
	size_t pyrIdxPrev = (frameIndex - 1) % mPyr.size();
	CpuPyramid& frame = mPyr[pyrIdx];
	CpuPyramid& previous = mPyr[pyrIdxPrev];
	assert(frame.frameIndex > 0 && frame.frameIndex == previous.frameIndex + 1 && "wrong frames to compute");
	//Mat<double>::precision(16);

	for (int threadIdx = 0; threadIdx < mData.cpuThreads; threadIdx++) mPool.add([&, threadIdx] {
		int ir = mData.ir;
		int iw = mData.iw;
		Matd jm = Matd::allocate(iw, iw);
		Matd delta = Matd::allocate(iw, iw);
		Matd sd = Matd::allocate(6, 1ull * iw * iw);
		Matd etaMat = Matd::allocate(6, 1);
		Matd wp = Matd::allocate(3, 3);
		Matd dwp = Matd::allocate(3, 3);
		Matd g = Matd::allocate(6, 6);
		Matd I = Matd::eye(6);

		for (int iy0 = threadIdx; iy0 < mData.iyCount; iy0 += mData.cpuThreads) {
			for (int ix0 = 0; ix0 < mData.ixCount; ix0++) {
				//start with null transform
				wp.setDiag(1.0);
				dwp.setDiag(1.0);

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
					SubMat<float> im = SubMat<float>::from(Y, 0ll + ym - ir, 0ll + xm - ir, iw, iw);

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
					//if (frameIndex == 1 && ix0 == 63 && iy0 == 1 && z == mData.zMax) sd.toConsole(); //----------------

					Mat s = sd.timesTransposed();
					//s.saveAsBinary(std::format("f:/1/{}-{}-{}-{}.mat", frameIndex, z, iy0, ix0));
					//if (frameIndex == 1 && ix0 == 63 && iy0 == 1) s.toConsole(); //----------------

					double ns = s.norm1();
					LUDecompositor<double>(s).solve(I, g); //decomposing will overwrite content of s
					double gs = g.norm1();
					double rcond = 1 / (ns * gs); //reciprocal condition number
					result = (std::isnan(rcond) || rcond < mData.deps) ? PointResultType::FAIL_SINGULAR : PointResultType::RUNNING;
					//g = PseudoInverter(s, 6).inv();
					//result = g.has_value() ? PointResultType::RUNNING : PointResultType::FAIL_SINGULAR;

					int iter = 0;
					double bestErr = std::numeric_limits<double>::max();
					while (result == PointResultType::RUNNING) {
						//search for selected patch in current frame
						jm.setValues([&] (size_t r, size_t c) {
							double x = (double) (c) - mData.ir;
							double y = (double) (r) - mData.ir;
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
						//if (frameIndex == 1 && ix0 == 48 && iy0 == 0 && iter == 1) etaMat.toConsole("cpu", 18); //------------------------

						double* eta = etaMat.data(); //[6 x 1]
						dwp.setValuesByRow(0, 0, 2, 3, { eta[2], eta[3], eta[0], eta[4], eta[5], eta[1] }); //set first 6 values
						wp = wp.times(dwp);
						//if (frameIndex == 1 && ix0 == 48 && iy0 == 0 && iter == 1) wp.toConsole("cpu", 18); //------------------------

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
				results[idx] = { idx, ix0, iy0, xm, ym, xm - mData.w / 2, ym - mData.h / 2, u, v, result, zp };
			}
		}
	});
	mPool.wait();
}

void CpuFrame::outputData(int64_t frameIndex, const Affine2D& trf) {
	size_t yuvidx = frameIndex % mYUV.size();
	const ImageYuv& input = mYUV[yuvidx];
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
		auto func2 = [&] (size_t r) {
			for (size_t c = 0; c < mData.w; c++) {
				float val = buf.at(r, c) + (buf.at(r, c) - mFilterResult.at(r, c)) * mData.unsharp[z];
				val = std::clamp(val, 0.0f, 1.0f);
				mOutput.at(z, r, c) = (unsigned char) std::rint(val * 255);
				//if (r==755 && c==478) std::printf("cpu %.14f %d\n", val * 255, yuvrow[c]);
			}
		};
		//forward to thread pool for iteration there
		mPool.addAndWait(func2, 0, mData.h);
	}
	mOutput.index = frameIndex;
}

void CpuFrame::getOutputYuv(int64_t frameIndex, ImageYuvData& image) {
	assert(frameIndex == mOutput.index && "invalid frame index");
	mOutput.copyTo(image, mPool);
}

void CpuFrame::getOutputRgba(int64_t frameIndex, ImageRGBA& image) {
	assert(frameIndex == mOutput.index && "invalid frame index");
	mOutput.toRGBA(image, mPool);
}

void CpuFrame::getOutput(int64_t frameIndex, unsigned char* cudaNv12ptr, int cudaPitch) {
	assert(frameIndex == mOutput.index && "invalid frame index");
	static std::vector<unsigned char> nv12(cudaPitch * mData.h * 3 / 2);
	mOutput.toNV12(nv12, cudaPitch, mPool);
	encodeNvData(nv12, cudaNv12ptr);
}

void CpuFrame::getWarped(int64_t frameIndex, ImageRGBA& image) {
	ImageYuvMatFloat(mData.h, mData.w, mData.w, mBuffer[0].data(), mBuffer[1].data(), mBuffer[2].data()).toRGBA(image, mPool);
}

Matf CpuFrame::getTransformedOutput() const {
	return Matf::concatVert(mBuffer[0], mBuffer[1], mBuffer[2]);
}

Matf CpuFrame::getPyramid(int64_t index) const {
	Matf out = Matf::zeros(mData.pyramidRowCount, mData.w);
	size_t row = 0;
	for (const Matf& mat : mPyr[index].mY) {
		out.setArea(row, 0, mat);
		row += mat.rows();
	}
	return out;
}

void CpuFrame::getInput(int64_t frameIndex, ImageRGBA& image) const {
	size_t idx = frameIndex % mYUV.size();
	mYUV[idx].toRGBA(image, mPool);
}

void CpuFrame::getInput(int64_t index, ImageYuv& image) const {
	size_t idx = index % mYUV.size();
	mYUV[idx].copyTo(image);
}
