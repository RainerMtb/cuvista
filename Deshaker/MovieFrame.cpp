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

#include "MovieFrame.hpp"

MovieFrame::~MovieFrame() {
	mPool.shutdown(); //shutdown threads
}

void MovieFrame::runDiagnostics(int64_t frameIndex) {
	for (auto& item : diagsList) {
		item->run(mFrameResult, frameIndex);
	}
}

std::map<int64_t, TransformValues> MovieFrame::readTransforms() {
	TransformsFile tf(mData.trajectoryFile, std::ios::in | std::ios::binary);
	return tf.readTransformMap();
}

//---------------------------------------------------------------------
//---------- GPU FRAME ------------------------------------------------
//---------------------------------------------------------------------

//destruct cuda stuff and get debug data if present
GpuFrame::~GpuFrame() {
	DebugData data = cudaShutdown(mData);

	//retrieve debug data from device if present
	std::vector<double>& debugData = data.debugData;
	size_t siz = (size_t) debugData[0];
	double* ptr = debugData.data() + 1;
	double* ptrEnd = debugData.data() + siz + 1;
	while (ptr != ptrEnd) {
		size_t h = (size_t) *ptr++;
		size_t w = (size_t) *ptr++;
		std::cout << std::endl << "Debug Data found, mat [" << h << " x " << w << "]" << std::endl;
		//Mat<double>::fromArray(h, w, ptr).saveAsCSV("f:/gpu.txt", true);
		Mat<double>::fromArray(h, w, ptr, false).toConsole("", 16);
		ptr += h * w;
	}

	//data.kernelTimings.saveAsBMP("f:/kernel.bmp");
}


//---------------------------------------------------------------------
//---------- CPU FRAME ------------------------------------------------
//---------------------------------------------------------------------


//constructor
CpuFrame::CpuFrame(MainData& data) : MovieFrame(data) {
	//buffer to hold input frames in yuv format
	YUV.assign(data.bufferCount, ImageYuv(data.h, data.w, data.w));
	
	//init cpuFrameItem structures, allocate pyramid
	for (int i = 0; i < data.pyramidCount; i++) pyr.emplace_back(data);

	//init storage for previous output frame to background colors
	prevOut.push_back(Mat<float>::values(data.h, data.w, data.bgcol_yuv.colors[0]));
	prevOut.push_back(Mat<float>::values(data.h, data.w, data.bgcol_yuv.colors[1]));
	prevOut.push_back(Mat<float>::values(data.h, data.w, data.bgcol_yuv.colors[2]));

	//buffer for output and pyramid creation
	buffer.assign(4, Mat<float>::allocate(data.h, data.w));
	filterBuffer = Mat<float>::allocate(data.h, data.w);
	filterResult = Mat<float>::allocate(data.h, data.w);
	yuv = Mat<float>::allocate(data.h, data.w);
}

//construct data for one pyramid
CpuFrame::CpuFrameItem::CpuFrameItem(MainData& data) {
	//allocate matrices for pyramid
	for (int z = 0; z <= data.zMax; z++) {
		int hz = data.h >> z;
		int wz = data.w >> z;
		Y.push_back(Mat<float>::allocate(hz, wz));
		DX.push_back(Mat<float>::allocate(hz, wz));
		DY.push_back(Mat<float>::allocate(hz, wz));
	}
}

//read input frame and put into buffer
void CpuFrame::inputData(ImageYuv& frame) {
	size_t idx = mStatus.frameInputIndex % YUV.size();
	YUV[idx] = frame;
}

void CpuFrame::createPyramid() {
	//ConsoleTimer ic("pyramid");
	size_t pyrIdx = mStatus.frameInputIndex % pyr.size();
	CpuFrameItem& frame = pyr[pyrIdx];
	frame.frameIndex = mStatus.frameInputIndex;

	//fill topmost level of pyramid
	size_t yuvIdx = mStatus.frameInputIndex % YUV.size();
	ImageYuv& yuv = YUV[yuvIdx];
	float f = 1.0f / 255.0f;
	frame.Y[0].setValues([&] (size_t r, size_t c) { return yuv.at(0, r, c) * f; }, mPool);

	//create pyramid levels below by downsampling level above
	auto& k = mData.kernelFilter[0];
	for (size_t z = 0; z < mData.zMax; z++) {
		Mat<float>& y = frame.Y[z];
		//gauss filtering
		Mat<float> filterTemp = filterBuffer.reuse(y.rows(), y.cols());
		Mat<float> mat = filterResult.reuse(y.rows(), y.cols());
		y.filter1D(k.data(), k.size(), filterTemp, Direction::HORIZONTAL, mPool);
		filterTemp.filter1D(k.data(), k.size(), mat, Direction::VERTICAL, mPool);
		//if (z == 0) mat.saveAsBinary("f:/buf_c.dat");
		//downsampling
		auto func = [&] (size_t r, size_t c) { return mat.interp2(c * 2, r * 2, 0.5f, 0.5f); };
		frame.Y[z + 1].setArea(func, mPool);
	}
	//if (status.frameInputIndex == 1) frame.Y[1].saveAsBinary("f:/cpu.dat");
	
	//create delta pyramids
	for (size_t z = 0; z <= mData.zMax; z++) {
		frame.Y[z].filter1D(mData.filterKernel.data(), mData.filterKernel.size(), frame.DX[z], Direction::HORIZONTAL, mPool);
		frame.Y[z].filter1D(mData.filterKernel.data(), mData.filterKernel.size(), frame.DY[z], Direction::VERTICAL, mPool);
	}
}

void CpuFrame::computeTerminate() {
	size_t pyrIdx = mStatus.frameInputIndex % pyr.size();
	size_t pyrIdxPrev = (pyrIdx == 0 ? pyr.size() : pyrIdx) - 1;
	CpuFrameItem& frame = pyr[pyrIdx];
	CpuFrameItem& previous = pyr[pyrIdxPrev];
	assert(frame.frameIndex > 0 && frame.frameIndex == previous.frameIndex + 1 && "wrong frames to compute");
	Mat<double>::precision(16);

	for (size_t threadIdx = 0; threadIdx < mData.cpuThreads; threadIdx++) mPool.add([&, threadIdx] {
		int ir = mData.ir;
		int iw = mData.iw;
		Mat jm = Mat<double>::allocate(iw, iw);
		Mat delta = Mat<double>::allocate(iw, iw);
		Mat sd = Mat<double>::allocate(6, 1ull * iw * iw);
		Mat etaMat = Mat<double>::allocate(6, 1);
		Mat wp = Mat<double>::allocate(3, 3);
		Mat dwp = Mat<double>::allocate(3, 3);
		for (size_t iy0 = threadIdx; iy0 < mData.iyCount; iy0 += mData.cpuThreads) {
			for (size_t ix0 = 0; ix0 < mData.ixCount; ix0++) {
				//start with null transform
				wp.setValuesByRow({ 1, 0, 0, 0, 1, 0, 0, 0, 1 });
				dwp.setValuesByRow({ 1, 0, 0, 0, 1, 0, 0, 0, 1 });

				//center of previous integration window
				int ym = (int) iy0 + ir;
				int xm = (int) ix0 + ir;
				PointResultType result = PointResultType::RUNNING;
				int z = mData.zMax;
				for (; z >= mData.zMin && result >= PointResultType::RUNNING; z--) {
					//based on previous frame
					SubMat<float> dx = previous.DX[z].subMatShared(ym - ir, xm - ir, iw, iw);
					SubMat<float> dy = previous.DY[z].subMatShared(ym - ir, xm - ir, iw, iw);
					SubMat<float> im = previous.Y[z].subMatShared(ym - ir, xm - ir, iw, iw);

					//affine transform
					for (size_t r = 0; r < iw; r++) {
						for (size_t c = 0; c < iw; c++) {
							double x = dx.at(c, r);
							double y = dy.at(c, r);
							double rd = (double) (r) - mData.ir;
							double cd = (double) (c) - mData.ir;
							size_t idx = r * iw + c;
							sd.at(0, idx) = x;
							sd.at(1, idx) = y;
							sd.at(2, idx) = x * rd;
							sd.at(3, idx) = y * rd;
							sd.at(4, idx) = x * cd;
							sd.at(5, idx) = y * cd;
						}
					}
					//if (mData.status.frameInputIndex == 1 && ix0 == 63 && iy0 == 1) sd.toConsole(); //----------------
					Mat s = sd.timesTransposed();
					//if (mData.status.frameInputIndex == 1 && ix0 == 63 && iy0 == 1) s.toConsole(); //----------------

					Mat g = s.inv().value();
					double rcond = 1 / (s.norm1() * g.norm1()); //reciprocal condition number

					//if (mData.status.frameInputIndex == 1 && ix0 == 63 && iy0 == 1) g.toConsole(); //----------------

					result = (std::isnan(rcond) || rcond < mData.deps) ? PointResultType::FAIL_SINGULAR : PointResultType::RUNNING;
					int iter = 0;
					double bestErr = std::numeric_limits<double>::max();
					while (result == PointResultType::RUNNING) {
						//search for selected patch in current frame
						jm.setValues([&] (size_t r, size_t c) {
							double x = (double) (c) - mData.ir;
							double y = (double) (r) - mData.ir;
							double ix = xm + x * wp.at(0, 0) + y * wp.at(1, 0) + wp.at(0, 2);
							double iy = ym + x * wp.at(0, 1) + y * wp.at(1, 1) + wp.at(1, 2);
							return frame.Y[z].interp2(ix, iy).value_or(mData.dnan);
							}
						);

						//TODO jm adjust for brightness and contrast
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
						//if (mData.status.frameInputIndex == 1 && ix0 == 27 && iy0 == 1) wp.toConsole("cpu"); //------------------------

						double err = eta[0] * eta[0] + eta[1] * eta[1];
						if (std::isnan(err)) result = PointResultType::FAIL_ETA_NAN;
						if (err < mData.compMaxTol) result = PointResultType::SUCCESS_ABSOLUTE_ERR;
						if (std::abs(err - bestErr) / bestErr < mData.compMaxTol * mData.compMaxTol) result = PointResultType::SUCCESS_STABLE_ITER;
						if (err < bestErr) bestErr = err;
						iter++;
						if (iter == mData.compMaxIter && result == PointResultType::RUNNING) result = PointResultType::FAIL_ITERATIONS;
					}
					//if (mData.status.frameInputIndex == 1 && ix0 == 63 && iy0 == 1) etaMat.toConsole("cpu"); //------------------------

					//center of integration window on next level
					ym *= 2;
					xm *= 2;
					//transformation * 2
					wp[0][2] *= 2.0;
					wp[1][2] *= 2.0;

					//if (mData.status.frameInputIndex == 1 && ix0 == 10 && iy0 == 1) wp.toConsole("cpu");
				}
				//bring values to level 0
				double u = wp.at(0, 2);
				double v = wp.at(1, 2);

				while (z < 0) { 
					xm /= 2; ym /= 2; u /= 2.0; v /= 2.0; z++; 
				}
				while (z > 0) { 
					xm *= 2; ym *= 2; u *= 2.0; v *= 2.0; z--; 
				}

				//transformation for points with respect to center of image and level 0 of pyramid
				size_t idx = iy0 * mData.ixCount + ix0;
				resultPoints[idx] = { idx, ix0, iy0, xm, ym, xm - mData.w / 2, ym - mData.h / 2, u, v, result };
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
	size_t yuvidx = mStatus.frameWriteIndex % YUV.size();
	const ImageYuv& input = YUV[yuvidx];
	for (size_t z = 0; z < 3; z++) {
		float f = 1.0f / 255.0f;
		yuv.setValues([&] (size_t r, size_t c) { return input.at(z, r, c) * f; }, mPool);
		//transform and evaluate pixels, write to out buffer
		auto func1 = [&] (size_t r, size_t c) {
			float bg = (mData.bgmode == BackgroundMode::COLOR ? mData.bgcol_yuv.colors[z] : prevOut[z].at(r, c));
			auto [x, y] = trf.transform(c, r); //pay attention to order of x and y
			return yuv.interp2(float(x), float(y)).value_or(bg);
		};
		Mat<float>& buf = buffer[z];
		buf.setValues(func1, mPool);
		prevOut[z].setData(buf);

		//unsharp masking
		//Mat gauss = buf.filter2D(MainData::FILTER[z], &mPool);
		auto& k = mData.kernelFilter[z];
		buf.filter1D(k.data(), k.size(), filterBuffer, Direction::HORIZONTAL, mPool);
		filterBuffer.filter1D(k.data(), k.size(), filterResult, Direction::VERTICAL, mPool);
		//gauss.saveAsCSV("f:/gauss_cpu.csv");

		//write output
		//define function with respect to row index
		ImageYuv* out = outCtx.outputFrame;
		unsigned char* yuvp = out->plane(z); //one plane for output
		auto func2 = [&] (size_t r) {
			unsigned char* yuvrow = yuvp + r * mData.pitch;
			for (size_t c = 0; c < mData.w; c++) {
				float val = buf.at(r, c) + (buf.at(r, c) - filterResult.at(r, c)) * mData.unsharp[z];
				val = std::clamp(val, 0.0f, 1.0f);
				yuvrow[c] = (unsigned char) std::round(val * 255);
			}
		};
		//forward to thread pool for iteration there
		mPool.add(func2, 0, mData.h);

		//blend input if requested
		if (mData.blendInput.blendWidth > 0) {
			const BlendInput& bi = mData.blendInput;
			const unsigned char* src = input.plane(z) + bi.blendStart;
			unsigned char* dest = out->plane(z) + bi.blendStart;
			unsigned char* sep = out->plane(z) + bi.separatorStart;
			for (size_t r = 0; r < mData.h; r++) {
				std::copy(src, src + bi.blendWidth, dest);
				src += input.stride;
				dest += out->stride;

				std::fill(sep, sep + bi.separatorWidth, (char) (mData.bgcol_yuv.colors[z] * 255));
				sep += out->stride;
			}
		}
	}
	outCtx.outputFrame->frameIdx = mStatus.frameWriteIndex;

	//when encoding on gpu is selected
	if (outCtx.encodeGpu) {
		static std::vector<unsigned char> nv12(outCtx.cudaPitch * mData.h * 3 / 2);
		outCtx.outputFrame->toNV12(nv12, outCtx.cudaPitch);
		encodeNvData(nv12, outCtx.cudaNv12ptr);
	}
}

Mat<float> CpuFrame::getTransformedOutput() const {
	return Mat<float>::concatVert(buffer[0], buffer[1], buffer[2]);
}

Mat<float> CpuFrame::getPyramid(size_t idx) const {
	assert(idx < pyr.size() && "pyramid index not available");
	Mat<float> out = Mat<float>::zeros(mData.pyramidRows * 3LL, mData.w);
	size_t row = 0;
	const auto& items = { pyr[idx].Y, pyr[idx].DX, pyr[idx].DY };
	for (const auto& item : items) {
		for (int i = 0; i <= mData.zMax; i++) {
			const Mat<float>& ymat = item[i];
			out.setArea(row, 0, ymat);
			row += ymat.rows();
		}
	}
	return out;
}

bool CpuFrame::getCurrentInputFrame(ImagePPM& image) {
	bool state = mStatus.frameReadIndex > 0;
	if (state) {
		size_t idxIn = (mStatus.frameReadIndex - 1) % YUV.size();
		YUV[idxIn].toPPM(image, mPool);
	}
	return state;
}

bool CpuFrame::getCurrentOutputFrame(ImagePPM& image) {
	bool state = mStatus.frameWriteIndex > 0;
	if (state) {
		ImageYuvMat(mData.h, mData.w, buffer[0], buffer[1], buffer[2]).toPPM(image, mPool);
	}
	return state;
}

ImageYuv CpuFrame::getInput(int64_t index) const {
	return YUV[index % YUV.size()];
}


//---------------------------------------------------------------------
//---------- DUMMY FRAME ----------------------------------------------
//---------------------------------------------------------------------

void DummyFrame::inputData(ImageYuv& frame) {
	size_t idx = mStatus.frameInputIndex % frames.size();
	frames[idx] = frame;
}

void DummyFrame::outputData(const AffineTransform& trf, OutputContext od) {
	ImageYuv& frameToEncode = frames[mStatus.frameWriteIndex % frames.size()];

	ImageYuv* ptr1 = od.outputFrame;
	if (ptr1) {
		*ptr1 = frameToEncode;
	}

	unsigned char* ptr2 = od.cudaNv12ptr;
	if (ptr2) {
		frameToEncode.toNV12(nv12, od.cudaPitch);
		encodeNvData(nv12, ptr2);
	}
}

ImageYuv DummyFrame::getInput(int64_t index) const {
	return frames[index % frames.size()];
}