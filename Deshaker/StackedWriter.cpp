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

#include "MovieWriter.hpp"
#include "MovieFrame.hpp"

void StackedWriter::open(EncodingOption videoCodec) {
	FFmpegWriter::open(videoCodec, AV_PIX_FMT_YUV420P, mData.h, mWidthTotal, mWidthTotal, mData.fileOut);
	mInputFrameScaled.setColor(mData.backgroundColor);
}

void StackedWriter::prepareOutput(FrameExecutor& executor) {
	executor.getOutputYuv(frameIndex, mOutputFrame);
}

void StackedWriter::writeOutput(const FrameExecutor& executor) {
	executor.getInput(frameIndex, mInputFrame);

	int bufferIndex = 0;
	ImageYuv& combinedFrame = imageBuffer[bufferIndex];
	unsigned char* in = mInputFrameScaled.data();
	unsigned char* out = mOutputFrame.data() + mData.stackCrop.left;
	unsigned char* dest = combinedFrame.data();

	//scale mInputFrame by min zoom and write to mInputFrameScaled
	auto fcn = [&] (size_t r) {
		for (int c = 0; c < mInputFrameScaled.w; c++) {
			double x = (c - mData.w / 2.0) / mData.zoomMin + mData.w / 2.0 + mData.stackCrop.left;
			double y = (r - mData.h / 2.0) / mData.zoomMin + mData.h / 2.0;
			if (x >= 0.0 && x <= mData.w && y >= 0.0 && y <= mData.h) {
				mInputFrameScaled.at(0, r, c) = mInputFrame.sample(0, x, y);
				mInputFrameScaled.at(1, r, c) = mInputFrame.sample(1, x, y);
				mInputFrameScaled.at(2, r, c) = mInputFrame.sample(2, x, y);
			}
		}
	};
	executor.mPool.addAndWait(fcn, 0, mData.h);

	//combine images
	auto color = mData.backgroundColor.getYUV();
	for (int row = 0; row < mData.h * 3; row++) {
		//source footage on left side
		std::copy(in, in + mInputFrameScaled.w, dest);
		//output frame on right side
		std::copy(out, out + combinedFrame.w / 2, dest + combinedFrame.w / 2);
		//middle 1% of width in background color
		for (int col = combinedFrame.w * 99 / 200; col < combinedFrame.w * 101 / 200; col++) dest[col] = color[row / mData.h];

		in += mInputFrameScaled.stride;
		out += mOutputFrame.stride;
		dest += combinedFrame.stride;
	}

	combinedFrame.index = frameIndex;
	FFmpegWriter::write(bufferIndex);
}
