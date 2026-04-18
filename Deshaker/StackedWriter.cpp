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

void StackedWriter::open(OutputOption outputOption) {
	FFmpegWriter::open(outputOption, AV_PIX_FMT_YUV420P, mData.h, mWidthTotal, util::alignValue(mWidthTotal * 4, 64), mData.fileOut);
}

void StackedWriter::writeOutput(const FrameExecutor& executor) {
	executor.getOutput(frameIndex, mOutputFrame);
	executor.getInput(frameIndex, mInputFrame);

	int bufferIndex = 0;
	ImageVuyx& combinedFrame = imageBuffer[bufferIndex];
	combinedFrame.setColor(mData.backgroundColor);

	//scale mInputFrame by min zoom and write to left side of mOutputFrame
	auto fcn = [&] (size_t r) {
		for (size_t c = 0; c < mWidth; c++) {
			float x = (c - mData.w / 2.0f) / mData.zoomMin + mData.w / 2.0f + mData.stackCrop.left;
			float y = (r - mData.h / 2.0f) / mData.zoomMin + mData.h / 2.0f;
			if (x >= 0.0f && x <= mData.w - 1.0f && y >= 0.0f && y <= mData.h - 1.0f) {
				combinedFrame.at(0, r, c) = mInputFrame.sample(0, x, y);
				combinedFrame.at(1, r, c) = mInputFrame.sample(1, x, y);
				combinedFrame.at(2, r, c) = mInputFrame.sample(2, x, y);
			}
		}
	};
	executor.mPool.addAndWait(fcn, 0, mData.h);

	//combine images
	unsigned char* out = mOutputFrame.data() + mData.stackCrop.left * 4;
	unsigned char* dest = combinedFrame.data();
	int byteCount = combinedFrame.w() / 2 * 4;
	for (int r = 0; r < mData.h; r++) {
		//output frame on right side
		std::copy_n(out, byteCount, dest + byteCount);
		out += mOutputFrame.stride();
		dest += combinedFrame.stride();
	}
	combinedFrame.setColor(0, combinedFrame.w() * 99ull / 200, combinedFrame.h(), combinedFrame.w() * 1ull / 100, mData.backgroundColor);
	
	combinedFrame.index = frameIndex;
	FFmpegWriter::write(bufferIndex);
}
