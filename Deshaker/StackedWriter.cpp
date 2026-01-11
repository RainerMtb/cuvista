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
	FFmpegWriter::open(outputOption, AV_PIX_FMT_YUV420P, mData.h, mWidthTotal, mWidthTotal, mData.fileOut);
}

void StackedWriter::writeOutput(const FrameExecutor& executor) {
	executor.getOutputYuv(frameIndex, mOutputFrame);
	executor.getInput(frameIndex, mInputFrame);

	int bufferIndex = 0;
	ImageYuv& combinedFrame = imageBuffer[bufferIndex];
	combinedFrame.setColor(mData.backgroundColor);
	unsigned char* out = mOutputFrame.data() + mData.stackCrop.left;
	unsigned char* dest = combinedFrame.data();

	//scale mInputFrame by min zoom and write to left side of mOutputFrame
	auto fcn = [&] (size_t threadIdx) {
		for (size_t r = threadIdx; r < mData.h; r += mData.cpuThreads) {
			for (size_t c = 0; c < mWidth; c++) {
				float x = (c - mData.w / 2.0f) / mData.zoomMin + mData.w / 2.0f + mData.stackCrop.left;
				float y = (r - mData.h / 2.0f) / mData.zoomMin + mData.h / 2.0f;
				if (x >= 0.0f && x <= mData.w - 1.0f && y >= 0.0f && y <= mData.h - 1.0f) {
					combinedFrame.at(0, r, c) = mInputFrame.sample(0, x, y);
					combinedFrame.at(1, r, c) = mInputFrame.sample(1, x, y);
					combinedFrame.at(2, r, c) = mInputFrame.sample(2, x, y);
				}
			}
		}
	};
	executor.mPool.addAndWait(fcn, 0, mData.cpuThreads);

	//combine images
	auto color = mData.backgroundColor.getYUV();
	for (int row = 0; row < mData.h * 3; row++) {
		//output frame on right side
		std::copy_n(out, combinedFrame.w / 2, dest + combinedFrame.w / 2);
		//middle 1% of width in background color
		for (int col = combinedFrame.w * 99 / 200; col < combinedFrame.w * 101 / 200; col++) dest[col] = color[row / mData.h];

		out += mOutputFrame.stride;
		dest += combinedFrame.stride;
	}
	
	combinedFrame.index = frameIndex;
	FFmpegWriter::write(bufferIndex);
}
