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
	FFmpegWriter::open(videoCodec, mData.h, mWidthTotal, mWidthTotal, mData.fileOut);

	im::ColorYuv bgcol = mData.bgcol_rgb.toYuv();
	for (int z = 0; z < 3; z++) {
		for (int i = 0; i < mData.h; i++) mBackground.push_back(bgcol.colors.at(z));
	}
}

void StackedWriter::prepareOutput(MovieFrame& frame) {
	frame.getOutput(frame.mWriter.frameIndex, mOutputFrame);
}

void StackedWriter::write(const MovieFrame& frame) {
	frame.getInput(frameIndex, mInputFrame);

	int bufferIndex = 0;
	ImageYuv& combinedFrame = imageBuffer[bufferIndex];
	int offset = int(mData.w * (1 + mStackPosition) / 8);
	unsigned char* in = mInputFrame.data() + offset;
	unsigned char* out = mOutputFrame.data() + offset;
	unsigned char* dest = combinedFrame.data();

	for (int row = 0; row < mData.h * 3; row++) {
		//source footage on left side
		std::copy(in, in + combinedFrame.w / 2, dest);
		//output frame on right side
		std::copy(out, out + combinedFrame.w / 2, dest + combinedFrame.w / 2);
		//middle 1% of width in background color
		for (int col = combinedFrame.w * 99 / 200; col < combinedFrame.w * 101 / 200; col++) dest[col] = mBackground[row];

		in += mInputFrame.stride;
		out += mOutputFrame.stride;
		dest += combinedFrame.stride;
	}

	combinedFrame.index = frameIndex;
	FFmpegWriter::write(bufferIndex);
}
