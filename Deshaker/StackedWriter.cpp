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

void StackedWriter::open(EncodingOption videoCodec) {
	FFmpegWriter::open(videoCodec, widthTotal, mData.h);

	ColorYuv bgcol = mData.bgcol_rgb.toYuv();
	for (int z = 0; z < 3; z++) {
		for (int i = 0; i < mData.h; i++) bg.push_back(bgcol.colors.at(z));
	}
}

OutputContext StackedWriter::getOutputContext() {
	return { true, false, &outputFrame, nullptr, 0, true, &inputFrame };
}

void StackedWriter::write() {
	int offset = int(mData.w * (1 + mData.blendInput.position) / 8);
	unsigned char* in = inputFrame.data() + offset;
	unsigned char* out = outputFrame.data() + offset;
	unsigned char* dest = combinedFrame.data();

	for (int row = 0; row < mData.h * 3; row++) {
		//source footage on left side
		std::copy(in, in + combinedFrame.w / 2, dest);
		//output frame on right side
		std::copy(out, out + combinedFrame.w / 2, dest + combinedFrame.w / 2);
		//middle 1% of width in background color
		for (int col = combinedFrame.w * 99 / 200; col < combinedFrame.w * 101 / 200; col++) {
			dest[col] = bg[row];
		}

		in += inputFrame.stride;
		out += outputFrame.stride;
		dest += combinedFrame.stride;
	}

	FFmpegWriter::write(combinedFrame);
	this->frameIndex++;
}