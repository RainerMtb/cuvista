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

const AffineTransform& MovieFrame::computeTransform(std::vector<PointResult> resultPoints) {
	return mFrameResult.computeTransform(resultPoints, mData, mPool, mData.rng.get());
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
//---------- DUMMY FRAME ----------------------------------------------
//---------------------------------------------------------------------

void DummyFrame::inputData(ImageYuv& frame) {
	size_t idx = mStatus.frameInputIndex % frames.size();
	frames[idx] = frame;
}

void DummyFrame::outputData(const AffineTransform& trf, OutputContext outCtx) {
	ImageYuv& frameToEncode = frames[mStatus.frameWriteIndex % frames.size()];

	if (outCtx.encodeCpu) {
		*outCtx.outputFrame = frameToEncode;
	}

	if (outCtx.encodeCuda) {
		encodeNvData(frameToEncode.toNV12(outCtx.cudaPitch), outCtx.cudaNv12ptr);
	}
}

ImageYuv DummyFrame::getInput(int64_t index) const {
	return frames[index % frames.size()];
}