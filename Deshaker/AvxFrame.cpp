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
#include <immintrin.h>

AvxFrame::AvxFrame(MainData& data, MovieReader& reader, MovieWriter& writer) :
	MovieFrame(data, reader, writer) {}

void AvxFrame::inputData() {
	//TODO
}

void AvxFrame::createPyramid(int64_t frameIndex) {
	//ConsoleTimer ic("pyramid");
	//TODO
}

void AvxFrame::computeStart(int64_t frameIndex) {}

void AvxFrame::computeTerminate(int64_t frameIndex) {
	//TODO
}

void AvxFrame::outputData(const AffineTransform& trf, OutputContext outCtx) {
	//TODO
}

Matf AvxFrame::getTransformedOutput() const {
	//TODO
	return Matf();
}

Matf AvxFrame::getPyramid(size_t idx) const {
	//TODO
	return Matf();
}

void AvxFrame::getInputFrame(int64_t frameIndex, ImagePPM& image) {
	//TODO
}

void AvxFrame::getTransformedOutput(int64_t frameIndex, ImagePPM& image) {
	//TODO
}

ImageYuv AvxFrame::getInput(int64_t index) const {
	//TODO
	return ImageYuv();
}
