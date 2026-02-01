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

#pragma once

#include <numbers>

#include "Mat.hpp"
#include "MovieFrame.hpp"
#include "MovieReader.hpp"
#include "MovieWriter.hpp"
#include "CpuFrame.hpp"
#include "CudaFrame.hpp"
#include "clMain.hpp"
#include "AvxFrame.hpp"
#include "FrameResult.hpp"
#include "MatrixInverter.hpp"
#include "cuTest.cuh"


void matTest();
void qrdec();
void subMat();
void matPerf();

void iteratorTest();
void similarTransform();
void readAndWriteOneFrame();
void checkVersions();
void draw(const std::string& filename);
void filterCompare();

void cudaInvSimple();
void cudaInvPerformanceTest();
void cudaInvEqualityTest();
void cudaInvParallel();
void cudaInvTest(size_t s1, size_t s2);
void cudaTextureRead();

void openClInvTest(size_t s1, size_t s2);
void openClInvGroupTest(int w1, int w2);
void openClnorm1Test();

void compareFramesPlatforms();
void testVideo1();
void testVideo2();

void flow();
void pinvTest();
void compareInv();

void avxCompute();
void avxTest();

void testZoom();
void testSampler();
void createTransformImages();
