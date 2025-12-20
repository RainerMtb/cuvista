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

#include "TestMain.hpp"

struct fp12 {
	int32_t value;

	fp12(int i) : value { 0 } {}
	fp12(float f) : value { 0 } {}
	fp12() : fp12(0) {}

	operator double() const { return 0.0; }
	operator float() const { return 0.0f; }
	operator int() const { return 0; }

	fp12 operator * (const fp12& other) { return 0; }
	friend fp12 operator * (fp12 a, const fp12& other) { return 0; }

	fp12& operator += (const fp12& other) { return *this; }
};

int main() {
	std::cout << "----------------------------" << std::endl << "TestMain:" << std::endl;
	//qrdec();
	//draw("f:/drawing.bmp"); 
	//filterCompare();
	//matPerf();
	//matTest();
	//subMat();
	//iteratorTest();
	//cudaInvSimple();
	//cudaInvPerformanceTest();
	//cudaInvEqualityTest();
	//cudaInvParallel();
	//cudaInvTest(1, 32);
	//cudaTextureRead();
	//readAndWriteOneFrame();
	//checkVersions();
	//transform();

	//openClInvTest(1, 32);
	//openClInvGroupTest(1, 9);
	//openClnorm1Test();
	//flow();
	//pinvTest();
	//compareInv();
	//similarTransform();

	//testSampler();
	//compareFramesPlatforms();
	avxCompute();
	//avxTest();

	//testZoom();
	//testVideo1();

	//createTransformImages();
}