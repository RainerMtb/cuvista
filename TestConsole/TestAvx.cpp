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
#include "AvxUtil.hpp"

void avxCompute() {
	for (size_t s = 1; s <= 8; s++) {
		for (int i = 0; i < 4; i++) {
			Matd a = Matd::rand(s, s, -20, 50, 1000);
			Matd ainvCPU = a.inv().value();

			//test inverse
			std::vector<VD8> vector(s);
			__mmask8 mask = (1 << s) - 1;
			for (int i = 0; i < s; i++) vector[i] = VD8(a.addr(i, 0), mask);
			Avx::inv(vector);
			Matd ainvAvx = Matd::zeros(s, s);
			for (int i = 0; i < s; i++) vector[i].storeu(ainvAvx.addr(i, 0), mask);

			Matd delta = ainvCPU.minus(ainvAvx);
			double deltaMax = delta.abs().max();
			if (deltaMax == 0.0) {
				std::cout << "avx invert test, dim=" << s << " EXACT" << std::endl;

			} else if (deltaMax < 1e-12) {
				std::cout << "avx invert test, dim=" << s << " delta " << deltaMax << std::endl;

			} else {
				std::cout << "avx invert test, dim=" << s << " FAIL " << deltaMax << std::endl;
				//a.toConsole("A");
				//ainvCPU.toConsole("cpu");
				//ainvAvx.toConsole("avx");
			}

			//test norm1
			double anCPU = a.norm1();
			for (int i = 0; i < s; i++) vector[i] = VD8(a.addr(i, 0), mask);
			double anAvx = Avx::norm1(vector);
			double d = anCPU - anAvx;

			if (d == 0.0) {
				std::cout << "avx norm1 test, dim=" << s << " EXACT" << std::endl;

			} else {
				std::cout << "avx norm1 test, dim=" << s << " FAIL " << d << std::endl;
			}
		}
	}
}
