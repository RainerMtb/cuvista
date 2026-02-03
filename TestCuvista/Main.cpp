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

#include "Deshaker.hpp"

int main() {
	std::cout << "------- CuvistaTest -------" << std::endl << std::endl;

	std::vector<std::string> argsLines = {
		"-i d:/VideoTest/02short.mp4 -o f:/videoOut.mp4 -y",
		"-i d:/VideoTest/example.mp4 -o f:/videoOut.mp4 -y",
		"-i d:/Documents/x.orig/beach.1.avi -o null -frames 100 -device 2",
		"-i d:/Documents/x.orig/beach.1.avi -o f:/videoOut.mp4 -y -frames 100 -device 2"
	};
	int idx = 1;

	std::string argsLine = argsLines[idx];
	auto args = util::splitString(argsLine, " ");
	deshake(args, &std::cout, {});
}
