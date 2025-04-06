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
#include <sstream>
#include <regex>
#include <filesystem>

static void printArgs(std::span<std::string> s) {
	std::cout << "ARGS: ";
	for (const std::string& str : s) std::cout << str << " ";
	std::cout << std::endl;
}

static DeshakerResult run(const std::string& argsLine) {
	std::vector<std::string> argsList = util::splitString(argsLine, " ");
	std::ostringstream oss;

	std::cout << std::endl;
	printArgs(argsList);
	DeshakerResult result = deshake(argsList, &oss);
	std::string str = oss.str();
	if (str.empty()) {
		std::cout << "no output" << std::endl;

	} else {
		std::cout << "output: " << std::endl << str << std::endl;
	}
	return result;
}

int main() {
	std::string folder = "d:/videoTest/out";
	std::cout << "--- Delete " << folder << " ---" << std::endl;
	std::filesystem::path fp(folder);
	for (const auto& entry : std::filesystem::directory_iterator(fp)) {
		if (entry.is_regular_file()) std::filesystem::remove(entry);
	}

	std::cout << "--- Short Runs ---" << std::endl;
	run("-frames 0 -i d:/VideoTest/example.mp4 -o d:/videoTest/out/00.mp4 -noheader -progress 0 -y");
	run("-frames 1 -i d:/VideoTest/example.mp4 -o d:/videoTest/out/00.mp4 -noheader -progress 0 -y");
	run("-frames 2 -i d:/VideoTest/example.mp4 -o d:/videoTest/out/00.mp4 -noheader -progress 0 -y");
	run("-frames 3 -i d:/VideoTest/example.mp4 -o d:/videoTest/out/00.mp4 -noheader -progress 0 -y");
	run("-frames 40 -i d:/VideoTest/example.mp4 -o d:/videoTest/out/00.mp4 -noheader -progress 0 -y");

	std::cout << "--- Images ---" << std::endl;
	run("-frames 10 -i d:/VideoTest/04.ts -resim d:/VideoTest/out/images/test%02d.jpg -progress 0 -y");
	run("-frames 5 -i d:/VideoTest/01.mp4 -o d:/VideoTest/out/images/im03d.bmp -progress 0 -y");
	run("-device 2 -frames 5 -i d:/VideoTest/01.mp4 -o d:/VideoTest/out/images/im03d.jpg -progress 0 -y");

	std::cout << "--- Videos ---" << std::endl;
	run("-i d:/VideoTest/example.mp4 -o d:/videoTest/out/example_cuda.mp4 -noheader -progress 0");
	run("-i d:/VideoTest/example.mp4 -o d:/videoTest/out/example_cuda.mkv -noheader -progress 0");
	run("-i d:/VideoTest/01.mp4 -o d:/videoTest/out/01.mp4 -noheader -progress 0");
	run("-i d:/VideoTest/02.mp4 -o d:/videoTest/out/02.mp4 -noheader -progress 0");
	run("-i d:/VideoTest/03.mp4 -o d:/videoTest/out/03.mp4 -noheader -progress 0");
	run("-i d:/VideoTest/04.ts -o d:/videoTest/out/04.mkv -noheader -progress 0");
	run("-i d:/VideoTest/04.ts -o d:/videoTest/out/04.mp4 -noheader -progress 0");
	run("-i d:/VideoTest/05.avi -o d:/videoTest/out/05.mp4 -noheader -progress 0");
	run("-i d:/VideoTest/06.mp4 -o d:/videoTest/out/06.mp4 -noheader -progress 0");
	run("-i d:/VideoTest/06a.mp4 -o d:/videoTest/out/06a.mkv -noheader -progress 0");
	run("-i d:/VideoTest/06b.mkv -o d:/videoTest/out/06b.mp4 -noheader -progress 0");
	run("-i d:/VideoTest/07.mp4 -o d:/videoTest/out/07.mp4 -noheader -progress 0");

	std::cout << "--- Device Cpu ---" << std::endl;
	run("-device cpu -i d:/VideoTest/example.mp4 -o d:/videoTest/out/example_cpu.mp4 -noheader -progress 0");
	run("-device cpu -i d:/VideoTest/example.mp4 -o d:/videoTest/out/example_cpu.mkv -noheader -progress 0");

	std::cout << "--- Device Avx ---" << std::endl;
	run("-device 1 -i d:/VideoTest/01.mp4 -o d:/videoTest/out/01.avx.mp4 -noheader -progress 0");
	run("-device 1 -i d:/VideoTest/06.mp4 -o d:/videoTest/out/06.avx.mp4 -noheader -progress 0");

	std::cout << "--- Device Ocl ---" << std::endl;
	run("-device 2 -i d:/VideoTest/01.mp4 -o d:/videoTest/out/01.ocl.mp4 -noheader -progress 0");
	run("-device 2 -i d:/VideoTest/02.mp4 -o d:/videoTest/out/02.ocl.mp4 -noheader -progress 0");
	run("-device 2 -i d:/VideoTest/03.mp4 -o d:/videoTest/out/03.ocl.mp4 -noheader -progress 0");
	run("-device 2 -i d:/VideoTest/04.ts -o d:/videoTest/out/04.ocl.mkv -noheader -progress 0");
	run("-device 2 -i d:/VideoTest/05.avi -o d:/videoTest/out/05.ocl.mp4 -noheader -progress 0");
	run("-device 2 -i d:/VideoTest/06.mp4 -o d:/videoTest/out/06.ocl.mp4 -noheader -progress 0");
	run("-device 2 -i d:/VideoTest/07.mp4 -o d:/videoTest/out/07.ocl.mkv -noheader -progress 0");

	std::cout << "--- Misc ---" << std::endl;
	run("-device 2 -stack 0 -i d:/VideoTest/06.mp4 -o d:/videoTest/out/06_stack.mp4 -noheader -progress 0");
	run("-device 2 -i d:/VideoTest/01.mp4 -o null -flow d:/videoTest/out/flow.mp4 -noheader -progress 0");

	std::cout << "--- Pass 12 ---" << std::endl;
	run("-pass 12 -device 3 -i d:/VideoTest/01.mp4 -o d:/videoTest/out/01.pass12.ocl.mp4 -noheader -progress 0");
	run("-pass 12 -device 2 -i d:/VideoTest/02.mp4 -o d:/videoTest/out/02.pass12.ocl.mp4 -noheader -progress 0");

	return 0;
}