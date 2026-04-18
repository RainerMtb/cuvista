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
#include "ImageClasses.hpp"

static void f1() {
	std::shared_ptr<RawMemoryStoreWriter> externalWriter = std::make_shared<RawMemoryStoreWriter>(250, true, true);
	util::debugLogger = std::make_shared<util::DebugLoggerTcp>("10.0.0.1", 5555);
	std::string argsLine = "-device 3 -i D:/VideoTest/12.mp4 -o null -frames 200";

	auto args = util::splitString(argsLine, " ");
	DeshakerResult result = deshake(args, &std::cout, externalWriter);
	std::cout << std::endl << "------- Log -------" << std::endl;
	std::cout << result.log << std::endl;
}

static void f2() {
	//util::debugLogger = std::make_shared<util::DebugLoggerString>();
	//util::debugLogger = std::make_shared<util::DebugLoggerTcp>("10.0.0.1", 5555);

	std::vector<std::string> argsLines = {
		/*0*/ "-info",
		/*1*/ "-i d:/VideoTest/example.mp4 -o f:/videoOut.mp4 -bgmode color -y -enc ffmpeg:hevc -frames 4 -progress 0",
		/*2*/ "-i d:/VideoTest/example.mp4 -o f:/videoOut.mp4 -bgmode color -y -zoom -8 -device 2 -enc nvenc:h264",
		/*3*/ "-i d:/VideoTest/example.mp4 -o f:/videoOut.mp4 -bgmode color -y -zoom -8",
		/*4*/ "-i d:/VideoTest/06b.mkv -o f:/videoOut.mp4 -bgmode color -y -zoom -8 -device 0 -log tcp://10.0.0.1:5555",
		/*5*/ "-device 3 -frames 5 -i d:/VideoTest/04.ts -o d:/VideoTest/out/images/im%03d.jpg -progress 0 -y",
		/*6*/ "-device 3 -i D:/VideoTest/12.mp4 -o f:/videoOut.mp4 -y -frames 200 -stack 200:200",
		/*7*/ "-device 3 -i D:/VideoTest/02short.mp4 -o f:/videoOut.mp4 -y -stack 60:60",
	};

	int idx = 4;
	std::string argsLine = argsLines[idx];
	std::cout << "------- CuvistaTest -------" << std::endl;
	std::cout << "------- params: " << argsLine << std::endl << std::endl;

	auto args = util::splitString(argsLine, " ");
	DeshakerResult result = deshake(args, &std::cout, {});

	std::cout << std::endl << "------- Log -------" << std::endl;
	std::cout << result.log << std::endl;
}

int main() {
	f2();
}