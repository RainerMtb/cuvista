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
#include "SystemStuff.hpp"

int main() {
	util::debugLoggerPtr = std::make_shared<util::DebugLoggerString>();
	//debugLogger().open("tcp://10.0.0.1:5555");

	std::vector<std::string> argsLines = {
		/*0*/ "-info",
		/*1*/ "-i d:/VideoTest/example.mp4 -o f:/videoOut.mp4 -bgmode color -y -enc ffmpeg:hevc -frames 4 -progress 0",
		/*2*/ "-device 1 -i d:/VideoTest/02short.mp4 -o null -mode 2",
		/*3*/ "-i d:/VideoTest/example.mp4 -o f:/videoOut.mp4 -bgmode color -y -zoom -8",
		/*4*/ "-i D:/VideoTest/x3/VID.mp4 -o f:/videoOut.mp4 -y -zoom 5 -radius 0.3",
		/*5*/ "-device 2 -i f:/pic/input.mp4 -o f:/videoOut.mp4 -y",
		/*6*/ "-device 3 -i D:/VideoTest/12.mp4 -o f:/videoOut.mp4 -y -frames 200",
		/*7*/ "-device 3 -i D:/VideoTest/02short.mp4 -o f:/videoOut.mp4 -y -stack 60:60",
		/*8*/ "-i d:/VideoTest/06b.mkv -o f:/videoOut.nv12 -resvid -bgmode color -y -zoom -8 -device 3 -noclassic",
		/*9*/ "-i d:/VideoTest/06b.mkv -o f:/videoOut.mp4 -y -device 2 -nodbscan",
		/*10*/ "-i d:/videoTest/15.ts -o f:/videoOut.mp4 -y -device 0 -frames 150 -log tcp://10.0.0.1:5555",
		/*11*/ "-i //READYNAS/Videos/Misc/AudioTestWettenDass.ts -o f:/videoOut.mp4 -y",
		/*12*/ "-i //READYNAS/Videos/Sugababes_push_the_button.mp4 -o f:/videoOut.mp4 -y",
	};

	int idx = 12;
	std::string argsLine = argsLines[idx];
	std::cout << "------- TestCuvista -------" << std::endl;
	std::cout << "------- params: " << argsLine << std::endl << std::endl;

	auto args = util::splitString(argsLine, " ");
	DeshakerResult result = deshake(args, &std::cout, {});

	std::cout << std::endl << "------- Log -------" << std::endl;
	std::cout << result.log << std::endl;
}
