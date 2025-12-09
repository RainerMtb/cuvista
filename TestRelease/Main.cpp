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
	std::cout << std::endl << "ARGS: ";
	for (const std::string& str : s) std::cout << str << " ";
	std::cout << std::endl;
}

static DeshakerResult run(std::vector<std::string> argsList, std::shared_ptr<MovieWriter> writer = {}, bool showOutput = true) {
	std::ostringstream oss;

	if (showOutput) {
		printArgs(argsList);
	}

	DeshakerResult result = deshake(argsList, &oss, writer);

	if (showOutput) {
		std::string str = oss.str();
		if (str.empty()) {
			std::cout << "no console output" << std::endl;

		} else {
			std::cout << "console output: " << std::endl << str << std::endl;
		}
	}
	return result;
}

static DeshakerResult run(const std::string& argsLine, std::shared_ptr<MovieWriter> writer = {}, bool showOutput = true) {
	return run(util::splitString(argsLine, " "), writer, showOutput);
}

static void testMain() {
	std::cout << "--- Short Runs ---" << std::endl;
	run("-frames 0 -i d:/VideoTest/example.mp4 -o d:/videoTest/out/000null.mp4 -noheader -progress 0 -y");
	run("-frames 0 -i d:/VideoTest/example.mp4 -o d:/videoTest/out/001null.mp4 -enc ffmpeg:hevc -noheader -progress 0 -y");
	run("-frames 1 -i d:/VideoTest/example.mp4 -o d:/videoTest/out/001.mp4 -noheader -progress 0 -y");
	run("-frames 2 -i d:/VideoTest/example.mp4 -o d:/videoTest/out/002.mp4 -noheader -progress 0 -y");
	run("-frames 3 -i d:/VideoTest/example.mp4 -o d:/videoTest/out/003.mp4 -noheader -progress 0 -y");
	run("-frames 40 -i d:/VideoTest/example.mp4 -o d:/videoTest/out/004.mp4 -noheader -progress 0 -y");
	run({ "-frames",  "40", "-i", "d:/VideoTest/example space.mp4", "-o", "d:/videoTest/out/000 space.mp4", "-noheader",  "-progress", "0", "-y" });

	run("-mode 1 -frames 0 -i d:/VideoTest/example.mp4 -o d:/videoTest/out/005null.mp4 -noheader -progress 0 -y");
	run("-mode 1 -frames 1 -i d:/VideoTest/example.mp4 -o d:/videoTest/out/006.mp4 -noheader -progress 0 -y");
	run("-mode 1 -frames 2 -i d:/VideoTest/example.mp4 -o d:/videoTest/out/007.mp4 -noheader -progress 0 -y");
	run("-mode 1 -frames 3 -i d:/VideoTest/example.mp4 -o d:/videoTest/out/008.mp4 -noheader -progress 0 -y");
	run("-mode 1 -frames 40 -i d:/VideoTest/example.mp4 -o d:/videoTest/out/009.mp4 -noheader -progress 0 -y");

	std::cout << "--- Images ---" << std::endl;
	run("-frames 10 -i d:/VideoTest/04.ts -o d:/VideoTest/out/images/test%02d.jpg -resim -progress 0 -y");
	run("-device 3 -frames 5 -i d:/VideoTest/01.mp4 -o d:/VideoTest/out/images/im%03d.bmp -progress 0 -y");
	run("-device 2 -frames 5 -i d:/VideoTest/01.mp4 -o d:/VideoTest/out/images/im%03d.jpg -progress 0 -y");

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
	run("-device 2 -i d:/VideoTest/06.mp4 -o d:/videoTest/out/06_stack.mp4 -stack 250:250 -noheader -progress 0");
	run("-device 2 -i d:/VideoTest/01.mp4 -o d:/videoTest/out/flow.mp4 -flow -noheader -progress 0");
	run("-device 1 -i d:/VideoTest/02short.mp4 -o d:/videoTest/out/raw02.yuv -noheader -progress 0");
	run("-device 1 -i d:/VideoTest/02short.mp4 -o d:/videoTest/out/raw02.nv12 -noheader -progress 0");
	run("-device 2 -i d:/VideoTest/02short.mp4 -o d:/videoTest/out/res.nv12 -resvid -noheader -progress 0");
	run("-device 3 -i d:/VideoTest/02short.mp4 -o d:/videoTest/out/bgmode.mkv -zoom -5 -bgmode color -bgcolor #F09B59 -noheader -progress 0");

	std::cout << "--- Encode Nvenc ---" << std::endl;
	run("-i d:/VideoTest/02short.mp4 -o d:/videoTest/out/nvenc00.mp4 -device 0 -enc nvenc:hevc -noheader -progress 0");
	run("-i d:/VideoTest/02short.mp4 -o d:/videoTest/out/nvenc01.mp4 -device 1 -enc nvenc:h264 -noheader -progress 0");
	run("-i d:/VideoTest/02short.mp4 -o d:/videoTest/out/nvenc02.mp4 -device 2 -enc nvenc:hevc -noheader -progress 0");

	std::cout << "--- Encoding to Cpu ---" << std::endl;
	run("-i d:/VideoTest/02short.mp4 -o d:/videoTest/out/enc_av1.mp4 -enc ffmpeg:av1 -progress 0");
	run("-i d:/VideoTest/02short.mp4 -o d:/videoTest/out/enc_hevc.mp4 -enc ffmpeg:hevc -progress 0");
	run("-enc ffmpeg:h264 -i d:/VideoTest/02short.mp4 -o d:/videoTest/out/enc_h264.mp4 -progress 0");
	run("-enc ffmpeg:ffv1 -i d:/VideoTest/02short.mp4 -o d:/videoTest/out/enc_ffv1.mp4 -progress 0");
}

static void testCrc() {
	std::string ansiGreen = "\x1b[1;32m";
	std::string ansiRed = "\x1b[1;31m";
	std::string ansiClear = "\x1b[0m";

	std::cout << std::endl << "--- Check Equal Files ---" << std::endl;
	std::vector<std::string> commands = {
		"-device 0 -i d:/VideoTest/02short.mp4 -o null -quiet",
		"-device 1 -i d:/VideoTest/02short.mp4 -o null -quiet",
		"-device 2 -i d:/VideoTest/02short.mp4 -o null -quiet",
		"-device 3 -i d:/VideoTest/02short.mp4 -o null -quiet"
	};

	for (size_t i = 0; i < commands.size(); i++) {
		std::shared_ptr<RawMemoryStoreWriter> externalWriter = std::make_shared<RawMemoryStoreWriter>(250, false, true);
		DeshakerResult result = run(commands[i], externalWriter);

		util::CRC64 crc;
		for (const TrajectoryItem& ti : result.trajectory) {
			crc.addDirect(ti.values.u);
			crc.addDirect(ti.values.v);
			crc.addDirect(ti.values.a);
			crc.addDirect(ti.smoothed.u);
			crc.addDirect(ti.smoothed.v);
			crc.addDirect(ti.smoothed.a);
			crc.addDirect(ti.sum.u);
			crc.addDirect(ti.sum.v);
			crc.addDirect(ti.sum.a);
			crc.addDirect(ti.isDuplicateFrame);
			crc.addDirect(ti.frameIndex);
			crc.addDirect(ti.zoom);
			crc.addDirect(ti.zoomRequired);
		}

		{
			uint64_t crcExpected = 0xcc66bbb8c0acc17c;
			bool match = crcExpected == crc;
			std::string color = match ? ansiGreen : ansiRed;
			std::cout << color << std::hex << "trajectory crc expected: " << crcExpected << ", actual crc: " << crc
				<< std::boolalpha << ", crc match: " << match << ansiClear << std::endl;
		}

		{
			//check yuv
			uint64_t crcExpectedYuv = 0xa0899de9a81fe6ae;
			util::CRC64 crcyuv;
			for (const ImageYuv& image : externalWriter->outputFramesYuv) crcyuv.add(image);

			bool match = crcExpectedYuv == crcyuv;
			std::cout << (match ? ansiGreen : ansiRed) << std::hex << "output yuv expected: " << crcExpectedYuv << ", actual crc: " << crcyuv
				<< std::boolalpha << ", crc match: " << match << ansiClear << std::endl;
		}

		{
			//check rgba
			uint64_t crcExpectedRgba = 0xf0cc957f24f5ae28;
			util::CRC64 crcrgba;
			for (const ImageRGBA& image : externalWriter->outputFramesRgba) crcrgba.add(image);

			bool match = crcExpectedRgba == crcrgba;
			std::cout << (match ? ansiGreen : ansiRed) << std::hex << "output rgba expected: " << crcExpectedRgba << ", actual crc: " << crcrgba
				<< std::boolalpha << ", crc match: " << match << ansiClear << std::endl;
		}

		{
			//check bgra
			uint64_t crcExpectedBgra = 0xf1cfe7cbe874441a;
			util::CRC64 crcbgra;
			for (const ImageBGRA& image : externalWriter->outputFramesBgra) crcbgra.add(image);

			bool match = crcExpectedBgra == crcbgra;
			std::cout << (match ? ansiGreen : ansiRed) << std::hex << "output bgra expected: " << crcExpectedBgra << ", actual crc: " << crcbgra
				<< std::boolalpha << ", crc match: " << match << ansiClear << std::endl;
		}
	}

	//------------------------------------------------------------------------------------
	std::cout << std::endl << "--- Check Mode 2 ---" << std::endl;
	std::vector<std::string> commandsMode2 = {
		"-device 0 -i d:/VideoTest/02short.mp4 -o null -quiet -mode 2",
		"-device 1 -i d:/VideoTest/02short.mp4 -o null -quiet -mode 2",
		"-device 2 -i d:/VideoTest/02short.mp4 -o null -quiet -mode 2",
		"-device 3 -i d:/VideoTest/02short.mp4 -o null -quiet -mode 2"
	};

	for (size_t i = 0; i < commandsMode2.size(); i++) {
		std::shared_ptr<RawMemoryStoreWriter> externalWriter = std::make_shared<RawMemoryStoreWriter>(250, false, true);
		DeshakerResult result = run(commandsMode2[i], externalWriter);

		uint64_t crcExpected = 0x62cc4863807117aa;
		util::CRC64 crcOutput;
		for (const ImageYuv& image : externalWriter->outputFramesYuv) crcOutput.add(image);

		bool match = crcExpected == crcOutput;
		std::cout << (match ? ansiGreen : ansiRed) << std::hex << "yuv file crc expected: " << crcExpected << ", actual crc: " << crcOutput
			<< std::boolalpha << ", crc match: " << match << ansiClear << std::dec << std::endl;
	}
}

static void testSpeed() {
	std::vector<std::string> commands = {
		"-device 0 -i d:/VideoTest/02.mp4 -o f:/videoOut.mp4 -y -frames 100",
		"-device 1 -i d:/VideoTest/02.mp4 -o f:/videoOut.mp4 -y -frames 200",
		"-device 2 -i d:/VideoTest/02.mp4 -o f:/videoOut.mp4 -y -frames 500",
		"-device 3 -i d:/VideoTest/02.mp4 -o f:/videoOut.mp4 -y -frames 500"
	};

	std::cout << std::endl << "SPEED TEST:" << std::endl;
	for (const std::string& str : commands) {
		DeshakerResult result = run(str, {}, false);
		std::cout << result.executorNameShort << ": " << result.framesWritten << " frames, " << result.secs << " sec" << std::endl;
	}
}

int main() {
	std::vector<std::string> folders = { "d:/videoTest/out", "d:/videoTest/out/images" };
	for (const std::string& folder : folders) {
		std::cout << "--- Delete " << folder << " ---" << std::endl;
		std::filesystem::path fp(folder);
		for (const auto& entry : std::filesystem::directory_iterator(fp)) {
			if (entry.is_regular_file()) std::filesystem::remove(entry);
		}
	}
	
	testSpeed();
	testCrc();
	testMain();
}

/*
return { (std::istreambuf_iterator<char>(infile)), (std::istreambuf_iterator<char>()) };
*/
