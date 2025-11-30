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

#include <vector>
#include <string>

enum class OutputGroup {
	VIDEO_NVENC,
	VIDEO_FFMPEG,
	VIDEO_OTHER,
	IMAGE_SEQUENCE,
	PIPE,
	NONE,
	INVALID,
	AUTO,
};

struct OutputOption {

private:
	OutputOption(int id, std::string nameGroup, std::string nameDetail, int qualityMin, int qualityMax, int qualityPercent, OutputGroup group) :
		id { id },
		nameGroup { nameGroup },
		nameDetail { nameDetail },
		qualityMin { qualityMin },
		qualityMax { qualityMax },
		qualityPercent { qualityPercent },
		group { group }
	{}

	static std::vector<OutputOption> videoOptions();

	static std::vector<OutputOption> validOptions();

public:
	int id;
	std::string nameGroup;
	std::string nameDetail;
	int qualityMin;
	int qualityMax;
	int qualityPercent;
	OutputGroup group;

	static OutputOption FFMPEG_AV1;
	static OutputOption FFMPEG_HEVC;
	static OutputOption FFMPEG_H264;
	static OutputOption FFMPEG_FFV1;

	static OutputOption NVENC_AV1;
	static OutputOption NVENC_HEVC;
	static OutputOption NVENC_H264;

	static OutputOption OPTION_AUTO;
	static OutputOption OPTION_NONE;
	static OutputOption OPTION_INVALID;

	static OutputOption VIDEO_STACK;
	static OutputOption VIDEO_FLOW;
	static OutputOption PIPE_RAW;
	static OutputOption PIPE_ASF;
	static OutputOption IMAGE_BMP;
	static OutputOption IMAGE_JPG;
	static OutputOption IMAGE_RESULTS;
	static OutputOption RAW_YUV444;
	static OutputOption RAW_NV12;

	OutputOption() :
		OutputOption(-1, "", "", -1, -1, -1, OutputGroup::INVALID)
	{}

	//name of encoding option for user selection
	std::string displayName() const;

	//name of encoding option for cli input
	std::string fullName() const;

	//map quality value 0..100 to crf value for encoder
	int percentToCrf(double value) const;

	//default crf value
	int defaultCrf() const;

	//find option from name string
	static OutputOption find(std::string optionName);

	//check if option refers to a video output file
	bool isVideoFile() const;

	//check if option refers to an image sequence
	bool isImageSequence() const;

	//check if option has variable quality
	bool hasQuality() const;

	bool operator < (const OutputOption& other) const;

	bool operator == (const OutputOption& other) const;
};