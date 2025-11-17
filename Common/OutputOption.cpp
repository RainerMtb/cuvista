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

#include "OutputOption.hpp"

#include <cmath>

RuntimeBase RuntimeBase::EMPTY = {};

std::string RuntimeBase::displayName() const {
	return "";
}

OutputOption OutputOption::FFMPEG_AV1 =  { 0, "FFMPEG",  "AV1", 63, 1, 55, OutputGroup::VIDEO_FFMPEG };
OutputOption OutputOption::FFMPEG_HEVC = { 1, "FFMPEG", "HEVC", 51, 1, 45, OutputGroup::VIDEO_FFMPEG };
OutputOption OutputOption::FFMPEG_H264 = { 2, "FFMPEG", "H264", 51, 1, 55, OutputGroup::VIDEO_FFMPEG };
OutputOption OutputOption::FFMPEG_FFV1 = { 3, "FFMPEG", "FFV1",  0, 0,  0, OutputGroup::VIDEO_FFMPEG };

OutputOption OutputOption::NVENC_AV1 =  { 4, "NVENC",  "AV1", 63, 1, 55, OutputGroup::VIDEO_NVENC };
OutputOption OutputOption::NVENC_HEVC = { 5, "NVENC", "HEVC", 51, 1, 45, OutputGroup::VIDEO_NVENC };
OutputOption OutputOption::NVENC_H264 = { 6, "NVENC", "H264", 51, 1, 55, OutputGroup::VIDEO_NVENC };

OutputOption OutputOption::VIDEO_STACK =    { 10, "FFMPEG", "H264", 51,  1, 55, OutputGroup::VIDEO_OTHER };
OutputOption OutputOption::VIDEO_FLOW =     { 11, "FFMPEG", "H264", 51,  1, 55, OutputGroup::VIDEO_OTHER };
OutputOption OutputOption::PIPE_RAW =       { 12,   "PIPE",  "RAW",  0,  0,  0, OutputGroup::PIPE };
OutputOption OutputOption::PIPE_ASF =       { 13,   "PIPE",  "ASF",  0,  0,  0, OutputGroup::PIPE };
OutputOption OutputOption::IMAGE_BMP =      { 14,  "IMAGE",  "BMP",  0,  0,  0, OutputGroup::IMAGE_SEQUENCE };
OutputOption OutputOption::IMAGE_JPG =      { 15,  "IMAGE",  "JPG", 31,  1, 80, OutputGroup::IMAGE_SEQUENCE };
OutputOption OutputOption::IMAGE_RESULTS =  { 16,  "IMAGE",  "BMP",  0,  0,  0, OutputGroup::IMAGE_SEQUENCE };

OutputOption OutputOption::RAW_YUV444 = { 20, "RAW",  "YUV444",  0, 0,  0, OutputGroup::VIDEO_OTHER };
OutputOption OutputOption::RAW_NV12 =   { 21, "RAW",    "NV12",  0, 0,  0, OutputGroup::VIDEO_OTHER };

OutputOption OutputOption::OPTION_AUTO =    { 90, "AUTO", "AUTO", 0, 0, 0, OutputGroup::AUTO };
OutputOption OutputOption::OPTION_NONE =    { 91,     "",     "", 0, 0, 0, OutputGroup::NONE };
OutputOption OutputOption::OPTION_INVALID = { 92,     "",     "", 0, 0, 0, OutputGroup::INVALID };


std::string OutputOption::displayName() const {
	return group + " - " + name;
}

std::string OutputOption::fullName() const {
	return group + ":" + name;
}

int OutputOption::defaultCrf() const {
	return percentToCrf(qualityPercent);
}

//input 0..100
int OutputOption::percentToCrf(double value) const {
	return (int) std::round(qualityMin + value / 100.0 * (qualityMax - qualityMin));
}

OutputOption OutputOption::find(std::string optionName) {
	OutputOption out;
	for (OutputOption& op : validOptions()) {
		if (optionName == op.fullName()) {
			out = op;
			break;
		}
	}
	return out;
}

bool OutputOption::isVideoFile() const {
	return device == OutputGroup::VIDEO_FFMPEG || device == OutputGroup::VIDEO_NVENC || device == OutputGroup::VIDEO_OTHER;
}

bool OutputOption::isImageSequence() const {
	return device == OutputGroup::IMAGE_SEQUENCE;
}

bool OutputOption::hasQuality() const {
	return qualityMax != qualityMin;
}

std::vector<OutputOption> OutputOption::videoOptions() {
	return { FFMPEG_AV1, FFMPEG_HEVC, FFMPEG_H264, FFMPEG_FFV1, NVENC_AV1, NVENC_HEVC, NVENC_H264 };
}

std::vector<OutputOption> OutputOption::validOptions() {
	return { 
		FFMPEG_AV1, FFMPEG_HEVC, FFMPEG_H264, FFMPEG_FFV1, NVENC_AV1, NVENC_HEVC, NVENC_H264,
		VIDEO_STACK, VIDEO_FLOW, PIPE_RAW, PIPE_ASF, IMAGE_BMP, IMAGE_JPG, IMAGE_RESULTS, RAW_YUV444, RAW_NV12
	};
}

bool OutputOption::operator < (const OutputOption& other) const {
	return id < other.id;
}

bool OutputOption::operator == (const OutputOption& other) const {
	return id == other.id;
}