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

#include "MainData.hpp"
#include "Trajectory.hpp"

class DiagnosticItem {

public:
	virtual void run(const FrameResult& fr, int64_t frameIndex) = 0;
};


class TransformsFile : std::fstream, public DiagnosticItem {

private:
	std::string id = "CUVI";

	template <class T> void writeValue(T val) {
		write(reinterpret_cast<const char*>(&val), sizeof(val));
	}

	template <class T> void readValue(T& val) {
		read(reinterpret_cast<char*>(&val), sizeof(val));
	}

public:
	TransformsFile(const std::string& filename, std::ios::openmode mode);

	void writeTransform(const Affine2D& transform, int64_t frameIndex);

	std::map<int64_t, TransformValues> readTransformMap();

	virtual void run(const FrameResult& fr, int64_t frameIndex) override;
};


class ResultDetailsFile : std::ofstream, public DiagnosticItem {

private:
	std::string delimiter = ";";

public:
	ResultDetailsFile(const std::string& filename);

	void write(const std::vector<PointResult>& results, int64_t frameIndex);

	virtual void run(const FrameResult& fr, int64_t frameIndex) override;
};


class ResultImage : public DiagnosticItem {

private:
	ImageBGR bgr;
	const MainData& data;
	std::function<ImageYuv(uint64_t)> imageGetter;

public:
	ResultImage(const MainData& data, std::function<ImageYuv(uint64_t)> imageGetter) : bgr(data.h, data.w), data { data }, imageGetter { imageGetter } {}

	void write(const FrameResult& fr, int64_t idx, const ImageYuv& yuv, const std::string& fname);

	virtual void run(const FrameResult& fr, int64_t frameIndex) override;
};