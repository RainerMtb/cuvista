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

#include "ErrorLogger.hpp"
#include "Image.hpp"

//how to put original and stabilized video side by side
struct BlendInput {
	double percent = 0.0;
	int blendStart = 0;
	int blendWidth = 0;
	int separatorStart = 0;
	int separatorWidth = 0;
};

struct Triplet {
	float y, u, v;

	float operator [] (size_t idx) const { 
		if (idx == 0) return y;
		if (idx == 1) return u;
		if (idx == 2) return v;
		throw std::exception("invalid index");
	}
};

struct OutputContext {
	bool encodeCpu;
	bool encodeCuda;
	ImageYuv* outputFrame;
	unsigned char* cudaNv12ptr;
	int cudaPitch;
};

//how to deal with background when frame does not cover complete output canvas
enum class BackgroundMode {
	BLEND,
	COLOR
};

struct CoreData {
	//inside CoreData struct all values must be initialized to be used as __constant__ variable in device code

protected:
	int MAX_POINTS_COUNT = 150;		//max number of points in x or y direction

public:
	int zMin = -1;
	int zMax = -1;				    //pyramid steps used for actual computing
	int zCount = 3;			        //number of pyramid levels to use for stabilization
	int pyramidLevels = -1;         //number of pyramid levels to create
	int pyramidRowCount = -1;	    //number of rows for one pyramid, for example all the rows of Y data
	size_t pyramidCount = 2;	    //number of pyramids to allocate in memory

	int cpupitch = 0;
	int compMaxIter = 20;			//max loop iterations
	double compMaxTol = 0.05;		//tolerance to stop window pattern matching

	Triplet unsharp = { 0.6f, 0.3f, 0.3f };	//ffmpeg unsharp=5:5:0.6:3:3:0.3
	ColorNorm bgcol_yuv = {};				//background fill colors in yuv
	BlendInput blendInput = {};
	BackgroundMode bgmode = BackgroundMode::BLEND;

	int ir = 3;					//integration window, radius around point to integrate
	int iw = 7;					//integration window, 2 * ir + 1
	double imZoom = 1.05;		//additional zoom
	double radsec = 0.5;		//radius in senconds
	int radius = -1;			//number of frames before and after used for smoothing

	int w = 0;                  //frame width
	int h = 0;					//frame height
	int64_t frameCount = -1;

	int bufferCount = -1;		//number of frames to read before starting to average out trajectory

	int ixCount = -1;
	int iyCount = -1;
	int resultCount = 0;		//number of points to compute in a frame

	uint8_t crf = 22;
	char fileDelimiter = ';';
};

//result type of one computed point
enum class PointResultType {
	FAIL_SINGULAR = -3,
	FAIL_ITERATIONS,
	FAIL_ETA_NAN,
	RUNNING = 0,
	SUCCESS_ABSOLUTE_ERR,
	SUCCESS_STABLE_ITER,
};

//result of one computed point in a frame
struct PointResult {

private:
	bool equal(double a, double b, double tol) const;

public:
	size_t idx, ix0, iy0;
	int px, py;
	int x, y;
	double u, v;
	PointResultType result;
	double distance;
	double length;
	double distanceRelative;

	//is valid when numeric stable result was found
	bool isValid() const;

	//numeric value for type of result
	int resultValue() const;

	bool equals(const PointResult& other, double tol = 0.0) const;

	bool operator == (const PointResult& other) const;

	bool operator != (const PointResult& other) const;

	friend std::ostream& operator << (std::ostream& out, const PointResult& res);
};

//simple class to measure runtime
class ConsoleTimer {

	std::chrono::steady_clock::time_point t1;

public:
	ConsoleTimer() {
		t1 = std::chrono::high_resolution_clock::now();
	}

	~ConsoleTimer() {
		auto t2 = std::chrono::high_resolution_clock::now();
		auto delta = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
		std::printf("elapsed %zd us\n", delta);
	}
};