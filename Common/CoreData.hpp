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
#include "AffineData.hpp"

struct Triplet {
	float y, u, v;

	float operator [] (size_t idx) const;
};


//how to deal with background when frame does not cover complete output canvas
enum class BackgroundMode {
	BLEND,
	COLOR
};

inline constexpr struct {
	double radsecMin = 0.1, radsecMax = 10.0;
	double radsec = 0.5;

	double imZoomMin = 0.1, imZoomMax = 10.0;
	double zoomMin = 1.05, zoomMax = 1.15;

	int radiusMin = 1, radiusMax = 500;
	int wMin = 100, hMin = 100;
	int levelsMin = 1, levelsMax = 6, levels = 3;
	int irMin = 0, irMax = 3;
	int modeMax = 6;
	
	int encodingQuality = 60;

	uint8_t bgColorRed = 0;
	uint8_t bgColorGreen = 150;
	uint8_t bgColorBlue = 0;
	int frameLimit = 500;
} defaultParam;

struct CoreData {
	int maxResultCount = 12'000;   //max number of result points to calculate
	int compMaxIter = 20;          //max loop iterations
	double compMaxTol = 0.15;      //tolerance to stop window pattern matching

	int w = 0;                     //frame width
	int h = 0;					   //frame height
	int ir = 3;					   //integration window, radius around point to integrate
	int iw = 7;					   //integration window, 2 * ir + 1
	int ixCount = -1;			   
	int iyCount = -1;			   
	int zMin = -1;				   
	int zMax = -1;				   //pyramid steps used for actual computing
	int pyramidLevelsRequested = defaultParam.levels;  //number of pyramid levels wanted for stabilization
	int pyramidLevels = -1;        //number of pyramid levels to create
	int pyramidRowCount = -1;	   //number of rows for one pyramid, all the rows of Y data for all levels
	int pyramidCount = 2;	       //number of pyramids to allocate in memory
	int resultCount = 0;		   //number of points to compute in a frame

	int cpupitch = 0;

	//numeric constants used in compute kernel, will be initialized once
	double dmin = std::numeric_limits<double>::min();
	double dmax = std::numeric_limits<double>::max();
	double deps = std::numeric_limits<double>::epsilon();
	double dnan = std::numeric_limits<double>::quiet_NaN();

	Triplet unsharp = { 0.6f, 0.3f, 0.3f };          //ffmpeg unsharp=5:5:0.6:3:3:0.3
	Triplet bgcolorYuv = {};                         //background fill colors in yuv
	BackgroundMode bgmode = BackgroundMode::BLEND;   //fill gap with previous frames or not

	int radius = -1;                    //temporal radius, number of frames before and after used for smoothing
	double radsec = defaultParam.radsec;    //temporal radius in seconds
	int bufferCount = -1;               //number of frames to buffer, set by MovieFrame

	double zoomMin = defaultParam.zoomMin;  //min additional zoom
	double zoomMax = defaultParam.zoomMax;  //max additioanl zoom
	double zoomFallbackTotal = 0.025;   //fallback rate for dynamic zoom, to be divided by temporal radius
	double zoomFallback = 0.0;          //fallback rate for dynamic zoom, to be applied per frame
	
	int cpuThreads = 1;                 //cpu threads to use in different places, leave room for other things
	unsigned int cudaThreadCount = 16;  //thread count used for texture reading
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
	int idx;        //absolute linear index of this point
	int ix0, iy0;   //absolute index per dimension
	double x, y;    //pixel coordinate with respect to image center
	double u, v;    //calculated translation of center point
	PointResultType result;
	int z;
	int direction;

	double length;        //displacement vector length
	bool isConsidered;    //flag for computing frame result
	bool isConsens;       //flag for computing frame result

	//is valid when numeric stable result was found
	bool isValid() const;

	//numeric value for type of result
	int resultValue() const;

	bool equals(const PointResult& other, double tol = 0.0) const;

	bool operator == (const PointResult& other) const;

	friend std::ostream& operator << (std::ostream& out, const PointResult& res);
};

struct PointContext {
	PointResult* ptr;
	double delta = 0.0;
	double distance = 0.0;
	double distanceRelative = 0.0;
	int clusterIndex = -2;
	int clusterGeneration = 0;

	PointContext(PointResult& pr) : ptr { &pr } {}
	PointContext() : ptr { nullptr } {}

	bool operator == (const PointContext& other) const;
};

struct PointBase {
	double x, y;
	double u, v;

	PointBase() : x { 0.0 }, y { 0.0 }, u { 0.0 }, v { 0.0 } {}
	PointBase(const PointResult& pr) : x { pr.x }, y { pr.y }, u { pr.u }, v { pr.v } {}
	PointBase(const PointContext& pc) : x { pc.ptr->x }, y { pc.ptr->y }, u { pc.ptr->u }, v { pc.ptr->v } {}

	bool operator == (const PointBase& other) const;
};