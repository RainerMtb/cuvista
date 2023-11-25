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

#include "clHeaders.hpp"
#include "ErrorLogger.hpp"
#include <map>


namespace cl {
	class Size2 : public array<size_type, 2> {
	public:
		Size2(size_type x, size_type y) : array<size_type, 2>{x, y} {}
		Size2(int x, int y) : Size2(size_type(x), size_type(y)) {}
		Size2() : Size2(0, 0) {}
	};

	//definition of results structure must be the same in device code
	struct cl_PointResult {
		cl_double u, v;
		cl_int idx, ix0, iy0;
		cl_int px, py;
		cl_int xm, ym;
		cl_char result;
	};

	struct BufferImages {
		Image2D filterH, filterV, result;
	};

	struct Data {
		Context context;
		CommandQueue queue;

		// input yuv frames
		std::vector<Image2D> yuv;
		// pyramid images, one image for all levels
		std::vector<Image2D> pyramid;
		// buffers for filtering on pyramid creation
		std::vector<BufferImages> buffer;

		//data storage for output
		std::array<Image2D, 5> out;
		Buffer yuvOut;
		Buffer rgbOut;

		//storage for results struct
		Buffer results;
		std::vector<cl_PointResult> cl_results;

		std::map<std::string, Kernel> kernelMap = {
			{"scale_8u32f_1", {}},
			{"scale_8u32f_3", {}},
			{"scale_32f8u_3", {}},
			{"filter_32f_1", {}},
			{"filter_32f_3", {}},
			{"remap_downsize_32f", {}},
			{"warp_back", {}},
			{"unsharp", {}},
			{"yuv8u_to_rgb", {}},
			{"yuv32f_to_rgb", {}},
			{"scrap", {}},
			{"compute", {}},
		};

		Kernel& kernel(const std::string& key) {
			return kernelMap.at(key);
		}
	};

	void scale_8u32f_1(Image src, Image dest, Data& clData);
	void scale_8u32f_3(Image src, Image dest, Data& clData);
	void scale_32f8u_3(Image src, Buffer dest, int pitch, Data& clData);

	void filter_32f_h1(Image src, Image dest, int filterIndex, Data& clData);
	void filter_32f_h3(Image src, Image dest, Data& clData);
	void filter_32f_v1(Image src, Image dest, int filterIndex, Data& clData);
	void filter_32f_v3(Image src, Image dest, Data& clData);

	void remap_downsize_32f(Image src, Image dest, Data& clData);
	void warp_back(Image src, Image dest, Data& clData, std::array<double, 6> trf);
	void unsharp(Image src, Image dest, Image gauss, Data& clData, cl_float4 factor);

	void yuv_to_rgb(const std::string& kernelName, Image src, unsigned char* imageData, Data& clData, int w, int h);

	void readImage(Image src, size_t destPitch, void* dest, CommandQueue queue);
}