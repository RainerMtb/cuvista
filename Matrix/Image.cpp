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

#include "Image.hpp"
#include <cassert>
#include <algorithm>

//allocate frame given height, width, and stride
template <class T> ImageBase<T>::ImageBase(int h, int w, int stride, int numPlanes, int arraySize) : 
	h { h }, 
	w { w }, 
	stride { stride }, 
	numPlanes { numPlanes }, 
	array(arraySize) 
{
	assert(stride >= w && "stride must be equal or greater to width");
}

template <class T> ImageBase<T>::ImageBase(int h, int w, int stride, int numPlanes) : 
	h { h },
	w { w },
	stride { stride },
	numPlanes { numPlanes },
	array(1ull * h * stride * numPlanes) 
{
	assert(stride >= w && "stride must be equal or greater to width");
}

template <class T> T* ImageBase<T>::data() {
	return array.data();
}

template <class T> const T* ImageBase<T>::data() const {
	return array.data();
}

//access one pixel on plane idx (0..2) and row / col
template <class T> T& ImageBase<T>::at(size_t idx, size_t r, size_t c) {
	return *addr(idx, r, c);
}

//read access one pixel on plane idx (0..2) and row / col
template <class T> const T& ImageBase<T>::at(size_t idx, size_t r, size_t c) const {
	return *addr(idx, r, c);
}

template <class T> int ImageBase<T>::colorValue(T pixelValue) const {
	return pixelValue;
}

template <> int ImageBase<float>::colorValue(float pixelValue) const {
	return int(pixelValue * 255.0f);
}

template <> int ImageBase<double>::colorValue(double pixelValue) const {
	return int(pixelValue * 255.0);
}

template <class T> bool ImageBase<T>::operator == (const ImageBase<T>& other) const {
	return compareTo(other) == 0.0;
}

template <class T> double ImageBase<T>::compareTo(const ImageBase<T>& other) const {
	double out = nan("");
	if (h == other.h && w == other.w) {
		//count differences
		int histogram[256] = { 0 };
		for (size_t z = 0; z < 3; z++) {
			for (size_t r = 0; r < h; r++) {
				for (size_t c = 0; c < w; c++) {
					T pix1 = at(z, r, c);
					T pix2 = other.at(z, r, c);
					int delta = colorValue(std::abs(pix1 - pix2));
					histogram[delta]++;
				}
			}
		}

		//calculate center of gravity
		int a = 0, w = 0;
		for (int i = 0; i < 256; i++) {
			a += histogram[i];
			w += histogram[i] * i;
		}
		out = 1.0 * w / a;
	}
	return out;
}

template <class T> void ImageBase<T>::setValues(int plane, T colorValue) {
	for (size_t r = 0; r < h; r++) {
		for (size_t c = 0; c < w; c++) {
			at(plane, r, c) = colorValue;
		}
	}
}

template <class T> void ImageBase<T>::setValues(const ColorBase<T>& color) {
	for (int z = 0; z < 3; z++) {
		setValues(z, color.colors[z]);
	}
}

template <class T> size_t ImageBase<T>::dataSizeInBytes() const {
	return array.size();
}

template <class T> void ImageBase<T>::convert8(ImageBase<unsigned char>& dest, int z0, int z1, int z2, ThreadPoolBase& pool) const {
	assert(w == dest.w && h == dest.h && "dimensions mismatch");
	auto func = [&] (size_t r) {
		for (int c = 0; c < w; c++) {
			yuv_to_rgb(at(0, r, c), at(1, r, c), at(2, r, c), dest.addr(z0, r, c), dest.addr(z1, r, c), dest.addr(z2, r, c));
		}
	};
	pool.add(func, 0, h);
}


//------------------------
//write text
//------------------------

template <class T> void ImageBase<T>::writeText(std::string_view text, int x0, int y0, int scaleX, int scaleY, ColorBase<T> fg, ColorBase<T> bg) {
	//fill background area
	for (int x = x0; x < x0 + scaleX + int(text.size()) * 6 * scaleX; x++) { //TODO
		for (int y = y0; y < y0 + 10 * scaleY; y++) {
			if (y < h && x < w) {
				plot(x, y, 1.0, bg);
			}
		}
	}

	//write foreground characters
	for (int charIdx = 0; charIdx < text.size(); charIdx++) {
		char ch = text.at(charIdx); //one character from the string
		auto it = charMap.find(ch);
		uint64_t bitmap = it == charMap.end() ? charMap.at('\0') : it->second; //mapping sequence 0 or 1

		for (int yi = 7; yi >= 0; yi--) { //row of character map
			for (int xi = 4; xi >= 0; xi--) { //column of digit image
				int x = x0 + scaleX + charIdx * 6 * scaleX + xi * scaleX; //character pixel to set
				int y = y0 + scaleY + yi * scaleY;  //character pixel to set
				if (bitmap & 1) {
					for (int sy = 0; sy < scaleY; sy++) {
						for (int sx = 0; sx < scaleX; sx++) {
							int xx = x + sx;
							int yy = y + sy;
							if (yy < h && xx < w) {
								plot(xx, yy, 1.0, fg);
							}
						}
					}
				}
				bitmap >>= 1; //next bitmap character
			}
		}
	}
}


//------------------------
//drawing funtions
//------------------------

template <class T> void ImageBase<T>::plot(int x, int y, double a, ColorBase<T> color) {
	if (x >= 0 && x < w && y >= 0 && y < h) {
		double alpha = a * color.alpha;
		for (int z = 0; z < 3; z++) {
			T* pix = addr(z, y, x);
			*pix = T(*pix * (1.0 - alpha) + color.colors[z] * alpha);
		}
	}
}

template <class T> void ImageBase<T>::plot(double x, double y, double a, ColorBase<T> color) {
	plot(int(x), int(y), a, color);
}

template <class T> void ImageBase<T>::plot4(double cx, double cy, double dx, double dy, double a, ColorBase<T> color) {
	plot(cx + dx, cy + dy, a, color);
	plot(cx - dx, cy + dy, a, color);
	plot(cx + dx, cy - dy, a, color);
	plot(cx - dx, cy - dy, a, color);
}

double fpart(double d) {
	return d - floor(d);
}

double rfpart(double d) {
	return 1.0 - fpart(d);
}

template <class T> void ImageBase<T>::drawLine(double x0, double y0, double x1, double y1, ColorBase<T> color) {
	bool steep = std::abs(y1 - y0) > std::abs(x1 - x0);
	if (steep) {
		std::swap(x0, y0);
		std::swap(x1, y1);
	}
	if (x0 > x1) {
		std::swap(x0, x1);
		std::swap(y0, y1);
	}

	double dx = x1 - x0;
	double dy = y1 - y0;
	double g = dx == 0.0 ? 1.0 : dy / dx;

	//first endpoint
	double xend = round(x0);
	double yend = y0 + g * (xend - x0);
	double xgap = rfpart(x0 + 0.5);
	double xpxl1 = xend;
	double ypxl1 = floor(yend);

	if (steep) {
		plot(ypxl1, xpxl1, rfpart(yend) * xgap, color);
		plot(ypxl1 + 1, xpxl1, fpart(yend) * xgap, color);

	} else {
		plot(xpxl1, ypxl1, rfpart(yend) * xgap, color);
		plot(xpxl1, ypxl1 + 1, fpart(yend) * xgap, color);
	}
	double inter = yend + g;

	//second endpoint
	xend = round(x1);
	yend = y1 + g * (xend - x1);
	xgap = fpart(x1 + 0.5);
	double xpxl2 = xend;
	double ypxl2 = floor(yend);

	if (steep) {
		plot(ypxl2, xpxl2, rfpart(yend) * xgap, color);
		plot(ypxl2 + 1, xpxl2, fpart(yend) * xgap, color);

	} else {
		plot(xpxl2, ypxl2, rfpart(yend) * xgap, color);
		plot(xpxl2, ypxl2 + 1, fpart(yend) * xgap, color);
	}

	//main loop
	if (steep) {
		for (double x = xpxl1; x < xpxl2; x++) {
			plot(floor(inter), x, rfpart(inter), color);
			plot(floor(inter) + 1, x, fpart(inter), color);
			inter += g;
		}

	} else {
		for (double x = xpxl1; x < xpxl2; x++) {
			plot(x, floor(inter), rfpart(inter), color);
			plot(x, floor(inter) + 1, fpart(inter), color);
			inter += g;
		}
	}
}

template <class T> void ImageBase<T>::drawEllipse(double cx, double cy, double rx, double ry, ColorBase<T> color, bool fill) {
	double rx2 = sqr(rx);
	double ry2 = sqr(ry);
	double h = sqrt(rx2 + ry2);

	if (rx < 5 && ry < 5) {
		//std::map<std::set<int>> pixels;

	} else {
		//upper and lower halves
		int quarterX = int(round(rx2 / h));
		for (double x = 0; x <= quarterX; x++) {
			double y = ry * sqrt(1.0 - x * x / rx2);
			double alpha = fpart(y);
			double fly = floor(y);

			plot4(cx, cy, x, fly + 1.0, alpha, color);
			if (fill) {
				for (int i = 0; i <= fly; i++) {
					plot4(cx, cy, x, i, 1.0, color);
				}

			} else {
				plot4(cx, cy, x, fly, 1.0 - alpha, color);
			}
		}

		//right and left halves
		int quarterY = int(round(ry2 / h));
		for (double y = 0; y <= quarterY; y++) {
			double x = rx * sqrt(1.0 - y * y / ry2);
			double alpha = fpart(x);
			double flx = floor(x);

			plot4(cx, cy, floor(x) + 1.0, y, alpha, color);
			if (fill) {
				for (int i = quarterX; i <= flx; i++) {
					plot4(cx, cy, i, y, 1.0, color);
				}

			} else {
				plot4(cx, cy, floor(x), y, 1.0 - alpha, color);
			}
		}
	}
}

template <class T> void ImageBase<T>::drawCircle(double cx, double cy, double r, ColorBase<T> color, bool fill) {
	drawEllipse(cx, cy, r, r, color, fill);
}

template <class T> void ImageBase<T>::drawDot(double cx, double cy, double rx, double ry, ColorBase<T> color) {
	//collect subpixels that need to be drawn
	int x0 = int(cx - rx);
	int y0 = int(cy - ry);
	int x1 = int(cx + rx + 2);
	int y1 = int(cy + ry + 2);
	int nx = x1 - x0;
	int ny = y1 - y0;
	std::vector<int8_t> alphaMap(1ull * nx * ny);

	const int steps = 8;
	double ds = 1.0 / steps;

	double rx2 = sqr(rx);
	double ry2 = sqr(ry);
	for (double x = ds / 2; x < rx + ds / 2; x += ds) {
		//y on the ellipse circumference
		double dy = sqrt(ry2 - sqr(x) * ry2 / rx2);
		for (double y = ds / 2; y < dy + ds / 2; y += ds) {
			//set subpixels 4 times around center
			for (double px : { cx + 0.5 - x, cx + 0.5 + x}) {
				for (double py : { cy + 0.5 - y, cy + 0.5 + y }) {
					int ix = int(px * steps + 0.5) / steps - x0;
					int iy = int(py * steps + 0.5) / steps - y0;
					alphaMap[1ull * iy * nx + ix]++;
				}
			}
		}
	}

	for (int x = 0; x < nx; x++) {
		for (int y = 0; y < ny; y++) {
			int a = alphaMap[1ull * y * nx + x];
			int ix = x + x0;
			int iy = y + y0;
			if (ix >= 0 && ix < w && iy >= 0 && iy < h) {
				plot(ix, iy, a * ds * ds, color);
			}
		}
	}
}


//------------------------------------------------
//explicitly instantiate Image specializations
//------------------------------------------------
template class ImageBase<unsigned char>;
template class ImageBase<float>;
