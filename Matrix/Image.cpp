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
	assert(h >= 0 && w >= 0 && "invalid dimensions");
	assert(stride >= w && "stride must be equal or greater to width");
}

template <class T> ImageBase<T>::ImageBase(int h, int w, int stride, int numPlanes) : 
	h { h },
	w { w },
	stride { stride },
	numPlanes { numPlanes },
	array(1ull * h * stride * numPlanes) 
{
	assert(h >= 0 && w >= 0 && "invalid dimensions");
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

template <class T> void ImageBase<T>::setPixel(size_t row, size_t col, std::vector<T> colors) {
	assert(numPlanes == colors.size() && "image plane count does not match number of color values");
	for (int i = 0; i < numPlanes; i++) {
		at(i, row, col) = colors[i];
	}
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
	for (int z = 0; z < color.colors.size(); z++) {
		setValues(z, color.colors[z]);
	}
}

template <class T> void ImageBase<T>::setArea(size_t r0, size_t c0, const ImageBase<T>& src, const ImageGray& mask) {
	assert(w >= r0 + src.w && h >= c0 + src.h && "pixel coordinates exceeding image bounds");
	for (size_t r = 0; r < src.h; r++) {
		for (size_t c = 0; c < src.w; c++) {
			if (mask.at(0, r, c) > 0) {
				for (size_t z = 0; z < numPlanes; z++) {
					at(z, r0 + r, c0 + c) = src.at(z, r, c);
				}
			}
		}
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
	pool.addAndWait(func, 0, h);
}

template <class T> void ImageBase<T>::shuffle8(ImageBase<T>& dest, int z0, int z1, int z2, ThreadPoolBase& pool) const {
	assert(w == dest.w && h == dest.h && "dimensions mismatch");
	auto func = [&] (size_t r) {
		for (int c = 0; c < w; c++) {
			dest.at(z0, r, c) = at(0, r, c);
			dest.at(z1, r, c) = at(1, r, c);
			dest.at(z2, r, c) = at(2, r, c);
		}
	};
	pool.addAndWait(func, 0, h);
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

template <class T> double ImageBase<T>::fpart(double d) {
	return d - floor(d);
}

template <class T> double ImageBase<T>::rfpart(double d) {
	return 1.0 - fpart(d);
}

template <class T> void ImageBase<T>::drawLine(double x0, double y0, double x1, double y1, ColorBase<T> color, double alpha) {
	/*
	Xiaolin Wu's line algorithm
	https://en.wikipedia.org/wiki/Xiaolin_Wu%27s_line_algorithm
	*/
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
		plot(ypxl1, xpxl1, rfpart(yend) * xgap * alpha, color);
		plot(ypxl1 + 1, xpxl1, fpart(yend) * xgap * alpha, color);

	} else {
		plot(xpxl1, ypxl1, rfpart(yend) * xgap * alpha, color);
		plot(xpxl1, ypxl1 + 1, fpart(yend) * xgap * alpha, color);
	}
	double inter = yend + g;

	//second endpoint
	xend = round(x1);
	yend = y1 + g * (xend - x1);
	xgap = fpart(x1 + 0.5);
	double xpxl2 = xend;
	double ypxl2 = floor(yend);

	if (steep) {
		plot(ypxl2, xpxl2, rfpart(yend) * xgap * alpha, color);
		plot(ypxl2 + 1, xpxl2, fpart(yend) * xgap * alpha, color);

	} else {
		plot(xpxl2, ypxl2, rfpart(yend) * xgap * alpha, color);
		plot(xpxl2, ypxl2 + 1, fpart(yend) * xgap * alpha, color);
	}

	//main loop
	if (steep) {
		for (double x = xpxl1 + 1.0; x < xpxl2; x++) {
			plot(floor(inter), x, rfpart(inter) * alpha, color);
			plot(floor(inter) + 1, x, fpart(inter) * alpha, color);
			inter += g;
		}

	} else {
		for (double x = xpxl1 + 1.0; x < xpxl2; x++) {
			plot(x, floor(inter), rfpart(inter) * alpha, color);
			plot(x, floor(inter) + 1, fpart(inter) * alpha, color);
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
		int quarterX = int(rx2 / h + 0.5);
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
		int quarterY = int(ry2 / h + 0.5);
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
	const int steps = 8;
	constexpr double ds = 1.0 / steps;
	//align center to nearest fraction
	cx = std::round((cx + 0.5) * steps) / steps;
	cy = std::round((cy + 0.5) * steps) / steps;

	//collect subpixels that need to be drawn
	std::map<int, int> alpha;

	double rx2 = sqr(rx);
	double ry2 = sqr(ry);
	for (double x = ds / 2; x <= rx; x += ds) {
		//y on the ellipse circumference
		double ey = sqrt(ry2 - sqr(x) * ry2 / rx2);
		for (double y = ds / 2; y <= ey; y += ds) {
			//set subpixels 4 times around center
			for (double px : { cx - x, cx + x}) {
				for (double py : { cy - y, cy + y }) {
					int ix = int(px);
					int iy = int(py);
					int idx = iy * stride + ix;
					alpha[idx]++;
				}
			}
		}
	}

	for (auto& [idx, a] : alpha) {
		int iy = idx / stride;
		int ix = idx % stride;
		if (ix >= 0 && ix < w && iy >= 0 && iy < h) {
			plot(ix, iy, a * ds * ds, color);
		}
	}
}


//------------------------
// image mat
//------------------------

//construct Mat per image plane
template <class T> ImageMat<T>::ImageMat(T* data, int h, int w, int stride, int planeIdx) :
	CoreMat<T>(data, h, stride, false) {}


//------------------------
// planar image
//------------------------

template <class T> ImagePlanar<T>::ImagePlanar(int h, int w, int stride, int numPlanes) :
	ImageBase<T>(h, w, stride, numPlanes) {
	for (int i = 0; i < numPlanes; i++) {
		size_t offset = 1ull * i * h * stride;
		mats.emplace_back(this->data() + offset, h, w, stride, i);
	}
}

template <class T> ImagePlanar<T>::ImagePlanar(int h, int w, T* y, T* u, T* v) :
	ImageBase<T>(h, w, w, 3, 0) {
	mats.emplace_back(y, h, w, w, 0);
	mats.emplace_back(u, h, w, w, 1);
	mats.emplace_back(v, h, w, w, 2);
}

//read access one pixel on plane idx (0..2) and row / col
template <class T> T* ImagePlanar<T>::addr(size_t idx, size_t r, size_t c) {
	return mats[idx].addr(r, c);
}

//read access one pixel on plane idx (0..2) and row / col
template <class T> const T* ImagePlanar<T>::addr(size_t idx, size_t r, size_t c) const {
	return mats[idx].addr(r, c);
}

//pointer to start of color plane
template <class T> T* ImagePlanar<T>::plane(size_t idx) {
	return addr(idx, 0, 0);
}

//pointer to start of color plane
template <class T> const T* ImagePlanar<T>::plane(size_t idx) const {
	return addr(idx, 0, 0);
}

template <class T> void ImagePlanar<T>::scaleTo(ImagePlanar<T>& dest) const {
	for (size_t z = 0; z < mats.size(); z++) {
		scaleTo(z, dest, z);
	}
}

template <class T> void ImagePlanar<T>::scaleTo(size_t srcPlane, ImageBase<T>& dest, size_t destPlane) const {
	for (int r = 0; r < dest.h; r++) {
		for (int c = 0; c < dest.w; c++) {
			double py = (0.5 + r) * this->h / dest.h - 0.5;
			double px = (0.5 + c) * this->w / dest.w - 0.5;
			dest.at(destPlane, r, c) = this->sample(srcPlane, px, py);
		}
	}
}

template <class T> T ImagePlanar<T>::sample(size_t plane, double x, double y) const {
	return (T) mats[plane].interp2clamped(x, y);
}


//------------------------------------------------
//explicitly instantiate Image specializations
//------------------------------------------------

template class ImageBase<float>;
template class ImageBase<unsigned char>;
template class ImagePlanar<float>;
template class ImagePlanar<unsigned char>;
