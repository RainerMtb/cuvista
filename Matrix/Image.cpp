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


template <class T> void im::ImageBase<T>::setPixel(size_t row, size_t col, std::vector<T> colors) {
	assert(numPlanes == colors.size() && "image plane count does not match number of color values");
	for (int i = 0; i < numPlanes; i++) {
		at(i, row, col) = colors[i];
	}
}

template <class T> int im::ImageBase<T>::colorValue(T pixelValue) const {
	return pixelValue;
}

template <> int im::ImageBase<float>::colorValue(float pixelValue) const {
	return int(pixelValue * 255.0f);
}

template <> int im::ImageBase<double>::colorValue(double pixelValue) const {
	return int(pixelValue * 255.0);
}

template <class T> bool im::ImageBase<T>::operator == (const ImageBase<T>& other) const {
	return compareTo(other) == 0.0;
}

template <class T> double im::ImageBase<T>::compareTo(const ImageBase<T>& other) const {
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

template <class T> void im::ImageBase<T>::setValues(int plane, T colorValue) {
	for (size_t r = 0; r < h; r++) {
		for (size_t c = 0; c < w; c++) {
			at(plane, r, c) = colorValue;
		}
	}
}

template <class T> void im::ImageBase<T>::setValues(const ColorBase<T>& color) {
	for (int z = 0; z < color.colors.size(); z++) {
		setValues(z, color.colors[z]);
	}
}

template <class T> void im::ImageBase<T>::yuvToRgb(ImageBase<unsigned char>& dest, std::vector<int> planes, ThreadPoolBase& pool) const {
	assert(w == dest.w && h == dest.h && "dimensions mismatch");
	auto func = [&] (size_t r) {
		for (int c = 0; c < w; c++) {
			yuv_to_rgb(at(0, r, c), at(1, r, c), at(2, r, c), dest.addr(planes[0], r, c), dest.addr(planes[1], r, c), dest.addr(planes[2], r, c));
		}
	};
	pool.addAndWait(func, 0, h);
}

template <class T> void im::ImageBase<T>::copyTo(ImageBase<T>& dest, size_t r0, size_t c0, std::vector<int> srcPlanes, std::vector<int> destPlanes, ThreadPoolBase& pool) const {
	assert(w <= dest.w && h <= dest.h && numPlanes == dest.numPlanes && "dimensions mismatch");
	auto func = [&] (size_t i) {
		for (size_t r = 0; r < h; r++) {
			const T* ptr = addr(srcPlanes[i], r, 0);
			std::copy(ptr, ptr + w, dest.addr(destPlanes[i], r + r0, c0));
		}
	};
	pool.addAndWait(func, 0, numPlanes);
	dest.index = index;
}

template <class T> void im::ImageBase<T>::copyTo(ImageBase<T>& dest, std::vector<int> srcPlanes, std::vector<int> destPlanes, ThreadPoolBase& pool) const {
	copyTo(dest, 0, 0, srcPlanes, destPlanes, pool);
}

template <class T> void im::ImageBase<T>::copyTo(ImageBase<T>& dest, ThreadPoolBase& pool) const {
	assert(w == dest.w && h == dest.h && numPlanes == dest.numPlanes && "dimensions mismatch");
	auto func = [&] (size_t i) {
		for (size_t r = 0; r < h; r++) {
			std::copy(addr(i, r, 0), addr(i, r, 0) + w, dest.addr(i, r, 0));
		}
	};
	pool.addAndWait(func, 0, numPlanes);
	dest.index = index;
}


//------------------------
//write text
//------------------------

template <class T> void im::ImageBase<T>::writeText(std::string_view text, int x0, int y0, int scaleX, int scaleY, ColorBase<T> fg, ColorBase<T> bg) {
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

template <class T> void im::ImageBase<T>::plot(int x, int y, double a, ColorBase<T> color) {
	if (x >= 0 && x < w && y >= 0 && y < h) {
		double alpha = a * color.alpha;
		for (int z = 0; z < 3; z++) {
			T* pix = addr(z, y, x);
			*pix = T(*pix * (1.0 - alpha) + color.colors[z] * alpha);
		}
	}
}

template <class T> void im::ImageBase<T>::plot(double x, double y, double a, ColorBase<T> color) {
	plot(int(x), int(y), a, color);
}

template <class T> void im::ImageBase<T>::plot4(double cx, double cy, double dx, double dy, double a, ColorBase<T> color) {
	plot(cx + dx, cy + dy, a, color);
	plot(cx - dx, cy + dy, a, color);
	plot(cx + dx, cy - dy, a, color);
	plot(cx - dx, cy - dy, a, color);
}

template <class T> double im::ImageBase<T>::fpart(double d) {
	return d - floor(d);
}

template <class T> double im::ImageBase<T>::rfpart(double d) {
	return 1.0 - fpart(d);
}

template <class T> void im::ImageBase<T>::drawLine(double x0, double y0, double x1, double y1, ColorBase<T> color, double alpha) {
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

template <class T> void im::ImageBase<T>::drawEllipse(double cx, double cy, double rx, double ry, ColorBase<T> color, bool fill) {
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

template <class T> void im::ImageBase<T>::drawCircle(double cx, double cy, double r, ColorBase<T> color, bool fill) {
	drawEllipse(cx, cy, r, r, color, fill);
}

template <class T> void im::ImageBase<T>::drawDot(double cx, double cy, double rx, double ry, ColorBase<T> color) {
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

template <class T> void im::ImageBase<T>::scaleByTwo(ImageBase<T>& dest) const {
	for (size_t z = 0; z < arrays.size(); z++) {
		scaleByTwo(z, dest, z);
	}
}

template <class T> void im::ImageBase<T>::scaleByTwo(size_t srcPlane, ImageBase<T>& dest, size_t destPlane) const {
	for (int r = 0; r < dest.h; r++) {
		for (int c = 0; c < dest.w; c++) {
			double py = (0.5 + r) * this->h / dest.h - 0.5;
			double px = (0.5 + c) * this->w / dest.w - 0.5;
			dest.at(destPlane, r, c) = this->sample(srcPlane, px, py);
		}
	}
}

template <class T> T im::ImageBase<T>::sample(size_t plane, double x, double y) const {
	double cx = std::clamp(x, 0.0, w - 1.0), cy = std::clamp(y, 0.0, h - 1.0);
	double flx = std::floor(cx), fly = std::floor(cy);
	double dx = cx - flx, dy = cy - fly;
	size_t ix = size_t(flx), iy = size_t(fly);
	double f00 = at(plane, iy, ix);
	size_t xd = dx != 0;
	size_t yd = dy != 0;
	double f01 = at(plane, iy, ix + xd);
	double f10 = at(plane, iy + yd, ix);
	double f11 = at(plane, iy + yd, ix + xd);
	double result = ((1 - dx) * (1 - dy) * f00 + (1 - dx) * dy * f10 + dx * (1 - dy) * f01 + dx * dy * f11);
	return (T) result;
}

template <class T> bool im::ImageBase<T>::saveAsBMP(const std::string& filename, T scale) const {
	std::ofstream os(filename, std::ios::binary);
	int h = this->h;
	int w = this->w;

	im::BmpGrayHeader(w, h * numPlanes).writeHeader(os);
	size_t stridedWidth = alignValue(w, 4);
	std::vector<char> data(stridedWidth);

	for (int i = numPlanes - 1; i >= 0; i--) {
		for (int r = h - 1; r >= 0; r--) {
			//prepare one line of bgr data
			for (int c = 0; c < w; c++) data[c] = (unsigned char) std::round(at(i, r, c) * scale);
			//write strided line
			os.write(data.data(), stridedWidth);
		}
	}
	return os.good();
}

template <class T> bool im::ImageBase<T>::saveAsPPM(const std::string& filename, T scale) const {
	std::ofstream os(filename, std::ios::binary);
	im::PgmHeader(stride, h).writeHeader(os);
	std::vector<char> data(w);

	for (int i = 0; i < numPlanes; i++) {
		for (int r = 0; r < h; r++) {
			for (int c = 0; c < w; c++) data[c] = (unsigned char) std::round(at(i, r, c) * scale);
			os.write(data.data(), w);
		}
	}
	return os.good();
}



//------------------------
// packed image
//------------------------

template <class T> void im::ImagePacked<T>::copyTo(ImageBase<T>& dest) const {
	assert(this->w == dest.w && this->h == dest.h && this->stride <= dest.stride && "invalid dimensions");
	size_t linesize = this->w * this->numPlanes;
	for (size_t r = 0; r < this->h; r++) {
		std::copy(addr(0, r, 0), addr(0, r, 0) + linesize, dest.addr(0, r, 0));
	}
	dest.index = this->index;
}

template <class T> void im::ImagePacked<T>::copyTo(ImageBase<T>& dest, std::vector<int> srcPlanes, std::vector<int> destPlanes) const {
	assert(this->w == dest.w && this->h == dest.h && this->stride <= dest.stride && srcPlanes.size() == destPlanes.size() && "invalid dimensions");
	for (size_t r = 0; r < this->h; r++) {
		for (size_t c = 0; c < this->w; c++) {
			for (int i = 0; i < srcPlanes.size(); i++) {
				dest.at(destPlanes[i], r, c) = this->at(srcPlanes[i], r, c);
			}
		}
	}
	dest.index = this->index;
}


//------------------------------------------------
//explicitly instantiate Image specializations
//------------------------------------------------

template class im::ImageBase<float>;
template class im::ImageBase<unsigned char>;
template class im::ImagePacked<float>;
template class im::ImagePacked<unsigned char>;
