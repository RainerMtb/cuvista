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
#include "ImageHeaders.hpp"
#include "Util.hpp"
#include "CharMap.hpp"

#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <map>

using namespace im;

template <class T> struct NullDeleter {
	void operator () (T* ptr) const {}
};

template <class T> ImageBase<T>::ImageBase(int h, int w, int stride, int numPlanes) :
	h { h },
	w { w },
	stride { stride },
	numPlanes { numPlanes },
	arraySizes { h * stride * numPlanes },
	arrays { std::make_shared<T[]>(arraySizes[0])}
{
	assert(h >= 0 && w >= 0 && "invalid dimensions");
	assert(stride >= w && "stride must be equal or greater to width");
}

template <class T> ImageBase<T>::ImageBase(int h, int w, int stride, int numPlanes, std::vector<std::shared_ptr<T[]>> arrays, std::vector<int> arraySizes) :
	h { h },
	w { w },
	stride { stride },
	numPlanes { numPlanes },
	arraySizes { arraySizes },
	arrays { arrays }
{
	assert(h >= 0 && w >= 0 && "invalid dimensions");
	assert(stride >= w && "stride must be equal or greater to width");
}

template <class T> ImageBase<T>::ImageBase() :
	ImageBase(0, 0, 0, 0) 
{}

template <class T> ImageType ImageBase<T>::type() const {
	return ImageType::BASE;
}

template <class T> void ImageBase<T>::setPixel(size_t row, size_t col, std::vector<T> colors) {
	assert(numPlanes == colors.size() && "image plane count does not match number of color values");
	for (int i = 0; i < numPlanes; i++) {
		at(i, row, col) = colors[i];
	}
}

template <class T> T* ImageBase<T>::addr(size_t idx, size_t r, size_t c) {
	assert(r < h && "row index out of range");
	assert(c < w && "column index out of range");
	assert(idx < numPlanes && "plane index out of range");
	return arrays[0].get() + idx * h * stride + r * stride + c;
}

template <class T> const T* ImageBase<T>::addr(size_t idx, size_t r, size_t c) const {
	assert(r < h && "row index out of range");
	assert(c < w && "column index out of range");
	assert(idx < numPlanes && "plane index out of range");
	return arrays[0].get() + idx * h * stride + r * stride + c;
}

template <class T> T* im::ImageBase<T>::plane(size_t idx) {
	return addr(idx, 0, 0);
}

template <class T> const T* im::ImageBase<T>::plane(size_t idx) const {
	return addr(idx, 0, 0);
}

template <class T> uint64_t ImageBase<T>::crc() const {
	util::CRC64 crc64;
	return crc64.add(*this).result();
}

template <class T> int ImageBase<T>::planes() const {
	return numPlanes;
}

template <class T> int ImageBase<T>::height() const {
	return h;
}

template <class T> int ImageBase<T>::width() const {
	return w;
}

template <class T> int ImageBase<T>::strideInBytes() const {
	return stride * sizeof(T);
}

template <class T> int ImageBase<T>::sizeInBytes() const {
	int siz = 0;
	for (int i : arraySizes) siz += i;
	return siz;
}

template <class T> void ImageBase<T>::setIndex(int64_t frameIndex) {
	index = frameIndex;
}

//access one pixel on plane idx and row / col
template <class T> T& ImageBase<T>::at(size_t idx, size_t r, size_t c) {
	return *addr(idx, r, c);
}

//read access one pixel on plane idx and row / col
template <class T> const T& ImageBase<T>::at(size_t idx, size_t r, size_t c) const {
	return *addr(idx, r, c);
}

template <class T> int ImageBase<T>::colorValue(T pixelValue) const {
	return pixelValue;
}

template <> int ImageBase<float>::colorValue(float pixelValue) const {
	return int(pixelValue * 255.0f);
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

template <class T> LocalColor<T> ImageBase<T>::getLocalColor(const Color& color) const {
	return {};
}

template <class T> void ImageBase<T>::setColorPlane(int plane, T colorValue) {
	for (size_t r = 0; r < h; r++) {
		for (size_t c = 0; c < w; c++) {
			at(plane, r, c) = colorValue;
		}
	}
}

template <class T> void ImageBase<T>::setColor(const Color& color) {
	LocalColor<T> localColor = getLocalColor(color);
	for (int z = 0; z < numPlanes; z++) {
		setColorPlane(z, localColor.colorData[z]);
	}
}

template <class T> void ImageBase<T>::yuvToRgb(ImageData<unsigned char>& dest, const std::vector<int>& planes, ThreadPoolBase& pool) const {
	assert(w == dest.width() && h == dest.height() && numPlanes <= dest.planes() && "dimensions mismatch");
	auto func = [&] (size_t r) {
		for (int c = 0; c < w; c++) {
			yuv_to_rgb(at(0, r, c), at(1, r, c), at(2, r, c), dest.addr(planes[0], r, c), dest.addr(planes[1], r, c), dest.addr(planes[2], r, c));
			if (planes.size() == 4) *dest.addr(planes[3], r, c) = 0xFF;
		}
	};
	pool.addAndWait(func, 0, h);
}

template <class T> void ImageBase<T>::copyTo(ImageData<T>& dest, size_t r0, size_t c0, 
	std::vector<int> srcPlanes, std::vector<int> destPlanes, ThreadPoolBase& pool) const {

	assert(w + c0 <= dest.width() && h + r0 <= dest.height() && planes() == dest.planes() && "dimensions mismatch");
	auto func = [&] (size_t i) {
		for (size_t r = 0; r < h; r++) {
			const T* ptr = addr(srcPlanes[i], r, 0);
			std::copy_n(ptr, w, dest.addr(destPlanes[i], r + r0, c0));
		}
	};
	pool.addAndWait(func, 0, srcPlanes.size());
	dest.setIndex(index);
}

template <class T> void ImageBase<T>::copyTo(ImageData<T>& dest, std::vector<int> srcPlanes, std::vector<int> destPlanes, ThreadPoolBase& pool) const {
	copyTo(dest, 0, 0, srcPlanes, destPlanes, pool);
}

template <class T> void ImageBase<T>::copyTo(ImageData<T>& dest, size_t r0, size_t c0, ThreadPoolBase& pool) const {
	std::vector<int> p(planes());
	std::iota(p.begin(), p.end(), 0);
	copyTo(dest, r0, c0, p, p, pool);
}

template <class T> void ImageBase<T>::copyTo(ImageData<T>& dest, ThreadPoolBase& pool) const {
	copyTo(dest, 0, 0, pool);
}


//------------------------
// write text
//------------------------

template <class T> Size ImageBase<T>::writeText(std::string_view text, int x, int y) {
	int scale = std::min(w, h) / 600 + 1;
	return writeText(text, x, y, scale, scale, TextAlign::BOTTOM_LEFT);
}

template <class T> Size ImageBase<T>::writeText(std::string_view text, int x, int y, int sx, int sy, TextAlign alignment) {
	return writeText(text, x, y, sx, sy, alignment, Color::WHITE, Color::BLACK_SEMI);
}

template <class T> Size ImageBase<T>::writeText(std::string_view text, int x, int y, int sx, int sy,
	TextAlign alignment, const Color& fg, const Color& bg) {

	//compute alignment
	int wt = int(text.size()) * 6 * sx;
	int ht = 10 * sy;
	int align = static_cast<int>(alignment);
	int x0 = x - (align % 3) * wt / 2;
	int y0 = y - (align / 3) * ht / 2;
	
	//fill background area
	for (int ix = x0; ix < x0 + sx + wt; ix++) {
		for (int iy = y0; iy < y0 + ht; iy++) {
			if (iy < h && ix < w) {
				plot(ix, iy, 1.0, bg);
			}
		}
	}

	//write foreground characters
	for (int charIdx = 0; charIdx < text.size(); charIdx++) {
		uint8_t charValue = (uint8_t) text.at(charIdx); //one character from the string
		uint64_t bitmap = charMap[charValue];

		for (int yi = 7; yi >= 0; yi--) {       //row of character
			for (int xi = 4; xi >= 0; xi--) {   //column of character
				//image pixels to set
				int ix = x0 + sx + charIdx * 6 * sx + xi * sx;
				int iy = y0 + sy + yi * sy;
				if (bitmap & 1) {
					for (int scaleY = 0; scaleY < sy; scaleY++) {
						for (int scaleX = 0; scaleX < sx; scaleX++) {
							int xx = ix + scaleX;
							int yy = iy + scaleY;
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

	return { ht, wt };
}


//------------------------
// drawing funtions
//------------------------

template <class T> void ImageBase<T>::plot(int x, int y, double a, const LocalColor<T>& localColor) {
	double alpha = a * localColor.alpha;
	if (alpha > 0.0 && x >= 0 && x < w && y >= 0 && y < h) {
		for (int z = 0; z < numPlanes && z < localColor.colorData.size(); z++) {
			T val = at(z, y, x);
			double pix = val * (1.0 - alpha) + localColor.colorData[z] * alpha;
			at(z, y, x) = (T) pix;
		}
	}
}

template <class T> void ImageBase<T>::plot(double x, double y, double a, const LocalColor<T>& localColor) {
	plot(int(x), int(y), a, localColor);
}

template <class T> void ImageBase<T>::plot(int x, int y, double a, const Color& color) {
	plot(x, y, a, getLocalColor(color));
}

template <class T> void ImageBase<T>::plot(double x, double y, double a, const Color& color) {
	plot(int(x), int(y), a, getLocalColor(color));
}

template <class T> void ImageBase<T>::plot4(double cx, double cy, double dx, double dy, double a, const LocalColor<T>& localColor) {
	plot(cx + dx, cy + dy, a, localColor);
	plot(cx - dx, cy + dy, a, localColor);
	plot(cx + dx, cy - dy, a, localColor);
	plot(cx - dx, cy - dy, a, localColor);
}

template <class T> double ImageBase<T>::fpart(double d) {
	return d - floor(d);
}

template <class T> double ImageBase<T>::rfpart(double d) {
	return 1.0 - fpart(d);
}

template <class T> void ImageBase<T>::drawLine(double x0, double y0, double x1, double y1, const Color& color, double alpha) {
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

	//color information
	LocalColor<T> col = getLocalColor(color);

	if (steep) {
		plot(ypxl1, xpxl1, rfpart(yend) * xgap * alpha, col);
		plot(ypxl1 + 1, xpxl1, fpart(yend) * xgap * alpha, col);

	} else {
		plot(xpxl1, ypxl1, rfpart(yend) * xgap * alpha, col);
		plot(xpxl1, ypxl1 + 1, fpart(yend) * xgap * alpha, col);
	}
	double inter = yend + g;

	//second endpoint
	xend = round(x1);
	yend = y1 + g * (xend - x1);
	xgap = fpart(x1 + 0.5);
	double xpxl2 = xend;
	double ypxl2 = floor(yend);

	if (steep) {
		plot(ypxl2, xpxl2, rfpart(yend) * xgap * alpha, col);
		plot(ypxl2 + 1, xpxl2, fpart(yend) * xgap * alpha, col);

	} else {
		plot(xpxl2, ypxl2, rfpart(yend) * xgap * alpha, col);
		plot(xpxl2, ypxl2 + 1, fpart(yend) * xgap * alpha, col);
	}

	//main loop
	if (steep) {
		for (double x = xpxl1 + 1.0; x < xpxl2; x++) {
			plot(floor(inter), x, rfpart(inter) * alpha, col);
			plot(floor(inter) + 1, x, fpart(inter) * alpha, col);
			inter += g;
		}

	} else {
		for (double x = xpxl1 + 1.0; x < xpxl2; x++) {
			plot(x, floor(inter), rfpart(inter) * alpha, col);
			plot(x, floor(inter) + 1, fpart(inter) * alpha, col);
			inter += g;
		}
	}
}

template <class T> void ImageBase<T>::drawEllipse(double cx, double cy, double rx, double ry, const Color& color, bool fill) {
	double rx2 = util::sqr(rx);
	double ry2 = util::sqr(ry);
	double h = sqrt(rx2 + ry2);
	LocalColor<T> col = getLocalColor(color);

	if (rx < 5 && ry < 5) {
		//std::map<std::set<int>> pixels;

	} else {
		//upper and lower halves
		int quarterX = int(rx2 / h + 0.5);
		for (double x = 0; x <= quarterX; x++) {
			double y = ry * sqrt(1.0 - x * x / rx2);
			double alpha = fpart(y);
			double fly = floor(y);

			plot4(cx, cy, x, fly + 1.0, alpha, col);
			if (fill) {
				for (int i = 0; i <= fly; i++) {
					plot4(cx, cy, x, i, 1.0, col);
				}

			} else {
				plot4(cx, cy, x, fly, 1.0 - alpha, col);
			}
		}

		//right and left halves
		int quarterY = int(ry2 / h + 0.5);
		for (double y = 0; y <= quarterY; y++) {
			double x = rx * sqrt(1.0 - y * y / ry2);
			double alpha = fpart(x);
			double flx = floor(x);

			plot4(cx, cy, floor(x) + 1.0, y, alpha, col);
			if (fill) {
				for (int i = quarterX; i <= flx; i++) {
					plot4(cx, cy, i, y, 1.0, col);
				}

			} else {
				plot4(cx, cy, floor(x), y, 1.0 - alpha, col);
			}
		}
	}
}

template <class T> void ImageBase<T>::drawCircle(double cx, double cy, double r, const Color& color, bool fill) {
	drawEllipse(cx, cy, r, r, color, fill);
}

template <class T> void ImageBase<T>::drawMarker(double cx, double cy, const Color& color, double radius, MarkerType type) {
	drawMarker(cx, cy, color, radius, radius, type);
}

template <class T> void ImageBase<T>::drawMarker(double cx, double cy, const Color& color, double rx, double ry, MarkerType type) {
	using namespace util;

	const int steps = 8;
	constexpr double ds = 1.0 / steps;
	//align center to nearest fraction
	cx = std::round((cx + 0.5) * steps) / steps;
	cy = std::round((cy + 0.5) * steps) / steps;

	//collect subpixels that need to be drawn
	std::map<int, int> alpha;

	//marker types
	std::function<double(double)> fy;
	if (type == MarkerType::BOX) fy = [&] (double x) { return ry; };
	if (type == MarkerType::DIAMOND) fy = [&] (double x) { return ry - ry / rx  * x; };
	if (type == MarkerType::DOT) fy = [&] (double x) { return sqrt(sqr(ry) - sqr(x) * sqr(ry) / sqr(rx)); };

	//collect subpixels
	for (double x = ds / 2; x <= rx; x += ds) {
		for (double y = ds / 2; y <= fy(x); y += ds) {
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

	//color information
	LocalColor col = getLocalColor(color);

	for (auto& [idx, a] : alpha) {
		int iy = idx / stride;
		int ix = idx % stride;
		if (ix >= 0 && ix < w && iy >= 0 && iy < h) {
			plot(ix, iy, a * ds * ds, col);
		}
	}
}

template <class T> T ImageBase<T>::sample(size_t plane, float x, float y, float x0, float x1, float y0, float y1) const {
	float cx = std::clamp(x, x0, x1), cy = std::clamp(y, y0, y1);
	float flx = std::floor(cx), fly = std::floor(cy);
	float dx = cx - flx, dy = cy - fly;
	size_t ix = size_t(flx), iy = size_t(fly);
	size_t xd = dx != 0;
	size_t yd = dy != 0;
	float f00 = at(plane, iy, ix);
	float f01 = at(plane, iy, ix + xd);
	float f10 = at(plane, iy + yd, ix);
	float f11 = at(plane, iy + yd, ix + xd);
	float result = ((1 - dx) * (1 - dy) * f00 + (1 - dx) * dy * f10 + dx * (1 - dy) * f01 + dx * dy * f11);
	return (T) result;
}

template <class T> T ImageBase<T>::sample(size_t plane, float x, float y) const {
	return sample(plane, x, y, 0.0f, w - 1.0f, 0.0f, h - 1.0f);
}

template <class T> T ImageBase<T>::sample(size_t plane, float x, float y, T defaultValue) const {
	return (x < 0.0 || x > w - 1.0 || y < 0.0 || y > h - 1.0) ? defaultValue : sample(plane, x, y);
}

template <class T> std::vector<T> ImageBase<T>::rawBytes() const {
	std::vector<T> data(sizeInBytes());
	T* dest = data.data();
	for (int i : arraySizes) {
		std::copy_n(arrays[i].get(), arraySizes[i], dest);
		dest += arraySizes[i];
	}
	return data;
}

template <class T> bool ImageBase<T>::saveAsBMP(const std::string& filename, T scale) const {
	std::ofstream os(filename, std::ios::binary);
	int h = this->h;
	int w = this->w;

	BmpGrayHeader(w, h * numPlanes).writeHeader(os);
	size_t stridedWidth = util::alignValue(w, 4);
	std::vector<char> data(stridedWidth);

	for (int i = numPlanes - 1; i >= 0; i--) {
		for (int r = h - 1; r >= 0; r--) {
			//prepare one line of data
			for (int c = 0; c < w; c++) data[c] = (unsigned char) std::round(at(i, r, c) * scale);
			//write strided line
			os.write(data.data(), stridedWidth);
		}
	}
	return os.good();
}

template <class T> bool ImageBase<T>::saveAsPGM(const std::string& filename, T scale) const {
	std::ofstream os(filename, std::ios::binary);
	PgmHeader(w, h).writeHeader(os);
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

template <class T> ImagePacked<T>::ImagePacked(int h, int w, int stride, int numPlanes, int arraysize) :
	ImageBase<T>(h, w, stride, numPlanes, { std::make_shared<T[]>(arraysize) }, { arraysize })
{}

template <class T> ImagePacked<T>::ImagePacked(int h, int w, int stride, int numPlanes, T* data, int arraysize) :
	ImageBase<T>(h, w, stride, numPlanes, { {data, NullDeleter<T>()} }, { arraysize })
{}

template <class T> T* ImagePacked<T>::addr(size_t idx, size_t r, size_t c) {
	assert(r < this->h && c < this->w && idx < this->numPlanes && "invalid pixel address");
	return this->data() + r * this->stride + c * this->numPlanes + idx;
}

template <class T> const T* ImagePacked<T>::addr(size_t idx, size_t r, size_t c) const {
	assert(r < this->h && c < this->w && idx < this->numPlanes && "invalid pixel address");
	return this->data() + r * this->stride + c * this->numPlanes + idx;
}

template <class T> T* ImagePacked<T>::data() {
	return this->arrays.at(0).get();
}

template <class T> const T* ImagePacked<T>::data() const {
	return this->arrays.at(0).get();
}

template <class T> std::vector<T> ImagePacked<T>::rawBytes() const {
	int linesiz = this->w * this->numPlanes;
	std::vector<T> data(this->h * linesiz);
	T* dest = data.data();
	const T* src = this->arrays[0].get();
	for (int r = 0; r < this->h; r++) {
		std::copy_n(src, linesiz, dest);
		src += linesiz;
		dest += linesiz;
	}
	return data;
}


//------------------------
// Image from shared Mat
//------------------------

template <class T> ImageMatShared<T>::ImageMatShared(int h, int w, int stride, T* mat) :
	ImageBase<T>(h, w, stride, 1, { { mat, NullDeleter<T>() } }, { h * stride })
{}

template <class T> ImageMatShared<T>::ImageMatShared(int h, int w, int stride, T* y, T* u, T* v) :
	ImageBase<T>(h, w, stride, 3, { { y, NullDeleter<T>() }, { u, NullDeleter<T>() }, { v, NullDeleter<T>() } }, { h * stride, h * stride, h * stride })
{}

template <class T> T* ImageMatShared<T>::addr(size_t idx, size_t r, size_t c) {
	assert(r < this->h && c < this->w && idx < this->numPlanes && "invalid pixel address");
	return this->arrays[idx].get() + r * this->stride + c;
}

template <class T> const T* ImageMatShared<T>::addr(size_t idx, size_t r, size_t c) const {
	assert(r < this->h && c < this->w && idx < this->numPlanes && "invalid pixel address");
	return this->arrays[idx].get() + r * this->stride + c;
}

template <class T> ImageType ImageMatShared<T>::type() const {
	return ImageType::SHARED;
}


//------------------------------------------------
// explicitly instantiate Image specializations
//------------------------------------------------

template class ImageBase<float>;
template class ImageBase<unsigned char>;
template class ImagePacked<float>;
template class ImagePacked<unsigned char>;
template class ImageMatShared<float>;
template class ImageMatShared<unsigned char>;
