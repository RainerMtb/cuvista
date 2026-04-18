#pragma once

#include "ImageColor.hpp"
#include "CharMap.hpp"

namespace im {

	template <class T> class ImageBase : public IImage<T> {

	protected:
		std::shared_ptr<ImageStoreBase<T>> storePtr;
		std::shared_ptr<ImageTypeBase<T>> typePtr;
		std::shared_ptr<ImageColorBase<T>> colorPtr;

		void plot(int x, int y, double a, const LocalColor<T>& localColor) {
			double alpha = a * localColor.alpha;
			if (x >= 0 && x < w() && y >= 0 && y < h()) {
				for (int i = 0; i < planes() && i < localColor.colorData.size(); i++) {
					int z = colorPtr->colorIndex[i];
					T& val = at(z, y, x);
					double pix = val * (1.0 - alpha) + localColor.colorData[i] * alpha;
					val = (T) pix;
				}
			}
		}

		void plot(double x, double y, double a, const LocalColor<T>& localColor) {
			plot(int(x), int(y), a, localColor);
		}

		void plot(int x, int y, double a, const Color& color) {
			plot(x, y, a, colorPtr->getLocalColor(color));
		}

		void plot(double x, double y, double a, const Color& color) {
			plot(int(x), int(y), a, colorPtr->getLocalColor(color));
		}

		void plot4(double cx, double cy, double dx, double dy, double a, const LocalColor<T>& localColor) {
			plot(cx + dx, cy + dy, a, localColor);
			plot(cx - dx, cy + dy, a, localColor);
			plot(cx + dx, cy - dy, a, localColor);
			plot(cx - dx, cy - dy, a, localColor);
		}

	public:
		virtual T* addr(size_t idx, size_t r, size_t c)               override { return typePtr->addr(idx, r, c); }
		virtual const T* addr(size_t idx, size_t r, size_t c)   const override { return typePtr->addr(idx, r, c); }

		virtual T& at(size_t idx, size_t r, size_t c)                 override { return typePtr->at(idx, r, c); }
		virtual const T& at(size_t idx, size_t r, size_t c)     const override { return typePtr->at(idx, r, c); }

		virtual T* row(size_t r)                        override { return typePtr->row(r); }
		virtual const T* row(size_t r)            const override { return typePtr->row(r); }

		virtual T* plane(size_t idx)                    override { return typePtr->plane(idx); }
		virtual const T* plane(size_t idx)        const override { return typePtr->plane(idx); }

		virtual T* data() { return typePtr->plane(0); }
		virtual const T* data()                            const { return typePtr->plane(0); }

		virtual int h()                           const override { return typePtr->h; }
		virtual int rows()                        const override { return typePtr->rows(); }
		virtual int w()                           const override { return typePtr->w; }
		virtual int cols()                        const override { return typePtr->cols(); }
		virtual int planes()                      const override { return typePtr->planes; }
		virtual int stride()                      const override { return typePtr->stride; }
		virtual int strideInBytes()               const override { return typePtr->stride * sizeof(T); }

		virtual size_t sizeInBytes()              const override { return storePtr->sizeInBytes(); }
		virtual std::vector<T> bytes()            const override { return storePtr->bytes(); }
		virtual uint64_t crc()                    const override { return typePtr->crc().result(); }
		virtual void crc(util::CRC64& base)       const override { return typePtr->crc(base); }
		virtual void write(std::ostream& os)               const { return storePtr->write(os); }

		virtual void gray(ThreadPoolBase& pool = defaultPool)    { return colorPtr->gray(pool); }

		virtual constexpr std::span<int> colorIndex()      const { return colorPtr->colorIndex; }
		virtual constexpr ColorBase colorBase()            const { return colorPtr->colorBase(); }

		virtual ImagePixel<T> pixelAt(size_t r, size_t c)  const { return colorPtr->pixelAt(r, c); }

		virtual void saveBmpPlanes(const std::string& filename) const override { colorPtr->saveBmpPlanes(filename); }
		virtual void saveBmpColor(const std::string& filename) const override;

		virtual void savePgm(const std::string& filename) const override {
			std::ofstream os(filename, std::ios::binary);
			PgmHeader(w(), h()).writeHeader(os);
			std::vector<char> data(w());
			float scale = 255.0f / colorPtr->maxValue;

			for (int z = 0; z < planes(); z++) {
				for (int r = 0; r < h(); r++) {
					for (int c = 0; c < w(); c++) data[c] = (uchar) std::round(at(z, r, c) * scale);
					os.write(data.data(), data.size());
				}
			}
		}

		//fill color value in one plane
		virtual void setColor(int plane, T colorValue) {
			colorPtr->setColor(plane, colorValue);
		}

		//fill color
		virtual void setColor(const Color& color) {
			colorPtr->setColor(color);
		}

		//set color values for one pixel
		virtual void setColor(size_t row, size_t col, const Color& color) {
			colorPtr->setColor(row, col, 1, 1, color);
		}

		//set color values for area pixel
		virtual void setColor(size_t row, size_t col, size_t h, size_t w, const Color& color) {
			colorPtr->setColor(row, col, h, w, color);
		}

		//write text into image
		virtual Size writeText(std::string_view text, int x, int y, int sx, int sy, TextAlign alignment, const Color& fg, const Color& bg) {
			//compute alignment
			int wt = int(text.size()) * 6 * sx;
			int ht = 10 * sy;
			int align = static_cast<int>(alignment);
			int x0 = x - (align % 3) * wt / 2;
			int y0 = y - (align / 3) * ht / 2;

			//fill background area
			for (int ix = x0; ix < x0 + sx + wt; ix++) {
				for (int iy = y0; iy < y0 + ht; iy++) {
					if (iy < h() && ix < w()) {
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
									if (yy < h() && xx < w()) {
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

		//write text into image
		virtual Size writeText(std::string_view text, int x, int y, int sx, int sy, TextAlign alignment) {
			return writeText(text, x, y, sx, sy, alignment, Color::WHITE, Color::BLACK_SEMI);
		}

		//write text into image
		virtual Size writeText(std::string_view text, int x, int y) {
			int scale = std::min(w(), h()) / 600 + 1;
			return writeText(text, x, y, scale, scale, TextAlign::BOTTOM_LEFT);
		}

		virtual void drawLine(double x0, double y0, double x1, double y1, const Color& color, double alpha = 1.0) {
			// Xiaolin Wu's line algorithm
			// https://en.wikipedia.org/wiki/Xiaolin_Wu%27s_line_algorithm
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
			LocalColor<T> col = colorPtr->getLocalColor(color);

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

		virtual void drawEllipse(double cx, double cy, double rx, double ry, const Color& color, bool fill = false) {
			double rx2 = util::sqr(rx);
			double ry2 = util::sqr(ry);
			double h = sqrt(rx2 + ry2);
			LocalColor<T> col = colorPtr->getLocalColor(color);

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

		virtual void drawCircle(double cx, double cy, double r, const Color& color, bool fill = false) {
			drawEllipse(cx, cy, r, r, color, fill);
		}

		virtual void drawMarker(double cx, double cy, const Color& color, double rx, double ry, MarkerType type) {
			const int steps = 8;
			constexpr double ds = 1.0 / steps;
			//align center to nearest fraction
			cx = std::round((cx + 0.5) * steps) / steps;
			cy = std::round((cy + 0.5) * steps) / steps;

			//collect subpixels that need to be drawn
			std::unordered_map<int, int> alpha;

			//marker types
			std::function<double(double)> fy;
			if (type == MarkerType::BOX) fy = [&] (double x) { return ry; };
			if (type == MarkerType::DIAMOND) fy = [&] (double x) { return ry - ry / rx * x; };
			if (type == MarkerType::DOT) fy = [&] (double x) { return sqrt(sqr(ry) - sqr(x) * sqr(ry) / sqr(rx)); };

			//collect subpixels
			for (double x = ds / 2; x <= rx; x += ds) {
				for (double y = ds / 2; y <= fy(x); y += ds) {
					//set subpixels 4 times around center
					for (double px : { cx - x, cx + x}) {
						for (double py : { cy - y, cy + y }) {
							int ix = int(px);
							int iy = int(py);
							int idx = iy * stride() + ix;
							alpha[idx]++;
						}
					}
				}
			}

			//color information
			LocalColor col = colorPtr->getLocalColor(color);

			for (auto& [idx, a] : alpha) {
				int iy = idx / stride();
				int ix = idx % stride();
				if (ix >= 0 && ix < w() && iy >= 0 && iy < h()) {
					plot(ix, iy, a * ds * ds, col);
				}
			}
		}

		virtual void drawMarker(double cx, double cy, const Color& color, double radius = 1.5, MarkerType type = MarkerType::DOT) {
			drawMarker(cx, cy, color, radius, radius, type);
		}

		virtual void copyTo(ImageBase<T>& dest, ThreadPoolBase& pool = defaultPool) const {
			assert(this->imageType() == dest.imageType() && w() <= dest.w() && h() <= dest.h() && "invalid image for copy");
			for (size_t r = 0; r < typePtr->rows(); r++) {
				typePtr->copyRow(r, dest.typePtr);
			}
			dest.setIndex(this->index);
		}

		virtual void writeTo(ImageBase<T>& dest, int y0, int x0, T alpha, ThreadPoolBase& pool = defaultPool) const {
			assert(this->colorBase() == dest.colorBase() && x0 + w() < dest.w() && y0 + h() < dest.h() && "invalid image for copy");
			T mv = colorPtr->maxValue;
			auto fcn = [&] (size_t r) {
				for (size_t c = 0; c < w(); c++) {
					float a = planes() < 4 ? mv : at(colorPtr->colorIndex[3], r, c);
					float srcAlpha = a * alpha / mv;
					float destAlpha = mv - srcAlpha;
					for (int z = 0; z < 3 && z < planes() && z < dest.planes(); z++) {
						float f1 = at(colorPtr->colorIndex[z], r, c) * srcAlpha;
						T* ptr = dest.addr(dest.colorPtr->colorIndex[z], r + y0, c + x0);
						float f2 = *ptr * destAlpha;
						*ptr = (T) ((f1 + f2) / mv);
					}
				}
			};
			pool.addAndWait(fcn, 0, h());
		}

		template <class R> friend class ImageBase;

		template <class R> void convertTo(ImageBase<R>& dest, ThreadPoolBase& pool = defaultPool) const {
			assert(w() <= dest.w() && h() <= dest.h() && "invalid conversion");
			if (this->imageType() == dest.imageType()) {
				colorPtr->convertValuesTo(dest.colorPtr, pool);

			} else if (dest.imageType() == ImageType::NV12) {
				if constexpr (std::is_same_v<uchar, R>) {
					colorPtr->convertToNV12(dest.colorPtr, pool);
				}

			} else if (this->imageType() == ImageType::NV12) {
				if constexpr (std::is_same_v<uchar, T>) {
					colorPtr->convertFromNV12(dest.colorPtr, pool);
				}

			} else {
				colorPtr->convertTo(dest.colorPtr, pool);
			}
			dest.setIndex(this->index);
		}

		//sample clamped to image bounds
		virtual T sample(size_t plane, float x, float y) const {
			float cx = std::clamp(x, 0.0f, w() - 1.0f), cy = std::clamp(y, 0.0f, h() - 1.0f);
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

		//compute sum of squared deltas
		double compareTo(const ImageBase<T>& other) const {
			double result = 0.0;
			if (this->imageType() == other.imageType() && w() == other.w() && h() == other.h()) {
				for (int z = 0; z < planes(); z++) {
					for (int r = 0; r < h(); r++) {
						for (int c = 0; c < w(); c++) {
							result += sqr(at(z, r, c) - other.at(z, r, c));
						}
					}
				}
				result = std::sqrt(result);

			} else {
				result = -1.0;
			}
			return result;
		}

		//equals operator
		virtual bool operator == (const ImageBase<T>& other) const {
			return compareTo(other) == 0.0;
		}

	private:
		double fpart(double d) const {
			return d - floor(d);
		}

		double rfpart(double d) const {
			return 1.0 - fpart(d);
		}

		double sqr(double a) const {
			return a * a;
		}
	};

} //namespace
