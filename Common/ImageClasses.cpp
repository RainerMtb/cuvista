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

#include <iostream>
#include "ImageClasses.hpp"

namespace im {

	ImageStretcher::ImageStretcher(int sourceWidth, int targetWidth, int targetColums) :
		sourceWidth { sourceWidth },
		targetWidth { targetWidth },
		x0(targetColums),
		x1(targetColums),
		f(targetColums)
	{
		constexpr uint32_t maxVal = 1 << 16;
		size_t idx = 0;
		int n = targetColums / targetWidth;

		for (int c = 0; c < targetWidth; c++) {
			float fx = 1.0f * c * sourceWidth / targetWidth;
			float fx0 = std::floor(fx);
			int ix0 = (int) fx0;
			int ix1 = std::min(sourceWidth - 1, ix0 + 1);
			for (int i = 0; i < n; i++) {
				x0[idx] = ix0 * n + i;
				x1[idx] = ix1 * n + i;
				f[idx] = (uint32_t) ((fx - fx0) * maxVal);
				idx++;
			}
		}
	}

	ImageStretcher::ImageStretcher(const Image8& image, int sourceWidth) :
		ImageStretcher(sourceWidth, image.w(), image.cols())
	{}

	ImageStretcher::ImageStretcher() :
		sourceWidth { 0 },
		targetWidth { 0 }
	{}

	ImageStretcher Image8::createStretcher(int sourceWidth) {
		return ImageStretcher(*this, sourceWidth);
	}

	void Image8::stretch(const ImageStretcher& stretcher, ThreadPoolBase& pool) {
		assert(w() == stretcher.targetWidth && cols() == stretcher.f.size() && "invalid parameters");
		if (stretcher.sourceWidth == stretcher.targetWidth) return;
		constexpr uint32_t maxVal = 1 << 16;
		int bufH = (int) pool.size();
		int bufW = cols();
		ImageY<uchar> buffer(bufH, bufW, 255);

		auto fcn = [&] (size_t r) {
			//store one row of pixels in buffer
			uchar* pixelRow = buffer.row(pool.currentThreadIndex());
			uchar* ptr = row(r);
			std::copy_n(ptr, cols(), pixelRow);

			//overwrite pixel row
			for (size_t c = 0; c < cols(); c++) {
				uint32_t x0 = pixelRow[stretcher.x0[c]];
				uint32_t x1 = pixelRow[stretcher.x1[c]];
				uint32_t f = stretcher.f[c];
				uint32_t x = x0 * (maxVal - f) + x1 * f;
				ptr[c] = x / maxVal;
			}
		};
		pool.addAndWait(fcn, 0, h());
	}


	//------------------------------------------------------------------------------

	ImageBgr::ImageBgr(int h, int w) {
		int stride = util::alignValue(w * 3, 4);
		storePtr = std::make_shared<ImageStore<uchar>>(h, w, stride, 3, h * stride, YAxisDir::DOWN, std::vector<int>{ 2, 1, 0 }, 255);
		typePtr = std::make_shared<ImageTypePacked<uchar>>(storePtr);
		colorPtr = std::make_shared<ImageColorRgb<uchar>>(typePtr);
	}

	ImageBgr::ImageBgr() :
		ImageBgr(0, 0)
	{}

	void ImageBgr::saveBmpColor(const std::string& filename) const {
		assert(strideInBytes() % 4 == 0 && "invalid stride");
		std::ofstream os(filename, std::ios::binary);
		BmpColorHeader(w(), h()).writeHeader(os);

		for (int r = h() - 1; r >= 0; r--) {
			os.write(reinterpret_cast<const char*>(addr(0, r, 0)), strideInBytes());
		}
		assert(os.good() && "error writing file");
	}

	ImageBgr ImageBgr::ImageBgr::readBmpFile(std::span<uchar> data, std::span<std::vector<uchar>> customColorMap) {
		ImageBgr image;
		try {
			int fileSize = (int) data.size();

			//analyse header
			if (fileSize < 54) throw std::runtime_error("invalid file");
			if (data[0] != 'B' || data[1] != 'M') throw std::runtime_error("not a bmp file");
			int siz = readBytes(&data[2], 4);
			if (siz != fileSize) throw std::runtime_error("invalid file size");

			int dataOffset = readBytes(&data[10], 4);
			int infoHeaderSize = readBytes(&data[14], 4);
			if (infoHeaderSize != 40) throw std::runtime_error("only BITMAPINFOHEADER is supported");

			int w = readBytes(&data[18], 4);
			int height = readBytes(&data[22], 4);
			int h = std::abs(height);
			int stride = (siz - dataOffset) / h;

			int planes = readBytes(&data[26], 2);
			if (planes != 1) throw std::runtime_error("number of planes must be 1");
			int compression = readBytes(&data[30], 4);
			if (compression != 0) throw std::runtime_error("only uncompressed images are supported");

			//when height is negative, rows are stored from top to bottom
			image = ImageBgr(h, w);
			uchar* dest = image.typePtr->row(h - 1ull);
			int destOffset = -image.stride();
			if (height < 0) {
				dest = image.typePtr->row(0);
				destOffset = image.stride();
			}

			int bits = readBytes(&data[28], 2);
			int colorCount = readBytes(&data[46], 2);
			if (colorCount == 0) colorCount = 1 << bits;

			//copy bytes to Image
			const uchar* src = data.data() + dataOffset;
			if (bits == 1) {
				std::vector<std::vector<uchar>> colorMap = {
					{ data[54], data[55], data[56], data[57] },
					{ data[58], data[59], data[60], data[61] }
				};
				if (customColorMap.size() > 0) {
					colorMap = { customColorMap[0], customColorMap[1] };
				}
				for (int r = 0; r < h; r++) {
					for (int c = 0; c < w; c++) {
						int byteIndex = c / 8;
						int bitIndex = 7 - c % 8;
						int colorIndex = (src[byteIndex] >> bitIndex & 1);
						std::copy_n(colorMap[colorIndex].data(), 3, dest + c * 3);
					}
					src += stride;
					dest += destOffset;
				}

			} else if (bits == 24) {
				for (int r = 0; r < h; r++) {
					std::copy_n(src, 3ull * w, dest);
					src += stride;
					dest += destOffset;
				}

			} else {
				throw std::runtime_error("unsupported bit depth");
			}

		} catch (const std::runtime_error& err) {
			std::cerr << err.what() << std::endl;
		}
		return image;
	}

	ImageBgr ImageBgr::ImageBgr::readBmpFile(const std::string& filename) {

		//read all bytes from file
		std::ifstream is(filename, std::ios::binary);
		std::vector<uchar> data((std::istreambuf_iterator<char>(is)), (std::istreambuf_iterator<char>()));
		is.close();
		return readBmpFile(data);
	}

	uint32_t ImageBgr::readBytes(const uchar* ptr, int byteCount) {
		assert(byteCount <= 4 && "invalid count");
		uint32_t out = 0;
		for (int i = 0; i < byteCount; i++, ptr++) {
			out |= uint8_t(*ptr) << i * 8;
		}
		return out;
	}

	ImageBgr ImageBgr::loadTestImage() {
		ImageBgr im(1080, 1920);
		for (int i = 0; i < 10; i++) {
			int gray = i * 255 / 10;
			Color c = Color::rgb(gray, gray, gray);
			im.fill(i * 108ull, 0, 108, 200, c);
			im.fill(1080 - 108 - i * 108ull, 1720, 108, 200, c);
		}

		im.drawCircle(960, 540, 450, Color::LIGHT_GRAY, true);
		im.drawCircle(960, 540, 250, Color::WHITE, true);

		im.fill(0, 200, 360, 200, Color::RED);
		im.writeText(" R ", 300, 180, TextAlign::MIDDLE_CENTER);
		im.fill(360, 200, 360, 200, Color::GREEN);
		im.writeText(" G ", 300, 540, TextAlign::MIDDLE_CENTER);
		im.fill(720, 200, 360, 200, Color::BLUE);
		im.writeText(" B ", 300, 900, TextAlign::MIDDLE_CENTER);

		im.fill(0, 1520, 360, 200, Color::CYAN);
		im.writeText(" C ", 1620, 180, TextAlign::MIDDLE_CENTER);
		im.fill(360, 1520, 360, 200, Color::MAGENTA);
		im.writeText(" M ", 1620, 540, TextAlign::MIDDLE_CENTER);
		im.fill(720, 1520, 360, 200, Color::YELLOW);
		im.writeText(" Y ", 1620, 900, TextAlign::MIDDLE_CENTER);

		return im;
	}


	//-----------------------------------------------------------------------

	ImageVuyxFloat::ImageVuyxFloat(int h, int w, int stride, float* data) {
		storePtr = std::make_shared<ImageStore<float>>(h, w, stride, 4, h * stride, YAxisDir::DOWN, std::vector<int>{ 2, 1, 0, 3 }, 1.0f, data);
		typePtr = std::make_shared<ImageTypePacked<float>>(storePtr);
		colorPtr = std::make_shared<ImageColorYuv<float>>(typePtr);
	}

	ImageVuyxFloat::ImageVuyxFloat(int h, int w, int stride) {
		storePtr = std::make_shared<ImageStore<float>>(h, w, stride, 4, h * stride, YAxisDir::DOWN, std::vector<int>{ 2, 1, 0, 3 }, 1.0f);
		typePtr = std::make_shared<ImageTypePacked<float>>(storePtr);
		colorPtr = std::make_shared<ImageColorYuv<float>>(typePtr);
	}

	ImageVuyxFloat::ImageVuyxFloat(int h, int w) :
		ImageVuyxFloat(h, w, w * 4) 
	{}

	ImageVuyxFloat::ImageVuyxFloat() :
		ImageVuyxFloat(0, 0)
	{}


	//-----------------------------------------------------------------------

	ImageVuyx::ImageVuyx(int h, int w, int stride) {
		storePtr = std::make_shared<ImageStore<uchar>>(h, w, stride, 4, h * stride, YAxisDir::DOWN, std::vector<int>{ 2, 1, 0, 3 }, 255);
		typePtr = std::make_shared<ImageTypePacked<uchar>>(storePtr);
		colorPtr = std::make_shared<ImageColorYuv<uchar>>(typePtr);
	}

	ImageVuyx::ImageVuyx(int h, int w, size_t stride) :
		ImageVuyx(h, w, (int) stride)
	{}

	ImageVuyx::ImageVuyx(int h, int w) :
		ImageVuyx(h, w, util::alignValue(w * 4, 64))
	{}

	ImageVuyx::ImageVuyx() :
		ImageVuyx(0, 0)
	{}

	ImageVuyx ImageVuyx::readPgmFile(const std::string& filename) {
		ImageYuv yuv = ImageYuv::readPgmFile(filename);
		ImageVuyx vuyx(yuv.h(), yuv.w());
		yuv.convertTo(vuyx);
		return vuyx;
	}

	ImageVuyx ImageVuyx::readBmpFile(const std::string& filename) {
		ImageBgr bgr = ImageBgr::readBmpFile(filename);
		ImageVuyx vuyx(bgr.h(), bgr.w());
		bgr.convertTo(vuyx);
		return vuyx;
	}

	uchar* ImageVuyx::addr(size_t idx, size_t r, size_t c) { 
		return storePtr->data() + r * storePtr->stride + c * planes() + idx;
	}

	const uchar* ImageVuyx::addr(size_t idx, size_t r, size_t c) const {
		return storePtr->data() + r * storePtr->stride + c * planes() + idx;
	}

	uchar& ImageVuyx::at(size_t idx, size_t r, size_t c) { 
		return *addr(idx, r, c); 
	}

	const uchar& ImageVuyx::at(size_t idx, size_t r, size_t c) const { 
		return *addr(idx, r, c); 
	}

	uchar* ImageVuyx::row(size_t r) {
		return storePtr->data() + r * storePtr->stride;
	}

	const uchar* ImageVuyx::row(size_t r) const {
		return storePtr->data() + r * storePtr->stride;
	}


	//-----------------------------------------------------------------------

	ImageYuv::ImageYuv(int h, int w, int stride) {
		storePtr = std::make_shared<ImageStore<uchar>>(h, w, stride, 3, h * stride * 3, YAxisDir::DOWN, std::vector<int>{ 0, 1, 2 }, 255);
		typePtr = std::make_shared<ImageTypePlanar<uchar>>(storePtr);
		colorPtr = std::make_shared<ImageColorYuv<uchar>>(typePtr);
	}

	ImageYuv::ImageYuv(int h, int w, size_t stride) :
		ImageYuv(h, w, (int) stride)
	{}

	ImageYuv::ImageYuv(int h, int w) :
		ImageYuv(h, w, util::alignValue(w, 64))
	{}

	ImageYuv::ImageYuv() :
		ImageYuv(0, 0)
	{}

	ImageYuv ImageYuv::readPgmFile(const std::string& filename) {
		ImageYuv yuv;
		std::ifstream file(filename, std::ios::binary);
		try {
			std::string p5;
			int w, h, maxVal;
			file >> p5 >> w >> h >> maxVal;
			if (p5 != "P5") throw std::runtime_error("file does not start with 'P5'");
			if (maxVal != 255) throw std::runtime_error("max value must be 255");
			file.get(); //read delimiter
			yuv = ImageYuv(h, w);
			file.read(reinterpret_cast<char*>(yuv.data()), 1ull * w * h);

		} catch (const std::exception& e) {
			std::cerr << "error reading from file: " << e.what() << std::endl;

		} catch (...) {
			std::cerr << "error reading from file" << std::endl;
		}
		return yuv;
	}

	ImageYuv ImageYuv::readBmpFile(const std::string& filename) {
		ImageBgr bgr = ImageBgr::readBmpFile(filename);
		ImageYuv yuv(bgr.h(), bgr.w());
		bgr.convertTo(yuv);
		return yuv;
	}

	double ImageYuv::lumaRms() const {
		int64_t sum = 0;
		for (int r = 0; r < h(); r++) {
			const unsigned char* ptr = typePtr->row(r);
			for (int c = 0; c < w(); c++) {
				int64_t val = ptr[c];
				sum += val * val;
			}
		}
		double s = w() * h();
		return std::sqrt(sum / s);
	}

	void ImageYuv::adjustGamma(float g) {
		for (int r = 0; r < h(); r++) {
			unsigned char* ptr = typePtr->row(r);
			for (int c = 0; c < w(); c++) {
				unsigned char& p = ptr[c];
				float x = p / 255.0f;
				p = (unsigned char) std::rint(std::pow(x, g) * 255.0f);
			}
		}
	}

	void ImageYuv::convertTo(Image8& dest, ThreadPoolBase& pool) const {
		int offset = stride() * h();
		if (dest.imageType() == ImageType::VUYX) {
			for (int r = 0; r < h(); r++) {
				uchar* destPtr = dest.row(r);
				for (int c = 0; c < w(); c++) {
					const uchar* srcPtr = addr(0, r, c);
					*destPtr++ = srcPtr[offset * 2];
					*destPtr++ = srcPtr[offset];
					*destPtr++ = srcPtr[0];
					*destPtr++ = 255;
				}
			}
			dest.index = index;

		} else if (dest.imageType() == ImageType::BGRA) {
			for (int r = 0; r < h(); r++) {
				uchar* destPtr = dest.row(r);
				for (int c = 0; c < w(); c++) {
					const uchar* srcPtr = addr(0, r, c);
					yuv_to_rgb(srcPtr[0], srcPtr[offset], srcPtr[offset * 2], destPtr + c * 4 + 2, destPtr + c * 4 + 1, destPtr + c * 4);
					destPtr[c * 4 + 3] = 255;
				}
			}
			dest.index = index;

		} else if (dest.imageType() == ImageType::BGR) {
			for (int r = 0; r < h(); r++) {
				uchar* destPtr = dest.row(r);
				for (int c = 0; c < w(); c++) {
					const uchar* srcPtr = addr(0, r, c);
					yuv_to_rgb(srcPtr[0], srcPtr[offset], srcPtr[offset * 2], destPtr + c * 3 + 2, destPtr + c * 3 + 1, destPtr + c * 3);
				}
			}
			dest.index = index;

		} else {
			ImageBase<uchar>::convertTo(dest, pool);
		}
	}


	//-----------------------------------------------------------------------

	ImageNV12::ImageNV12(int h, int w, int stride) {
		storePtr = std::make_shared<ImageStore<uchar>>(h, w, stride, 3, h * stride * 3 / 2, YAxisDir::DOWN, std::vector<int>{ 0, 1, 2 }, 255);
		typePtr = std::make_shared<ImageTypeNV12<uchar>>(storePtr);
		colorPtr = std::make_shared<ImageColorYuv<uchar>>(typePtr);
	}

	ImageNV12::ImageNV12(int h, int w) :
		ImageNV12(h, w, w)
	{}

	ImageNV12::ImageNV12() :
		ImageNV12(0, 0)
	{}

	Size ImageNV12::writeText(std::string_view text, int x, int y, TextAlign alignment, int sx, int sy, const Color& fg, const Color& bg) {
		ImageYuv image(h(), w());
		convertTo(image);
		Size textSize = image.writeText(text, x, y, alignment, sx, sy, fg, bg);
		image.convertTo(*this);
		return textSize;
	}

	Size ImageNV12::writeText(std::string_view text, int x, int y, TextAlign alignment, int sx, int sy) {
		return writeText(text, x, y, alignment, sx, sy);
	}

	Size ImageNV12::writeText(std::string_view text, int x, int y, TextAlign alignment) {
		return writeText(text, x, y, alignment);
	}


	//-----------------------------------------------------------------------

	ImageBGRA::ImageBGRA(int h, int w, int stride, uchar* data) {
		storePtr = std::make_shared<ImageStore<uchar>>(h, w, stride, 4, h * stride, YAxisDir::DOWN, std::vector<int>{ 2, 1, 0, 3 }, 255, data);
		typePtr = std::make_shared<ImageTypePacked<uchar>>(storePtr);
		colorPtr = std::make_shared<ImageColorRgb<uchar>>(typePtr);
	}

	ImageBGRA::ImageBGRA(int h, int w, int stride) {
		storePtr = std::make_shared<ImageStore<uchar>>(h, w, stride, 4, h * stride, YAxisDir::DOWN, std::vector<int>{ 2, 1, 0, 3 }, 255);
		typePtr = std::make_shared<ImageTypePacked<uchar>>(storePtr);
		colorPtr = std::make_shared<ImageColorRgb<uchar>>(typePtr);
	}

	ImageBGRA::ImageBGRA(int h, int w) :
		ImageBGRA(h, w, util::alignValue(w * 4, 32))
	{}

	ImageBGRA::ImageBGRA() :
		ImageBGRA(0, 0)
	{}

	void ImageBGRA::saveBmpColor(const std::string& filename) const {
		std::ofstream os(filename, std::ios::binary);
		BmpColorHeader(w(), h()).writeHeader(os);
		std::vector<uchar> line(util::alignValue(w() * 3, 4));

		for (int r = h() - 1; r >= 0; r--) {
			uchar* dest = line.data();
			const uchar* src = row(r);
			for (int c = 0; c < w(); c++) {
				*dest++ = *src++;
				*dest++ = *src++;
				*dest++ = *src++;
				src++;
			}
			os.write(reinterpret_cast<char*>(line.data()), line.size());
		}
		assert(os.good() && "error writing file");
	}


	//-----------------------------------------------------------------------

	ImageRGBA::ImageRGBA(int h, int w, int stride, uchar* data) {
		storePtr = std::make_shared<ImageStore<uchar>>(h, w, stride, 4, h * stride, YAxisDir::DOWN, std::vector<int>{ 0, 1, 2, 3 }, 255, data);
		typePtr = std::make_shared<ImageTypePacked<uchar>>(storePtr);
		colorPtr = std::make_shared<ImageColorRgb<uchar>>(typePtr);
	}

	ImageRGBA::ImageRGBA(int h, int w, int stride) {
		storePtr = std::make_shared<ImageStore<uchar>>(h, w, stride, 4, h * stride, YAxisDir::DOWN, std::vector<int>{ 0, 1, 2, 3 }, 255);
		typePtr = std::make_shared<ImageTypePacked<uchar>>(storePtr);
		colorPtr = std::make_shared<ImageColorRgb<uchar>>(typePtr);
	}

	ImageRGBA::ImageRGBA(int h, int w) :
		ImageRGBA(h, w, util::alignValue(w * 4, 32))
	{}

	ImageRGBA::ImageRGBA() :
		ImageRGBA(0, 0)
	{}

}
