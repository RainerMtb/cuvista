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


#include "OfxUtil.hpp"
#include "ofxBanner.hpp"
#include "ErrorLogger.hpp"
#include "ImageClasses.hpp"
#include "ofxMain.hpp"

namespace ofx {

	OfxImageFloat::OfxImageFloat(int h, int w, int stride, float* data) {
		storePtr = std::make_shared<im::ImageStore<float>>(h, w, stride, 4, h * stride, im::YAxisDir::UP, std::vector<int>{ 0, 1, 2, 3 }, 1.0f, data);
		typePtr = std::make_shared<im::ImageTypePacked<float>>(storePtr);
		colorPtr = std::make_shared<im::ImageColorRgb<float>>(typePtr);
	}

	OfxImageFloat::OfxImageFloat(int h, int w, int stride) {
		storePtr = std::make_shared<im::ImageStore<float>>(h, w, stride, 4, h * stride, im::YAxisDir::UP, std::vector<int>{ 0, 1, 2, 3 }, 1.0f);
		typePtr = std::make_shared<im::ImageTypePacked<float>>(storePtr);
		colorPtr = std::make_shared<im::ImageColorRgb<float>>(typePtr);
	}

	OfxImageFloat::OfxImageFloat(int h, int w) :
		OfxImageFloat(h, w, w * 4)
	{}

	OfxImageFloat::OfxImageFloat() :
		OfxImageFloat(0, 0)
	{}

	void OfxImageFloat::saveBmpColor(const std::string& filename) const {
		std::ofstream os(filename, std::ios::binary);
		im::BmpColorHeader(w(), h()).writeHeader(os);
		std::vector<unsigned char> imageRow(util::alignValue(w() * 3, 4));

		for (int r = 0; r < h(); r++) {
			const float* src = data() + r * stride();
			unsigned char* dest = imageRow.data();
			for (int c = 0; c < w(); c++) {
				*dest++ = (unsigned char) std::clamp(src[2] * 255.0f, 0.0f, 255.0f);
				*dest++ = (unsigned char) std::clamp(src[1] * 255.0f, 0.0f, 255.0f);
				*dest++ = (unsigned char) std::clamp(src[0] * 255.0f, 0.0f, 255.0f);
				src += 4;
			}
			os.write(reinterpret_cast<char*>(imageRow.data()), imageRow.size());
		}
	}

	void OfxImageFloat::copyTo(int y, int x, int h, int w, ImageBase<float>& dest, int destY, int destX, float alpha, ThreadPoolBase& pool) const {
		if (imageType() != dest.imageType()) 
			ImageBase<float>::copyTo(y, x, h, w, dest, destY, destX, alpha, pool);

		assert(x + w <= this->w() && destY + h <= dest.h() && y + h <= this->h() && destX + w <= dest.w() && "invalid parameters for copy");
		for (int r = 0; r < h; r++) {
			const float* srcPtr = row(y + r) + x * 4;
			float* destPtr = dest.row(destY + r) + destX * 4;
			for (int c = 0; c < w; c++) {
				destPtr[0] = destPtr[0] * (1.0f - alpha) + srcPtr[0] * alpha;
				destPtr[1] = destPtr[1] * (1.0f - alpha) + srcPtr[1] * alpha;
				destPtr[2] = destPtr[2] * (1.0f - alpha) + srcPtr[2] * alpha;
				destPtr[3] = 255;
				destPtr += 4;
				srcPtr += 4;
			}
		}
	}

	void  OfxImageFloat::copyTo(int y, int x, int h, int w, ImageBase<float>& dest, int destY, int destX) const {
		if (imageType() != dest.imageType())
			ImageBase<float>::copyTo(y, x, h, w, dest, destY, destX);

		assert(x + w <= this->w() && destY + h <= dest.h() && y + h <= this->h() && destX + w <= dest.w() && "invalid parameters for copy");
		for (int r = 0; r < h; r++) {
			const float* srcPtr = row(y + r) + x * 4;
			float* destPtr = dest.row(destY + r) + destX * 4;
			std::copy_n(srcPtr, w * 4, destPtr);
		}
	}

	void  OfxImageFloat::copyTo(ImageBase<float>& dest, int destY, int destX) const {
		copyTo(0, 0, h(), w(), dest, destY, destX);
	}

	void OfxImageFloat::copyTo(ImageBase<float>& dest) const {
		copyTo(0, 0, h(), w(), dest, 0, 0);
	}


	//----------------------------------------------------------------------------

	OfxImageByte::OfxImageByte(int h, int w, int stride, uint8_t* data) {
		storePtr = std::make_shared<im::ImageStore<uint8_t>>(h, w, stride, 4, h * stride, im::YAxisDir::UP, std::vector<int>{ 0, 1, 2, 3 }, 255, data);
		typePtr = std::make_shared<im::ImageTypePacked<uint8_t>>(storePtr);
		colorPtr = std::make_shared<im::ImageColorRgb<uint8_t>>(typePtr);
	}

	OfxImageByte::OfxImageByte(int h, int w, int stride) {
		storePtr = std::make_shared<im::ImageStore<uint8_t>>(h, w, stride, 4, h * stride, im::YAxisDir::UP, std::vector<int>{ 0, 1, 2, 3 }, 255);
		typePtr = std::make_shared<im::ImageTypePacked<uint8_t>>(storePtr);
		colorPtr = std::make_shared<im::ImageColorRgb<uint8_t>>(typePtr);
	}

	void OfxImageByte::saveBmpColor(const std::string& filename) const {
		std::ofstream os(filename, std::ios::binary);
		im::BmpColorHeader(w(), h()).writeHeader(os);
		std::vector<uint8_t> imageRow(util::alignValue(w() * 3, 4));

		for (int r = 0; r < h(); r++) {
			const uint8_t* src = data() + r * stride();
			uint8_t* dest = imageRow.data();
			for (int c = 0; c < w(); c++) {
				*dest++ = src[2];
				*dest++ = src[1];
				*dest++ = src[0];
				src += 4;
			}
			os.write(reinterpret_cast<char*>(imageRow.data()), imageRow.size());
		}
	}


	//----------------------------------------------------------------------------

	void handleStatus(OfxStatus status, const std::string& message) {
		if (status != kOfxStatOK) {
			std::string str = main.ofxStatsMap.at(status) + ", " + message;
			errorLogger().logError(str, ErrorSource::OFX);
		}
	}


	OfxImageFloat loadBannerElement() {
		int w = 2600;
		int h = 340;
		int stripeSize = 100;

		uchar r = 245;
		uchar g = 200;
		uchar b = 35;

		std::vector<std::vector<uchar>> colorMap = {
			{ b, g, r, 255 },
			{ 0, 0, 0, 255 }
		};
		std::vector<uchar> imageData = util::base64_decode(ofxBannerText);
		ImageBgr bannerText = ImageBgr::readBmpFile(imageData, colorMap);
		ImageBgr bannerImage(h, w);
		bannerImage.setColor(Color::rgb(r, g, b));

		for (int r = 0; r < bannerImage.h(); r++) {
			uchar* ptr = bannerImage.row(r);
			for (int c = 0; c < bannerImage.w(); c++) {
				int flag = (c + r) / stripeSize;
				if (flag & 1) std::fill_n(ptr, 3, 0);
				ptr += 3;
			}
		}

		int y = (bannerImage.h() - bannerText.h()) / 2;
		int x = (bannerImage.w() - bannerText.w()) / 2;
		bannerText.copyTo(bannerImage, y, x);

		OfxImageFloat out(h, w);
		bannerImage.convertTo(out);
		return out;
	}


	OfxImageFloat loadBannerInstance(int targetHeight, int targetWidth, const OfxImageFloat& element) {
		int h = std::min(targetHeight, targetWidth) / 8;
		int w = element.w() * h / element.h();

		OfxImageFloat elementScaled(h, w);
		for (int r = 0; r < h; r++) {
			float y = 1.0f * r / h * element.h();
			float* ptr = elementScaled.row(r);
			for (int c = 0; c < w; c++) {
				float x = 1.0f * c / w * element.w();
				*ptr++ = element.sample(0, x, y);
				*ptr++ = element.sample(1, x, y);
				*ptr++ = element.sample(2, x, y);
				ptr++;
			}
		}

		int n = 1;
		while (n * w < targetWidth * 2) n++;

		OfxImageFloat banner(h, w * n);
		for (int i = 0; i < n; i++) {
			elementScaled.copyTo(banner, 0, i * w);
		}

		banner.setColor(3, 1.0f);
		return banner;
	}

}
