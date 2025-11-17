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


#include "ImageYuvFFmpeg.hpp"
#include "Image2.hpp"
#include <cassert>

ImageYuvFFmpeg::ImageYuvFFmpeg(AVFrame* av_frame) :
    av_frame { av_frame } {}

uint8_t* ImageYuvFFmpeg::addr(size_t idx, size_t r, size_t c) {
    return av_frame->data[idx] + r * av_frame->linesize[idx] + c;
}

const uint8_t* ImageYuvFFmpeg::addr(size_t idx, size_t r, size_t c) const {
    return av_frame->data[idx] + r * av_frame->linesize[idx] + c;
}

int ImageYuvFFmpeg::strideInBytes() const {
    assert(av_frame->linesize[0] == av_frame->linesize[1] && av_frame->linesize[0] == av_frame->linesize[2]);
    return av_frame->linesize[0];
}

int ImageYuvFFmpeg::sizeInBytes() const {
    return height() * strideInBytes() * planes();
}

int ImageYuvFFmpeg::height() const {
    return av_frame->height;
}

int ImageYuvFFmpeg::width() const {
    return av_frame->width;
}

int ImageYuvFFmpeg::planes() const {
    return 3;
}

void ImageYuvFFmpeg::setIndex(int64_t frameIndex) {
    index = frameIndex;
}

bool ImageYuvFFmpeg::saveAsBMP(const std::string& filename, uint8_t scale) const {
    int h = height();
    int w = width();
    ImageMatYuv8 mat(h, w, w, (uint8_t*) addr(0, 0, 0), (uint8_t*) addr(1, 0, 0), (uint8_t*) addr(2, 0, 0));
    return mat.saveAsBMP(filename, scale);
}

std::vector<uint8_t> ImageYuvFFmpeg::rawBytes() const {
    int planeSize = strideInBytes() * height();
    std::vector<uint8_t> data(3ll * planeSize);
    std::copy_n(addr(0, 0, 0), planeSize, data.data());
    std::copy_n(addr(1, 0, 0), planeSize, data.data() + planeSize);
    std::copy_n(addr(2, 0, 0), planeSize, data.data() + planeSize * 2);
    return data;
}

ImageType ImageYuvFFmpeg::type() const {
    return ImageType::SHARED;
}