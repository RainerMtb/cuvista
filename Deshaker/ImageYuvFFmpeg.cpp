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
#include "Mat.hpp"

ImageYuvFFmpeg::ImageYuvFFmpeg(AVFrame* av_frame) :
    av_frame { av_frame } {}

uint8_t* ImageYuvFFmpeg::addr(size_t idx, size_t r, size_t c) {
    return av_frame->data[idx] + r * av_frame->linesize[idx] + c;
}

const uint8_t* ImageYuvFFmpeg::addr(size_t idx, size_t r, size_t c) const {
    return av_frame->data[idx] + r * av_frame->linesize[idx] + c;
}

uint8_t* ImageYuvFFmpeg::plane(size_t idx) {
    return av_frame->data[idx];
}

const uint8_t* ImageYuvFFmpeg::plane(size_t idx) const {
    return av_frame->data[idx];
}

int ImageYuvFFmpeg::strideInBytes() const {
    assert(av_frame->linesize[0] == av_frame->linesize[1] && av_frame->linesize[0] == av_frame->linesize[2]);
    return av_frame->linesize[0];
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
    Matc y = Matc::fromRowData(h, w, strideInBytes(), plane(0));
    Matc u = Matc::fromRowData(h, w, strideInBytes(), plane(1));
    Matc v = Matc::fromRowData(h, w, strideInBytes(), plane(2));
    return ImageMatYuv8(h, w, w, y.data(), u.data(), v.data()).saveAsBMP(filename, scale);
}


