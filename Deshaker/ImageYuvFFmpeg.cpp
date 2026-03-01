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
#include "ImageClasses.hpp"

ImageYuvFFmpeg::ImageYuvFFmpeg(AVFrame* av_frame) :
    av_frame { av_frame } 
{}

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

size_t ImageYuvFFmpeg::sizeInBytes() const {
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

void ImageYuvFFmpeg::saveBmpPlanes(const std::string& filename) const {
    ImageMatShared<uint8_t> mat(height(), width(), width(), (uint8_t*) addr(0, 0, 0), (uint8_t*) addr(1, 0, 0), (uint8_t*) addr(2, 0, 0), 255);
    return mat.saveBmpPlanes(filename);
}
