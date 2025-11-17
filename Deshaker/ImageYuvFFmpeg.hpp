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

#pragma once

#include "FFmpegUtil.hpp"
#include "ImageData.hpp"

//Image using ffmpeg frame buffer
class ImageYuvFFmpeg : public ImageData<uint8_t> {

private:
    AVFrame* av_frame;

public:
    int64_t index = 0;

    ImageYuvFFmpeg(AVFrame* av_frame = nullptr);

    uint8_t* addr(size_t idx, size_t r, size_t c) override;
    const uint8_t* addr(size_t idx, size_t r, size_t c) const override;
    int planes() const override;
    int height() const override;
    int width() const override;
    int strideInBytes() const override;
    int sizeInBytes() const override;

    void setIndex(int64_t frameIndex) override;
    bool saveAsBMP(const std::string& filename, uint8_t scale = 1) const override;
    std::vector<uint8_t> rawBytes() const override;
    ImageType type() const override;
};
