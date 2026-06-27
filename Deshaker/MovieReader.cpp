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

#include "MovieReader.hpp"
#include "ImageClasses.hpp"

using namespace im;


//----------------------------------
//-------- Movie Reader Main -------
//----------------------------------


std::future<void> MovieReader::readAsync(FrameExecutor& executor) {
    return std::async(std::launch::async, [&] { read(executor); });
}

std::optional<int64_t> MovieReader::ptsForFrameAsMillis(int64_t frameIndex) {
    auto fcn = [&] (const VideoPacketContext& vpc) { return vpc.readIndex == frameIndex; };
    std::unique_lock<std::mutex> lock(mVideoPacketMutex);
    auto result = std::find_if(mVideoPacketList.cbegin(), mVideoPacketList.cend(), fcn);
    if (result != mVideoPacketList.end()) {
        return result->bestTimestamp * 1000 * timeBaseNum / timeBaseDen;
    } else {
        return std::nullopt;
    }
}

double MovieReader::ptsForFrame(int64_t frameIndex) {
    auto millis = ptsForFrameAsMillis(frameIndex);
    return millis.has_value() ? (millis.value() / 1000.0) : std::numeric_limits<double>::quiet_NaN();
}

std::optional<std::string> MovieReader::ptsForFrameAsString(int64_t frameIndex) {
    auto millis = ptsForFrameAsMillis(frameIndex);
    if (millis.has_value()) {
        return millisToTimeString(millis.value());
    } else {
        return std::nullopt;
    }
}

std::string MovieReader::videoStreamSummary() const {
    std::string str = frameCount == 0 ? "unknown" : std::to_string(frameCount);
    return std::format("video {} x {} px @{:.3f} fps ({}:{})\nvideo frames: {}\n", w, h, fps(), fpsNum, fpsDen, str);
}


//----------------------------------
//-------- Placeholder Class -------
//----------------------------------


NullReader::NullReader() {
    w = 120;
    h = 120;
    frameCount = 1;
}

bool NullReader::read(Image8& inputFrame) {
    inputFrame.setColor(Color::BLACK);
    frameIndex++;
    endOfInput = false;
    return false;
}
