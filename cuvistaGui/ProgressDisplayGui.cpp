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

#include "ProgressDisplayGui.hpp"
#include "StabilizerThread.hpp"

void ProgressDisplayGui::update(bool force) {
    if (isDue(force)) {
        thread->progress(isFinite(), progressPercent());
    }

    auto timePointNow = std::chrono::steady_clock::now();
    std::chrono::nanoseconds delta = timePointNow - timePoint;
    bool imageDue = delta.count() / 1'000'000 > 250;

    if (imageDue && frame.mReader.frameIndex > 0) {
        timePoint = timePointNow;
        uint64_t idx = frame.mReader.frameIndex - 1;
        frame.getInput(idx, ppmInput);
        QPixmap im(ppmInput.w, ppmInput.h);
        im.loadFromData(ppmInput.header(), ppmInput.sizeTotal(), "PPM");
        thread->updateInput(im, QString::fromStdString(frame.getTimeForFrame(idx)));
    }
    if (imageDue && frame.mWriter.frameIndex > 0) {
        timePoint = timePointNow;
        uint64_t idx = frame.mWriter.frameIndex - 1;
        frame.getWarped(idx, ppmOutput);
        QPixmap im(ppmOutput.w, ppmOutput.h);
        im.loadFromData(ppmOutput.header(), ppmInput.sizeTotal(), "PPM");
        thread->updateOutput(im, QString::fromStdString(frame.getTimeForFrame(idx)));
    }
}
