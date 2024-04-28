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

#include "ProgressDisplay.hpp"
#include "MovieFrame.hpp"
#include <QPixmap>
#include <QDebug>

class StabilizerThread;

//progress handler
class ProgressDisplayGui : public ProgressDisplay {

private:
    StabilizerThread* thread;
    MovieFrame& frame;
    ImagePPM ppmInput;
    ImagePPM ppmOutput;
    std::chrono::steady_clock::time_point timePoint = std::chrono::steady_clock::now();

public:
    ProgressDisplayGui(MainData& data, StabilizerThread* thread, MovieFrame& frame) :
        ProgressDisplay(frame, 50),
        thread { thread },
        frame { frame },
        ppmInput(data.h, data.w),
        ppmOutput(data.h, data.w) {}

    void update(bool force = false) override;
};