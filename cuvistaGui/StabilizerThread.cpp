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

#include "StabilizerThread.hpp"
#include "ProgressDisplayGui.hpp"

void StabilizerThread::run() {
    auto t1 = std::chrono::high_resolution_clock::now();
    std::unique_ptr<MovieWriter> writer;
    std::unique_ptr<MovieFrame> frame;

    try {
        //clear all previous errors
        mData.status.reset();
        //rewind reader to beginning of input
        mReader.rewind();
        //check input parameters
        mData.validate();

        if (mData.deviceNum == -1) {
            frame = std::make_unique<CpuFrame>(mData);
            if (mData.encodingDevice == EncodingDevice::AUTO || mData.encodingDevice == EncodingDevice::CPU)
                writer = std::make_unique<FFmpegWriter>(mData);
            else
                writer = std::make_unique<CudaFFmpegWriter>(mData);

        } else {
            frame = std::make_unique<GpuFrame>(mData);
            if (mData.encodingDevice == EncodingDevice::AUTO || mData.encodingDevice == EncodingDevice::GPU)
                writer = std::make_unique<CudaFFmpegWriter>(mData);
            else
                writer = std::make_unique<FFmpegWriter>(mData);
        }

        writer->open();
        MovieFrame::DeshakerLoopCombined loop;
        ProgressDisplayGui progress(mData, this, frame.get());
        loop.run(*frame, progress, mReader, *writer, inputHandler);

    } catch (const AVException& e) {
        errorLogger.logError(e.what());
    }

    //make sure to destruct writer before frame
    writer.reset();
    frame.reset();

    //stopwatch
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedSec = t2 - t1;
    double fps = mData.status.frameWriteIndex / elapsedSec.count();

    //emit signals to report result back to main thread
    if (errorLogger.hasError())
        failed(errorLogger.getErrorMessage());
    else if (inputHandler.current != UserInputEnum::CONTINUE)
        cancelled("Operation was cancelled");
    else
        succeeded(mData.fileOut, std::format(" written in {:.1f} min at {:.1f} fps", elapsedSec.count() / 60.0, fps));
}