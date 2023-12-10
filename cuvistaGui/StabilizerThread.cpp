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
    mData.status.timeStart();
    std::unique_ptr<MovieWriter> writer;
    std::unique_ptr<MovieFrame> frame;

    try {
        //rewind reader to beginning of input
        mReader.rewind();
        //clear all previous errors

        mData.reset();
        //check input parameters
        mData.validate();
        //select frame handler and writer
        if (mData.deviceList[mData.deviceSelected]->type == DeviceType::CPU) {
            frame = std::make_unique<CpuFrame>(mData);
            if (mData.requestedEncoding.device == EncodingDevice::NVENC)
                writer = std::make_unique<CudaFFmpegWriter>(mData);
            else
                writer = std::make_unique<FFmpegWriter>(mData);

        } else if (mData.deviceList[mData.deviceSelected]->type == DeviceType::CUDA) {
            frame = std::make_unique<CudaFrame>(mData);
            if (mData.requestedEncoding.device == EncodingDevice::NVENC)
                writer = std::make_unique<CudaFFmpegWriter>(mData);
            else
                writer = std::make_unique<FFmpegWriter>(mData);

        } else if (mData.deviceList[mData.deviceSelected]->type == DeviceType::OPEN_CL) {
            frame = std::make_unique<OpenClFrame>(mData);
            writer = std::make_unique<FFmpegWriter>(mData);
        }

        writer->open(mData.requestedEncoding);
        ProgressDisplayGui progress(mData, this, frame.get());
        Writers writers;
        frame->runLoop(DeshakerPass::COMBINED, progress, mReader, *writer, inputHandler, writers);

    } catch (const AVException& e) {
        errorLogger.logError(e.what());
    }

    //always destruct writer before frame
    writer.reset();
    frame.reset();

    //stopwatch
    double secs = mData.status.timeElapsedSeconds();
    double fps = mData.status.frameWriteIndex / secs;

    //emit signals to report result back to main thread
    if (errorLogger.hasError())
        failed(errorLogger.getErrorMessage());
    else if (inputHandler.current != UserInputEnum::CONTINUE)
        cancelled("Operation was cancelled");
    else
        succeeded(mData.fileOut, std::format(" written in {:.1f} min at {:.1f} fps", secs / 60.0, fps));
}