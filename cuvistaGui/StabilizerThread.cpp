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
    mData.timeStart();
    std::unique_ptr<MovieWriter> writer;
    std::unique_ptr<MovieFrame> frame;

    try {
        //rewind reader to beginning of input
        mReader.rewind();
        //check input parameters
        mData.validate(mReader);

        //select frame writer
        if (mData.blendInput.enabled)
            writer = std::make_unique<StackedWriter>(mData, mReader);
        else if (mData.requestedEncoding.device == EncodingDevice::NVENC)
            writer = std::make_unique<CudaFFmpegWriter>(mData, mReader);
        else 
            writer = std::make_unique<FFmpegWriter>(mData, mReader);
        //open writer
        writer->open(mData.requestedEncoding);

        //select frame handler class
        DeviceType devtype = mData.deviceList[mData.deviceSelected]->type;
        if (devtype == DeviceType::CPU)
            frame = std::make_unique<CpuFrame>(mData, mReader, *writer);
        else if (devtype == DeviceType::AVX)
            frame = std::make_unique<AvxFrame>(mData, mReader, *writer);
        else if (devtype == DeviceType::CUDA)
            frame = std::make_unique<CudaFrame>(mData, mReader, *writer);
        else if (devtype == DeviceType::OPEN_CL)
            frame = std::make_unique<OpenClFrame>(mData, mReader, *writer);

        //open process handler
        ProgressDisplayGui progress(mData, this, *frame);
        //no secondary writers
        AuxWriters writers;
        //run processing loop
        frame->runLoop(DeshakerPass::COMBINED, progress, inputHandler, writers);

    } catch (const AVException& e) {
        errorLogger.logError(e.what());
    }

    //stopwatch
    double secs = mData.timeElapsedSeconds();
    double fps = writer->frameEncoded / secs;

    //always destruct writer before frame
    writer.reset();
    frame.reset();

    //emit signals to report result back to main thread
    if (errorLogger.hasError())
        failed(errorLogger.getErrorMessage());
    else if (inputHandler.current != UserInputEnum::CONTINUE)
        cancelled("Operation was cancelled");
    else
        succeeded(mData.fileOut, std::format(" written in {:.1f} min at {:.1f} fps", secs / 60.0, fps));
}