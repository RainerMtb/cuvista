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

#include <QDebug>

#include "player.h"
#include "MovieFrame.hpp"
#include "Image2.hpp"

 //------------- Player Window ----------------------

Player::Player(QWidget* parent) :
    QMainWindow(parent) 
{
    ui.setupUi(this);
}

void Player::open(int h, int w, int stride) {
    ui.player->open(h, w, stride);
}

void Player::upload(int64_t frameIndex, int h, int w, int stride, unsigned char* pixels) {
    ui.player->upload(frameIndex, pixels);
}

void Player::playNextFrame(int64_t idx) {
    ui.player->playNextFrame(idx);
}

void Player::progress(QString str) {
    ui.lblFrame->setText(str);
}

//------------- Writer ----------------------

PlayerWriter::PlayerWriter(MainData& data, MovieReader& reader, Player* player) :
    NullWriter(data, reader),
    mPlayer { player },
    mOutput(mData.h, mData.w),
    mNextDts {} {}

void PlayerWriter::open(EncodingOption videoCodec) {
    mPlayer->show();
    mPlayer->open(mOutput.h, mOutput.w, mOutput.stride);
}

//load image data from MovieFrame into openGL texture
void PlayerWriter::prepareOutput(MovieFrame& frame) {
    int64_t idx = frame.mWriter.frameIndex;
    frame.getOutput(idx, mOutput);
    mPlayer->sigUpload(idx, mOutput.h, mOutput.w, mOutput.stride, mOutput.data());
}

//wait until presentation time has arrived and show video frame
void PlayerWriter::write(const MovieFrame& frame) {
    std::this_thread::sleep_until(mNextDts);
    mPlayer->playNextFrame(frameIndex);

    auto t1 = mReader.dtsForFrameMillis(frameIndex);
    auto t2 = mReader.dtsForFrameMillis(frameIndex + 1);
    int64_t delta = t1.has_value() && t2.has_value() ? (*t2 - *t1) : 0;
    mNextDts = std::chrono::steady_clock::now() + std::chrono::milliseconds(delta);
    frameIndex++;
}

//close video player
bool PlayerWriter::flush() {
    //wait some time after the last frame is displayed before closing the player
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    //upload black image data to textures
    mOutput.setValues({ 0, 0, 0, 255 });
    mPlayer->sigUpload(0, mOutput.h, mOutput.w, mOutput.stride, mOutput.data());
    mPlayer->sigUpload(1, mOutput.h, mOutput.w, mOutput.stride, mOutput.data());
    return false; 
}

bool PlayerWriter::startFlushing() { 
    return true; 
}

//------------- Progress ----------------------

void PlayerProgress::update(bool force) {
    int64_t idx = frame.mWriter.frameIndex - 1;
    auto opstr = frame.mReader.dtsForFrameString(idx);
    QString str = opstr.has_value() ? QString("%1 (%2)").arg(idx).arg(QString::fromStdString(*opstr)) : "";
    mPlayer->sigProgress(str);
}