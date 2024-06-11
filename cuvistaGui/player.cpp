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
#include <QVideoSink>

#include "player.h"
#include "MovieFrame.hpp"
#include "Image2.hpp"

Player::Player(QWidget* parent) :
    QMainWindow(nullptr) 
{
    ui.setupUi(this);
}

void Player::progress(QString str) {
    ui.lblFrame->setText(str);
}

void Player::play(QVideoFrame frame, int64_t idx) {
    ui.player->videoSink()->setVideoFrame(frame);
}

void PlayerWriter::prepareOutput(MovieFrame& frame) {
    size_t idx = frame.mWriter.frameIndex % mFrameBuffer.size();
    QVideoFrame& qvf = mFrameBuffer[idx];
    qvf.map(QVideoFrame::WriteOnly);
    //ImageARGB argb(qvf.height(), qvf.width(), qvf.bytesPerLine(0), qvf.bits(0));
    //frame.getOutput(idx, argb);
    qvf.unmap();
}

void PlayerWriter::write(const MovieFrame& frame) { 
    size_t idx = frame.mWriter.frameIndex % mFrameBuffer.size();
    QVideoFrame& qvf = mFrameBuffer[idx];
    mPlayer->signalPlay(qvf, frame.mWriter.frameIndex);
    frameIndex++; 
}

bool PlayerWriter::startFlushing() { 
    return false; 
}

bool PlayerWriter::flush() { 
    return false; 
}

void PlayerProgress::update(bool force) {
    uint64_t idx = frame.mWriter.frameIndex;
    QString str = QString("%1 (%2)").arg(idx).arg(QString::fromStdString(frame.getTimeForFrame(idx)));
    mPlayer->signalProgress(str);
}