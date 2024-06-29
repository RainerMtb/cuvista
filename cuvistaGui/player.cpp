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
#include <QCloseEvent>

#include "player.h"
#include "MovieFrame.hpp"
#include "Image2.hpp"

 //------------- Player Window ----------------------

Player::Player(QWidget* parent) :
    QMainWindow(parent) 
{
    ui.setupUi(this);
    setWindowModality(Qt::ApplicationModal);
    connect(ui.btnPause, &QPushButton::clicked, this, &Player::pause);
    connect(ui.btnPlay, &QPushButton::clicked, this, &Player::play);
}

void Player::open(int h, int w, int stride, QImage imageWorking) {
    QImage scaledToFit = imageWorking.scaled(w, h, Qt::KeepAspectRatio);
    int x = (scaledToFit.width() - w) / 2;
    int y = (scaledToFit.height() - h) / 2;
    ui.player->open(h, w, stride, scaledToFit.copy(x, y, w, h));
    ui.lblStatus->setText("Buffering...");
}

void Player::upload(int64_t frameIndex, ImageRGBA image) {
    ui.player->upload(frameIndex, image);
}

void Player::playNextFrame(int64_t idx) {
    ui.player->playNextFrame(idx);
}

void Player::progress(QString str, QString status) {
    ui.lblFrame->setText(str);
    ui.lblStatus->setText(status);
}

void Player::pause() {
    isPaused = true;
    ui.lblStatus->setText("Pausing...");
}

void Player::play() {
    isPaused = false;
    ui.lblStatus->setText("Playing...");
}

void Player::closeEvent(QCloseEvent* event) {
    cancel();
    isPaused = false;
    event->ignore(); //hide only after output is terminated in main window
}

//------------- Writer ----------------------

PlayerWriter::PlayerWriter(MainData& data, MovieReader& reader, Player* player, QImage imageWorking) :
    NullWriter(data, reader),
    mPlayer { player },
    mOutput(mData.h, mData.w),
    mNextDts {},
    mImageWorking { imageWorking } {}

void PlayerWriter::open(EncodingOption videoCodec) {
    mPlayer->show();
    mPlayer->open(mOutput.h, mOutput.w, mOutput.stride, mImageWorking);
}

//load image data from MovieFrame into openGL texture
void PlayerWriter::prepareOutput(MovieFrame& frame) {
    int64_t idx = frame.mWriter.frameIndex;
    frame.getOutput(idx, mOutput);
    mPlayer->sigUpload(idx, mOutput);
}

//wait until presentation time has arrived and show video frame
void PlayerWriter::write(const MovieFrame& frame) {
    //check time and player state
    auto tnow = std::chrono::steady_clock::now();
    while (tnow < mNextDts || mPlayer->isPaused) {
        std::this_thread::sleep_for(std::chrono::microseconds(500));
        tnow = std::chrono::steady_clock::now();
    }
    mPlayer->playNextFrame(frameIndex);

    //presentation time for next frame
    auto t1 = mReader.ptsForFrameMillis(frameIndex);
    auto t2 = mReader.ptsForFrameMillis(frameIndex + 1);
    int64_t delta = t1.has_value() && t2.has_value() ? (*t2 - *t1) : 0;
    mNextDts = std::chrono::steady_clock::now() + std::chrono::milliseconds(delta);
    frameIndex++;
    frameEncoded++;
}

//close video player
bool PlayerWriter::flush() {
    //wait some time after the last frame is displayed before closing the player
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    //upload black image data to textures
    mOutput.setValues({ 0, 0, 0, 255 });
    mPlayer->sigUpload(0, mOutput);
    mPlayer->sigUpload(1, mOutput);
    return false; 
}

bool PlayerWriter::startFlushing() { 
    return true; 
}

//------------- Progress ----------------------

void PlayerProgress::update(bool force) {
    int64_t idx = frame.mWriter.frameIndex - 1;
    auto opstr = frame.mReader.ptsForFrameString(idx);

    //frame stats
    QString str = "";
    if (opstr.has_value()) str = QString("%1 (%2)").arg(idx).arg(QString::fromStdString(*opstr));

    //player state
    QString status = "Playing...";
    if (idx < 0) status = "Buffering...";
    if (mPlayer->isPaused) status = "Pausing...";

    mPlayer->sigProgress(str, status);
}