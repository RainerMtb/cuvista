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
#include <QMediaDevices>
#include <algorithm>

#include "player.h"
#include "MovieFrame.hpp"
#include "MovieReader.hpp"
#include "Image2.hpp"

//-------------------------------------------------
//------------- Audio Player ----------------------
//-------------------------------------------------

qint64 AudioPlayer::readData(char* data, qint64 maxSize) {
    //read samples tread safe
    std::unique_lock<std::mutex> lock(mMutex);
    int64_t sampleSizes[] = {
        mChunkSize / mBytesPerSample,
        maxSize / mBytesPerSample,
        mSamplesWritten - mSamplesRead,
        mPlayerLimit - mSamplesRead
    };
    int64_t samples = *std::min_element(sampleSizes, sampleSizes + 4);
    int64_t samplesBytes = samples * mBytesPerSample;
    int64_t bufferFreeSize = mBufferLimit - mReadIndex;
    if (samples > bufferFreeSize) {
        int64_t firstPart = bufferFreeSize * mBytesPerSample;
        memcpy(data, mBuffer.data() + mReadIndex * mBytesPerSample, firstPart);
        memcpy(data + firstPart, mBuffer.data(), samplesBytes - firstPart);
        mReadIndex = samples - bufferFreeSize;

    } else {
        memcpy(data, mBuffer.data() + mReadIndex * mBytesPerSample, samplesBytes);
        mReadIndex += samples;
    }
    mSamplesRead += samples;
    qDebug() << "--require " << maxSize << " got " << samplesBytes;
    //for (int i = samplesBytes; i < mChunkSize; i++) data[i] = 0;
    return mChunkSize;
}

qint64 AudioPlayer::writeData(const char* data, qint64 maxSize) {
    return 0;
}

bool AudioPlayer::isSequential() const {
    return true;
}

qint64 AudioPlayer::bytesAvailable() const {
    //return mChunkSize + QIODevice::bytesAvailable();
    return mChunkSize;
}

qint64 AudioPlayer::size() const {
    return mBuffer.size();
}

int AudioPlayer::getSampleRate() const {
    return mStreamCtx->audioInCtx->sample_rate;
}


//-------------------------------------------------
//------------- Player Window ---------------------
//-------------------------------------------------

Player::Player(QWidget* parent) :
    QMainWindow(parent) 
{
    ui.setupUi(this);
    setWindowModality(Qt::ApplicationModal);
    connect(ui.btnPause, &QPushButton::clicked, this, &Player::pause);
    connect(ui.btnPlay, &QPushButton::clicked, this, &Player::play);
}

void Player::open(int h, int w, int stride, QImage imageWorking, StreamContext* scptr, double audioBufferSecs) {
    if (scptr) {
        //init ffmpeg decoder
        audioPlayer.openFFmpeg(scptr, audioBufferSecs);

        //start audio player
        QAudioFormat format;
        format.setSampleFormat(QAudioFormat::Float);
        format.setSampleRate(audioPlayer.getSampleRate());
        format.setChannelConfig(QAudioFormat::ChannelConfigStereo);
        //qDebug() << "audio format: " << format;

        QAudioDevice device = QMediaDevices::defaultAudioOutput();
        if (device.isFormatSupported(format)) {
            audioSink = new QAudioSink(format);
            //auto fcn = [] (QtAudio::State state) { qDebug() << "--state change: " << state; };
            //connect(audioSink, &QAudioSink::stateChanged, this, fcn);

            audioPlayer.open(QIODevice::ReadOnly);
            audioSink->start(&audioPlayer);
        }
    }

    //placeholder frame
    QImage scaledToFit = imageWorking.scaled(w, h, Qt::KeepAspectRatio);
    int x = (scaledToFit.width() - w) / 2;
    int y = (scaledToFit.height() - h) / 2;

    //start video player
    ui.player->open(h, w, stride, scaledToFit.copy(x, y, w, h));
    ui.lblStatus->setText("Buffering...");
}

void Player::decodeAudio() {
    audioPlayer.decodePackets();
}

void Player::setAudioLimit(std::optional<int64_t> millis) {
    audioPlayer.setAudioLimit(millis);
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

void Player::stopAudio() {
    if (audioSink) {
        audioSink->stop();
        delete audioSink;
        audioSink = nullptr;
    }
}

void Player::closeEvent(QCloseEvent* event) {
    cancel();
    isPaused = false;

    //hide only after output is terminated in main window
    event->ignore();
}

//-------------------------------------------------
//------------- Writer Class ----------------------
//-------------------------------------------------

PlayerWriter::PlayerWriter(MainData& data, MovieReader& reader, Player* player, QImage imageWorking, StreamContext* audioStreamCtx) :
    NullWriter(data, reader),
    mPlayer { player },
    mOutput(mData.h, mData.w),
    mNextPts {},
    mImageWorking { imageWorking },
    mAudioStreamCtx { audioStreamCtx } {}

void PlayerWriter::open(EncodingOption videoCodec) {
    mPlayer->show();
    mPlayer->open(mOutput.h, mOutput.w, mOutput.stride, mImageWorking, mAudioStreamCtx, mData.radsec * 8);
    mFrameTimeMillis = mReader.ptsForFrameAsMillis(frameIndex);
}

//load image data from MovieFrame
//this runs on a background thread
void PlayerWriter::prepareOutput(FrameExecutor& executor) {
    int64_t idx = frameIndex;
    executor.getOutput(idx, mOutput);
    mPlayer->sigUpload(idx, mOutput); //upload is done on Main Thread
    mPlayer->decodeAudio();
}

//wait until presentation time has arrived and show video frame
void PlayerWriter::write(const FrameExecutor& executor) {
    //presentation time for next frame
    auto timePoint = mReader.ptsForFrameAsMillis(frameIndex + 1);
    int64_t delta = mFrameTimeMillis.has_value() && timePoint.has_value() ? (*timePoint - *mFrameTimeMillis) : 0;
    std::swap(mFrameTimeMillis, timePoint);

    //check time and player state
    auto tnow = std::chrono::steady_clock::now();
    while (tnow < mNextPts || mPlayer->isPaused) {
        tnow = std::chrono::steady_clock::now();
    }
    mPlayer->playNextFrame(frameIndex);
    mPlayer->setAudioLimit(mFrameTimeMillis);

    //set next presentation time
    mNextPts = tnow + std::chrono::milliseconds(delta);
    frameIndex++;
    frameEncoded++;
}

//close video player
bool PlayerWriter::flush() {
    //wait some time after the last frame is displayed before closing the player
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return false; 
}

bool PlayerWriter::startFlushing() { 
    return true; 
}

PlayerWriter::~PlayerWriter() {
    mPlayer->stopAudio();
    //upload black image data to textures
    mOutput.setValues({ 0, 0, 0, 255 });
    mPlayer->sigUpload(0, mOutput);
    mPlayer->sigUpload(1, mOutput);
    mPlayer->playNextFrame(0);
}


//-------------------------------------------------
//------------- Progress Class --------------------
//-------------------------------------------------

void PlayerProgress::update(bool force) {
    int64_t idx = frame.mWriter.frameIndex - 1;
    auto opstr = frame.mReader.ptsForFrameAsString(idx);

    //frame stats
    QString str = "";
    if (opstr.has_value()) str = QString("%1 (%2)").arg(idx).arg(QString::fromStdString(*opstr));

    //player state
    QString status = "Playing...";
    if (mPlayer->isPaused) status = "Pausing...";
    else if (idx < 0) status = "Buffering...";
    else if (frame.mWriter.frameIndex == frame.mReader.frameIndex) status = "Ending...";

    mPlayer->sigProgress(str, status);
}