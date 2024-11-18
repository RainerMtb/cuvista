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
#include <QAudioDevice>
#include <QVideoSink>
#include <QThread>
#include <algorithm>

#include "player.h"
#include "MovieFrame.hpp"
#include "MovieReader.hpp"


//-------------------------------------------------
//------------- Player Window ---------------------
//-------------------------------------------------

PlayerWindow::PlayerWindow(QWidget* parent) :
    QMainWindow(parent) 
{
    ui.setupUi(this);
    setWindowModality(Qt::ApplicationModal);
    connect(ui.btnPause, &QPushButton::clicked, this, &PlayerWindow::pause);
    connect(ui.btnPlay, &QPushButton::clicked, this, &PlayerWindow::play);
    connect(this, &PlayerWindow::sigUpdate, ui.videoWidget->videoSink(), &QVideoSink::videoFrameChanged);
    connect(ui.sliderVolume, &QSlider::valueChanged, this, &PlayerWindow::sigVolume);
}

void PlayerWindow::open(const QVideoFrame& videoFrame) {
    ui.sliderVolume->valueChanged(ui.sliderVolume->value());
    ui.videoWidget->videoSink()->setVideoFrame(videoFrame);
    ui.lblStatus->setText("Buffering...");
}

void PlayerWindow::progress(QString str, QString status) {
    ui.lblFrame->setText(str);
    ui.lblStatus->setText(status);
}

void PlayerWindow::pause() {
    isPaused = true;
    ui.lblStatus->setText("Pausing...");
}

void PlayerWindow::play() {
    isPaused = false;
    ui.lblStatus->setText("Playing...");
}

void PlayerWindow::closeEvent(QCloseEvent* event) {
    cancel();
    isPaused = false;

    //hide only after output is terminated in main window
    event->ignore();
}


//-------------------------------------------------
//------------- Writer Class ----------------------
//-------------------------------------------------

PlayerWriter::PlayerWriter(MainData& data, MovieReader& reader, PlayerWindow* player, QImage imageWorking, int audioStreamIndex) :
    NullWriter(data, reader),
    mPlayer { player },
    mImageWorking { imageWorking },
    mAudioStreamIndex { audioStreamIndex },
    mAudioSink { nullptr },
    mAudioDevice { nullptr } {}

//---- on gui thread
void PlayerWriter::open(EncodingOption videoCodec) {
    //buffering frame
    QImage scaledToFit = mImageWorking.scaled(mData.w, mData.h, Qt::KeepAspectRatio);
    int x = (scaledToFit.width() - mData.w) / 2;
    int y = (scaledToFit.height() - mData.h) / 2;
    QImage image = scaledToFit.copy(x, y, mData.w, mData.h).convertToFormat(QImage::Format_RGBA8888);
    mVideoFrame = QVideoFrame(image);

    //handling input streams
    for (StreamContext& sc : mReader.inputStreams) {
        if (sc.inputStream->index == mReader.videoStream->index) {
            sc.handling = StreamHandling::STREAM_STABILIZE;

        } else if (sc.inputStream->index == mAudioStreamIndex) {
            int sampleRate = mReader.openAudioDecoder(mAudioStreamIndex);
            QAudioFormat fmt;
            fmt.setSampleRate(sampleRate);
            fmt.setChannelCount(2);
            fmt.setSampleFormat(QAudioFormat::Float);
            QAudioDevice device(QMediaDevices::defaultAudioOutput());
            if (device.isFormatSupported(fmt)) {
                mAudioSink = new QAudioSink(fmt, this);
                mAudioDevice = mAudioSink->start();
                sc.handling = StreamHandling::STREAM_DECODE;

                //auto fcnState = [] (QtAudio::State state) { qDebug() << "--- state:" << state; };
                //connect(sink, &QAudioSink::stateChanged, this, fcnState);

                connect(this, &PlayerWriter::sigPlayAudio, this, &PlayerWriter::playAudio);
                connect(mPlayer, &PlayerWindow::sigVolume, this, &PlayerWriter::setVolume);
            }

        } else {
            sc.handling = StreamHandling::STREAM_IGNORE;
        }
    }

    //open player window
    mPlayer->open(mVideoFrame);
    mPlayer->show();
}

//---- on gui thread
void PlayerWriter::playAudio(QByteArray bytes) {
    mAudioDevice->write(bytes);
}

//---- on gui thread
void PlayerWriter::setVolume(int volume) {
    mAudioSink->setVolume(volume / 100.0);
}

//---- on frame executor thread
void PlayerWriter::prepareOutput(FrameExecutor& executor) {
    //cannot reuse QVideoFrame, cannot be mapped more than once ???
    mVideoFrame = QVideoFrame(QVideoFrameFormat(QSize(mData.w, mData.h), QVideoFrameFormat::Format_RGBX8888));
    if (mVideoFrame.map(QVideoFrame::WriteOnly)) {
        int64_t idx = frameIndex;
        ImageRGBA image(mVideoFrame.height(), mVideoFrame.width(), mVideoFrame.bytesPerLine(0), mVideoFrame.bits(0));
        executor.getOutputRgba(idx, image);
        mVideoFrame.unmap();

    } else {
        errorLogger.logError("cannot map video frame");
    }
}

//---- on frame executor thread
void PlayerWriter::write(const FrameExecutor& executor) {
    //presentation time for next frame
    auto t1 = mReader.ptsForFrameAsMillis(frameIndex);
    auto t2 = mReader.ptsForFrameAsMillis(frameIndex + 1);
    int64_t delta = t1.has_value() && t2.has_value() ? (*t2 - *t1) : 0;

    //check time to play video frame
    auto tnow = std::chrono::steady_clock::now();
    while (tnow < mNextPts || mPlayer->isPaused) {
        tnow = std::chrono::steady_clock::now();
    }
    mPlayer->sigUpdate(mVideoFrame);

    //play audio
    if (mAudioStreamIndex != -1 && mAudioSink != nullptr && mAudioDevice != nullptr) {
        StreamContext& sc = mReader.inputStreams[mAudioStreamIndex];
        double videoPts = t1.value_or(0.0) / 1000.0;
        for (auto it = sc.packets.begin(); it != sc.packets.end() && it->pts < videoPts + 0.25; ) {
            QByteArray audioBytes(reinterpret_cast<char*>(it->audioData.data()), it->audioData.size());
            it = sc.packets.erase(it);
            sigPlayAudio(audioBytes);
        }
    }

    //set next presentation time
    mNextPts = tnow + std::chrono::milliseconds(delta);
    frameIndex++;
    frameEncoded++;
}

//---- on frame executor thread
bool PlayerWriter::startFlushing() {
    return true;
}

//---- on frame executor thread
bool PlayerWriter::flush() {
    //wait some time after the last frame is displayed before closing the player
    QThread::msleep(750);
    return false;
}

PlayerWriter::~PlayerWriter() {
    if (mAudioSink != nullptr) {
        mAudioSink->stop();
    }
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

    //send signal to player window
    mPlayer->sigProgress(str, status);
}