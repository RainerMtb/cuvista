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
    connect(ui.btnStop, &QPushButton::clicked, this, &PlayerWindow::close);
    connect(this, &PlayerWindow::sigUpdate, ui.videoWidget->videoSink(), &QVideoSink::videoFrameChanged);
    connect(ui.sliderVolume, &QSlider::valueChanged, this, &PlayerWindow::sigVolume);
}

void PlayerWindow::open(const QVideoFrame& videoFrame, bool hasAudio) {
    ui.videoWidget->videoSink()->setVideoFrame(videoFrame);
    ui.lblStatus->setText("Buffering...");
    ui.sliderVolume->setEnabled(hasAudio);
    QPixmap speaker = hasAudio ? mSpeakerOn : mSpeakerOff;
    ui.lblSpeaker->setPixmap(speaker.scaled(18, 18, Qt::KeepAspectRatio, Qt::SmoothTransformation));
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

int PlayerWindow::getAudioVolume() {
    return ui.sliderVolume->value();
}


//-------------------------------------------------
//------------- Writer Class ----------------------
//-------------------------------------------------

PlayerWriter::PlayerWriter(MainData& data, MovieReader& reader, PlayerWindow* player, QImage imageWorking, int audioStreamIndex) :
    NullWriter(data, reader),
    mPlayer { player },
    mImageWorking { imageWorking },
    mAudioStreamIndex { audioStreamIndex },
    mPlayAudio { false },
    mAudioSink { nullptr },
    mAudioIODevice { nullptr } {}

//---- on gui thread
void PlayerWriter::open(EncodingOption videoCodec) {
    //buffering frame
    QImage scaledToFit = mImageWorking.scaled(mData.w, mData.h, Qt::KeepAspectRatio);
    int x = (scaledToFit.width() - mData.w) / 2;
    int y = (scaledToFit.height() - mData.h) / 2;
    QImage image = scaledToFit.copy(x, y, mData.w, mData.h).convertToFormat(QImage::Format_RGBA8888);
    mVideoFrame = QVideoFrame(image);

    //handling input streams
    for (StreamContext& sc : mReader.mInputStreams) {
        auto posc = std::make_shared<OutputStreamContext>();
        posc->inputStream = sc.inputStream;

        if (sc.inputStream->index == mReader.videoStream->index) {
            posc->handling = StreamHandling::STREAM_STABILIZE;

        } else if (sc.inputStream->index == mAudioStreamIndex) {
            int sampleRate = mReader.openAudioDecoder(*posc);
            mAudioFormat.setSampleRate(sampleRate);
            mAudioFormat.setChannelCount(2);
            mAudioFormat.setSampleFormat(QAudioFormat::Float);
            mAudioFormat.setChannelConfig(QAudioFormat::ChannelConfigStereo);
            mAudioDevice = QMediaDevices::defaultAudioOutput();

            // qDebug() << "preferred " << mAudioDevice.preferredFormat();
            // qDebug() << "source format" << mAudioFormat;
            // NOTE: QT IS SUCH AN ABSOLUTE CRAP
            // QAudioDevice does not allow audio formats which the device CLEARLY DOES support
            // seems like ONLY the preferred format is supported
            mPlayAudio = false;
            if (mAudioDevice.isFormatSupported(mAudioFormat)) {
                posc->handling = StreamHandling::STREAM_DECODE;
                connect(mPlayer, &PlayerWindow::sigVolume, this, &PlayerWriter::setVolume);
                mPlayAudio = true;
            }

        } else {
            posc->handling = StreamHandling::STREAM_IGNORE;
        }

        outputStreams.push_back(posc);
        sc.outputStreams.push_back(posc);
    }

    //open player window
    mPlayer->open(mVideoFrame, mPlayAudio);
    mPlayer->show();
}

//---- on gui thread
void PlayerWriter::setVolume(int volume) {
    mAudioSink->setVolume(volume / 100.0);
}

//---- on frame executor thread
void PlayerWriter::start() {
    if (mPlayAudio && mAudioStreamIndex != -1) {
        mAudioSink = new QAudioSink(mAudioDevice, mAudioFormat);
        mAudioSink->setVolume(mPlayer->getAudioVolume() / 100.0);
        mAudioIODevice = mAudioSink->start();
        //mAudioSink->dumpObjectInfo();
    }
}

//---- on frame executor thread
void PlayerWriter::prepareOutput(FrameExecutor& executor) {
    // cannot reuse QVideoFrame, cannot be mapped more than once ???
    // have to create a new QVideoFrame
    // NOTE: MAN QT IS SUCH A CRAP
    mVideoFrame = QVideoFrame(QVideoFrameFormat(QSize(mData.w, mData.h), QVideoFrameFormat::Format_RGBX8888));
    if (mVideoFrame.map(QVideoFrame::WriteOnly)) {
        int64_t idx = frameIndex;
        ImageRGBA image(mVideoFrame.height(), mVideoFrame.width(), mVideoFrame.bytesPerLine(0), mVideoFrame.bits(0));
        executor.getOutputRgba(idx, image);
        mVideoFrame.unmap();

    } else {
        errorLogger().logError("cannot map video frame");
    }
}

//---- on frame executor thread
void PlayerWriter::writeOutput(const FrameExecutor& executor) {
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
    if (mAudioStreamIndex != -1 && mAudioSink != nullptr && mAudioIODevice != nullptr) {
        std::shared_ptr<OutputStreamContext> posc = outputStreams[mAudioStreamIndex];
        std::unique_lock<std::mutex> lock(posc->mMutexSidePackets);
        double videoPts = t1.value_or(0.0) / 1000.0;
        for (auto it = posc->sidePackets.begin(); it != posc->sidePackets.end() && it->pts < videoPts + 0.25; ) {
            mAudioIODevice->write(reinterpret_cast<char*>(it->audioData.data()), it->audioData.size());
            it = posc->sidePackets.erase(it);
        }
    }

    //set next presentation time
    mNextPts = tnow + std::chrono::milliseconds(delta);
    frameIndex++;
}

//---- on frame executor thread
bool PlayerWriter::startFlushing() {
    if (mAudioSink != nullptr) {
        mAudioSink->stop();
        mAudioSink->deleteLater();
    }
    return true;
}

//---- on frame executor thread
bool PlayerWriter::flush() {
    //wait some time after the last frame is displayed before closing the player
    QThread::msleep(750);
    return false;
}

PlayerWriter::~PlayerWriter() {}


//-------------------------------------------------
//------------- Progress Class --------------------
//-------------------------------------------------

void PlayerProgress::update(const ProgressInfo& progress, bool force) {
    int64_t idx = progress.writeIndex - 1;
    auto opstr = mExecutor.mFrame.mReader.ptsForFrameAsString(idx);

    //frame stats
    QString str = "";
    if (opstr.has_value()) str = QString("%1 (%2)").arg(idx).arg(QString::fromStdString(*opstr));

    //player state
    QString status = "Playing...";
    if (mPlayer->isPaused) status = "Pausing...";
    else if (idx < 0) status = "Buffering...";
    else if (progress.writeIndex == progress.readIndex) status = "Ending...";

    //send signal to player window
    mPlayer->sigProgress(str, status);
}