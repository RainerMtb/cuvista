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

#include <QMainWindow>
#include <QMediaPlayer>
#include <vector>

#include "ui_player.h"
#include "MovieWriter.hpp"
#include "ProgressDisplay.hpp"
#include "FrameExecutor.hpp"
#include "PlayerBufferDevice.h"

//player window
class PlayerWindow : public QMainWindow {
    Q_OBJECT

private:
    Ui::playerWindow ui;

public:
    volatile bool isPaused = false;

    PlayerWindow(QWidget* parent);
    QVideoWidget* videoWidget();

    void closeEvent(QCloseEvent* event) override;

signals:
    void cancel();

public slots:
    void progress(QString str, QString status);
    void pause();
    void play();
};

//Image using ffmpeg frame buffer
class ImageYuvFFmpeg : public ImageData<uint8_t> {

private:
    AVFrame* av_frame;

public:
    int64_t index = 0;

    ImageYuvFFmpeg(AVFrame* av_frame = nullptr);

    uint8_t* addr(size_t idx, size_t r, size_t c) override;
    const uint8_t* addr(size_t idx, size_t r, size_t c) const override;
    uint8_t* plane(size_t idx) override;
    const uint8_t* plane(size_t idx) const override;
    int planes() const override;
    int height() const override;
    int width() const override;
    int strideInBytes() const override;
    void setIndex(int64_t frameIndex) override;
    bool saveAsBMP(const std::string& filename, uint8_t scale = 1) const override;
};

//writer
class PlayerWriter : public QObject, public FFmpegWriter {
    Q_OBJECT

private:
    ImageYuvFFmpeg mImageFrame;
    PlayerWindow* mPlayer;
    QImage mImageWorking;
    QMediaPlayer mMediaPlayer;
    int mAudioStreamIndex;
    unsigned char* mBuffer = nullptr;
    AVIOContext* m_av_avio = nullptr;
    PlayerBufferDevice mBufferDevice;

    static int writeBuffer(void* opaque, const uint8_t* buf, int bufsiz);

public:
    PlayerWriter(MainData& data, MovieReader& reader, PlayerWindow* player, QImage imageWorking, int audioStreamCtx);
    ~PlayerWriter();

    void open(EncodingOption videoCodec) override;
    void prepareOutput(FrameExecutor& executor) override;
    void write(const FrameExecutor& executor) override;
    bool startFlushing() override;
    bool flush() override;
};

//progress handler
class PlayerProgress : public ProgressDisplay {

private:
    PlayerWindow* mPlayer;

public:
    PlayerProgress(MainData& data, MovieFrame& frame, PlayerWindow* player) :
        ProgressDisplay(frame, 0),
        mPlayer { player } {}

    void update(bool force = false) override;
};