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
#include <QString>
#include <QAudioSink>

#include "ui_player.h"
#include "MovieWriter.hpp"
#include "ProgressDisplay.hpp"
#include "FrameExecutor.hpp"
#include "AudioDecoder.hpp"

class AudioPlayer : public AudioDecoder, public QIODevice {

private:
    int mChunkSize = 16384;

public:
    qint64 readData(char* data, qint64 maxSize) override;
    qint64 writeData(const char* data, qint64 maxSize) override;
    bool isSequential() const override;
    qint64 bytesAvailable() const override;
    qint64 size() const override;

    int getSampleRate() const;
};

//player window
class Player : public QMainWindow {
    Q_OBJECT

private:
    Ui::playerWindow ui;
    QAudioSink* audioSink = nullptr;

public:
    volatile bool isPaused = false;

    Player(QWidget* parent);
    void open(int h, int w, int stride, QImage imageWorking, StreamContext* scptr, double audioBufferSecs);
    void playNextFrame(int64_t idx);

    void decodeAudio();
    void setAudioLimit(std::optional<int64_t> millis);
    void stopAudio();

    void closeEvent(QCloseEvent* event) override;

signals:
    void cancel();
    void sigProgress(QString str, QString status);
    void sigUpload(int64_t frameIndex, ImageRGBA image);

public slots:
    void progress(QString str, QString status);
    void upload(int64_t frameIndex, ImageRGBA image);
    void pause();
    void play();
};

//writer
class PlayerWriter : public NullWriter {

private:
    Player* mPlayer;
    ImageRGBA mOutput;
    QImage mImageWorking;
    StreamContext* mAudioStreamCtx;
    std::chrono::time_point<std::chrono::steady_clock> mNextPts;
    std::optional<int64_t> mFrameTimeMillis;
    AudioPlayer mAudioPlayer;

public:
    PlayerWriter(MainData& data, MovieReader& reader, Player* player, QImage imageWorking, StreamContext* audioStreamCtx);
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
    Player* mPlayer;

public:
    PlayerProgress(MainData& data, MovieFrame& frame, Player* player) :
        ProgressDisplay(frame, 0),
        mPlayer { player } {}

    void update(bool force = false) override;
};