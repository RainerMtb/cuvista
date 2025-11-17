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
#include <QVideoFrame>
#include <QAudioSink>
#include <QIODevice>
#include <QThread>

#include "ui_player.h"
#include "MovieWriter.hpp"
#include "ProgressDisplay.hpp"
#include "FrameExecutor.hpp"


//player window
class PlayerWindow : public QMainWindow {
    Q_OBJECT

private:
    Ui::playerWindow ui;
    QPixmap mSpeakerOn = QPixmap(":/cuvistaGui/res/09_speaker_on.png");
    QPixmap mSpeakerOff = QPixmap(":/cuvistaGui/res/09_speaker_off.png");

public:
    QAtomicInt isPaused;
    bool hasAudio = false;

    PlayerWindow(QWidget* parent);
    void open(const QVideoFrame& videoFrame, bool hasAudio);
    void closeEvent(QCloseEvent* event) override;
    int getAudioVolume();

signals:
    void cancel();
    void sigProgress(QString str, QString status);
    void sigUpdate(const QVideoFrame& frame);
    void sigLate(bool isLate);
    void sigVolume(int volume);

public slots:
    void progress(QString str, QString status);
    void pause();
};


//writer
class PlayerWriter : public QObject, public NullWriter {
    Q_OBJECT

private:
    PlayerWindow* mPlayer;
    QImage mImageWorking;
    int mAudioStreamIndex;
    bool mPlayAudio;
    QAudioFormat mAudioFormat;
    QVideoFrame mVideoFrame;
    QAudioDevice mAudioDevice;
    QAudioSink* mAudioSink;
    QIODevice* mAudioIODevice;
    std::chrono::time_point<std::chrono::steady_clock> mNextPts;
    std::vector<std::shared_ptr<OutputStreamContext>> outputStreams;

public:
    PlayerWriter(MainData& data, MovieReader& reader, PlayerWindow* player, QImage imageWorking, int audioStreamCtx);

    void open(OutputOption outputOption) override;
    void start() override;
    void writeOutput(const FrameExecutor& executor) override;
    bool flush() override;

public slots:
    void setVolume(int volume);
};


//progress handler
class PlayerProgress : public ProgressDisplay {

private:
    PlayerWindow* mPlayer;
    FrameExecutor& mExecutor;

public:
    PlayerProgress(MainData& data, PlayerWindow* player, FrameExecutor& executor) :
        ProgressDisplay(0),
        mPlayer { player },
        mExecutor { executor } {}

    void update(const ProgressInfo& progress, bool force) override;
};

QImage imageScaledToFit(QImage source, int w, int h);