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
#include <QAtomicInt>
#include "ui_player.h"

#include "MovieWriter.hpp"
#include "ProgressDisplay.hpp"

//player window
class Player : public QMainWindow {
    Q_OBJECT

public:
    QAtomicInt isPaused;

    Player(QWidget* parent);
    void open(int h, int w, int stride, QImage imageWorking);
    void playNextFrame(int64_t idx);

    void closeEvent(QCloseEvent* event) override;

private:
    Ui::playerWindow ui;

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
    std::chrono::time_point<std::chrono::steady_clock> mNextDts;
    QImage mImageWorking;

public:
    PlayerWriter(MainData& data, MovieReader& reader, Player* player, QImage imageWorking);

    void open(EncodingOption videoCodec) override;
    void prepareOutput(MovieFrame& frame) override;
    void write(const MovieFrame& frame) override;
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