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

#include <QtWidgets/QMainWindow>
#include <QString>
#include <QVideoframe>
#include "ui_player.h"

#include "MovieWriter.hpp"
#include "ProgressDisplay.hpp"

//player window
class Player : public QMainWindow {
    Q_OBJECT

public:
    Player(QWidget* parent);

private:
    Ui::playerWindow ui;

public slots:
    void progress(QString str);
    void play(QVideoFrame frame, int64_t idx);

signals:
    void signalProgress(QString str);
    void signalPlay(QVideoFrame frame, int64_t idx);
};

//writer
class PlayerWriter : public NullWriter {

private:
    std::vector<QVideoFrame> mFrameBuffer;
    Player* mPlayer;

public:
    PlayerWriter(MainData& data, MovieReader& reader, Player* player) :
        NullWriter(data, reader),
        mPlayer { player }
    {
        for (int i = 0; i < 8; i++) {
            mFrameBuffer.emplace_back(QVideoFrameFormat(QSize(data.w, data.h), QVideoFrameFormat::PixelFormat::Format_XRGB8888));
        }
    }

    void open(EncodingOption videoCodec) override {}
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