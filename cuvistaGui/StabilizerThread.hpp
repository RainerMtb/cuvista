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

#include <QThread>
#include <QPixmap>
#include "UserInputGui.hpp"
#include "MovieFrame.hpp"
#include "CpuFrame.hpp"
#include "OpenClFrame.hpp"
#include "CudaFrame.hpp"
#include "AvxFrame.hpp"

//background thread to work the stabilization
class StabilizerThread : public QThread {
    Q_OBJECT

private:
    MainData& mData;
    MovieReader& mReader;
    UserInputGui inputHandler;

public:
    StabilizerThread(MainData& data, MovieReader& reader) :
        mData { data },
        mReader { reader } {}

    void run() override;

signals:
    void succeeded(const std::string& file, const std::string& msg) const;
    void failed(const std::string& msg) const;
    void cancelled(const std::string& msg) const;
    void progress(bool isFinite, double value) const;
    void updateInput(QPixmap pm, QString time) const;
    void updateOutput(QPixmap pm, QString time) const;
};