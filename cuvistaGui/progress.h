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
#include "ui_progress.h"

#include "ProgressDisplay.hpp"
#include "MainData.hpp"

class ProgressWindow : public QMainWindow {
    Q_OBJECT

private:
    Ui::ProgressWindow ui;

signals:
    void cancel();
    void sigProgress(bool isFinite, double value);
    void sigUpdateInput(QImage im, QString time);
    void sigUpdateOutput(QImage im, QString time);

public slots:
    void progress(bool isFinite, double value);
    void updateInput(QImage pm, QString time);
    void updateOutput(QImage pm, QString time);

public:
    ProgressWindow(QWidget* parent);

    void changeEvent(QEvent* ev) override;
    void closeEvent(QCloseEvent* event) override;
    void showEvent(QShowEvent* event) override;
};


class ProgressGui : public ProgressDisplay {

private:
    ImageRGBA input;
    ImageRGBA output;
    std::chrono::steady_clock::time_point timePoint = std::chrono::steady_clock::now();
    ProgressWindow* progressWindow;

public:
    ProgressGui(MainData& data, MovieFrame& frame, ProgressWindow* progressWindow) :
        ProgressDisplay(frame, 50),
        input(data.h, data.w),
        output(data.h, data.w),
        progressWindow { progressWindow } {}

    void update(bool force = false) override;
};