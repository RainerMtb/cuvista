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
#include <QFileDialog>
#include <QThread>
#include <QSettings>

#include "MovieFrame.hpp"
#include "MovieReader.hpp"
#include "FrameResult.hpp"
#include "FrameExecutor.hpp"
#include "UserInputGui.hpp"

#include "ui_cuvistaGui.h"
#include "player.h"
#include "progress.h"

//main window
class cuvistaGui : public QMainWindow {
    Q_OBJECT

public:
    cuvistaGui(QWidget *parent = nullptr);
    ~cuvistaGui();

public slots:
    void seek(double frac);
    void stabilize();
    void done();
    void showInfo();
    void resetGui();
    void showStatusMessage(const std::string& msg);
    void addInputFile(const QString& inputPath);

signals:
    void sigShowStatusMessage(const std::string& str);

private:
    QSettings mSettings = QSettings("RainerMtb", "cuvista");
    QImage mErrorImage = QImage(":/cuvistaGui/res/signs-01.png");
    QImage mWorkingImage = QImage(":/cuvistaGui/res/signs-02.png");
    QPixmap mInputImagePlaceholder = QPixmap(100, 100);

    Ui::cuvistaGuiClass ui;
    QString mMovieDir;
    QString mInputDir;
    QString mOutputDir;
    QString mOutputFilterSelected;
    QFileInfo mFileInput;
    QFileInfo mFileOutput;

    PlayerWindow* mPlayerWindow;
    ProgressWindow* mProgressWindow;
    QThread* mThread;
    std::shared_ptr<MovieWriter> mWriter;
    std::shared_ptr<MovieFrame> mFrame;
    std::shared_ptr<FrameExecutor> mExecutor;
    std::shared_ptr<ProgressBase> mProgress;

    UserInputGui mInputHandler;

    MainData mData;
    ImageYuv mInputYUV;
    ImageBGR mInputBGR;
    QImage mInputImage;
    FFmpegReader mReader;

    QColor mBackgroundColor;

    bool mInputReady = false;
    bool mOutputReady = false;

    void updateInputImage();
    void setInputFile(const QString& inputPath);
    void setBackgroundColor(const QColor& color);
};

class InfoDialog : public QDialog {
    Q_OBJECT

public:
    QThread* worker = nullptr;

    InfoDialog(QWidget* parent) :
        QDialog(parent) {}

    void closeEvent(QCloseEvent* event) override;
 };