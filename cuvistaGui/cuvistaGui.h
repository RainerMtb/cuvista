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

#include "MovieFrame.hpp"
#include "MovieFrame.hpp"
#include "FrameResult.hpp"
#include "UserInputGui.hpp"

#include "ui_cuvistaGui.h"
#include "player.h"
#include "progress.h"

//main window
class cuvistaGui : public QMainWindow {
    Q_OBJECT

public:
    cuvistaGui(QWidget *parent = nullptr);

public slots:
    void seek(double frac);
    void stabilize();
    void done();
    void doneSuccess(const std::string& file, const std::string& str);
    void doneFail(const std::string& str);
    void doneCancel(const std::string& str);
    void showInfo();

private:
    QImage mErrorImage = QImage(":/cuvistaGui/res/signs-01.png");
    QImage mWorkingImage = QImage(":/cuvistaGui/res/signs-02.png");

    Ui::cuvistaGuiClass ui;
    QString mMovieDir;
    QString mInputDir;
    QString mOutputDir;
    QString mOutputFilterSelected;
    QFileInfo mFileInput;
    QFileInfo mFileOutput;

    QThread* mThread = QThread::create([] {});
    Player* mPlayerWindow;
    ProgressWindow* mProgressWindow;
    std::unique_ptr<MovieWriter> mWriter;
    std::unique_ptr<MovieFrame> mFrame;
    std::shared_ptr<ProgressBase> mProgress;

    UserInputGui mInputHandler;

    MainData mData;
    ImageYuv mInputYUV;
    FFmpegReader mReader;

    QColor mBackgroundColor;
    QString mDefaultMessage = QString("select file for input, then click 'stabilize'...");
    QLabel* mStatusLinkLabel;

    bool mInputReady = false;
    bool mOutputReady = false;

    void updateInputImage();
    void setInputFile(const QString& filename);
    void showMessage(const QString& msg);
    void setColorIcon(ClickLabel* btn, QColor& color);
};
