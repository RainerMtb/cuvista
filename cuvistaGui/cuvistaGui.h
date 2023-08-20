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
#include <QFileDialog>

#include "ui_cuvistaGui.h"
#include "progress.h"
#include "MovieFrame.hpp"
#include "FrameResult.hpp"
#include "ProgressDisplayGui.hpp"
#include "StabilizerThread.hpp"

struct EncoderSetting {
    QString text;
    EncodingOption encoder;
};

//main window
class cuvistaGui : public QMainWindow {
    Q_OBJECT

public:
    cuvistaGui(QWidget *parent = nullptr);

public slots:
    void seek(double frac);
    void stabilize();
    void cancelRequest();
    void doneSuccess(const std::string& file, const std::string& str);
    void doneFail(const std::string& str);
    void doneCancel(const std::string& str);
    void showInfo();

private:
    QPixmap mPixmapError = QPixmap(":/cuvistaGui/res/signs-01.png");
    QPixmap mPixmapWorking = QPixmap(":/cuvistaGui/res/signs-02.png");

    Ui::cuvistaGuiClass ui;
    QString mMovieDir;
    QString mInputDir;
    QString mOutputDir;
    QString mOutputFilterSelected;
    QFileInfo mFileInput;
    QFileInfo mFileOutput;

    ProgressWindow* mProgressWindow;
    StabilizerThread* mThread;

    MainData mData;
    InputContext mInputCtx;
    ImageYuv mInputYUV;
    FFmpegReader mReader;

    QColor mBackgroundColor;
    QString mDefaultMessage = QString("select file for input, then click 'stabilize'...");
    QLabel* statusLinkLabel;

    bool mInputReady = false;
    bool mOutputReady = false;

    std::vector<EncoderSetting> encoderSettings;

    void updateInputImage();
    void setInputFile(const QString& filename);
    void showMessage(const QString& msg);
};
