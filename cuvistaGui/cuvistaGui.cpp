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

#include "cuvistaGui.h"

#include <QDebug>
#include <QStandardPaths>
#include <QMessageBox>
#include <QColorDialog>
#include <QDesktopServices>

#include "MainData.hpp"
#include "UserInput.hpp"

void setColorIcon(QPushButton* btn, QColor& color) {
    QPixmap icon(btn->iconSize());
    icon.fill(color);
    btn->setIcon(icon);
}

cuvistaGui::cuvistaGui(QWidget *parent) : QMainWindow(parent) {
    ui.setupUi(this);
    version = QString("Version %1").arg(CUVISTA_VERSION.c_str());
    mProgressWindow = new ProgressWindow(this); //destructs when parent destructs
    mMovieDir = QStandardPaths::locate(QStandardPaths::MoviesLocation, QString(), QStandardPaths::LocateDirectory);
    mInputDir = mMovieDir;
    mOutputDir = mMovieDir;
    ui.labelVersion->setText(version);

    mData.console = &nullStream; //suppress any console output
    mData.probeCudaDevices();

    //combo box for encoding settings
    encoderSettings.emplace_back("AUTO", EncodingDevice::AUTO, OutputCodec::AUTO);
    if (mData.deviceProps.size() > 0) {
        encoderSettings.emplace_back("GPU - H265", EncodingDevice::GPU, OutputCodec::H265);
        encoderSettings.emplace_back("GPU - H264", EncodingDevice::GPU, OutputCodec::H264);
    }
    encoderSettings.emplace_back("CPU - H265", EncodingDevice::CPU, OutputCodec::H265);
    encoderSettings.emplace_back("CPU - H264", EncodingDevice::CPU, OutputCodec::H264);

    for (auto& setting : encoderSettings) {
        ui.comboEncoding->addItem(setting.text);
    }

    //devices
    ui.comboDevice->addItem(QString("[CPU] use cpu only - %1 threads").arg(std::thread::hardware_concurrency()));
    for (int i = 0; i < mData.deviceProps.size(); i++) {
        const cudaDeviceProp& prop = mData.deviceProps[i].cudaProps;
        ui.comboDevice->addItem(QString("[GPU %1] %2").arg(i).arg(prop.name));
    }
    ui.comboDevice->setCurrentIndex(ui.comboDevice->count() - 1);

    //file for reading
    auto fcnOpen = [&] () {
        QString str = QFileDialog::getOpenFileName(this, QString("Select Video file to open"), mInputDir, "All Files (*.*)");
        if (!str.isEmpty()) setInputFile(str);
    };
    connect(ui.btnOpen, &QToolButton::clicked, this, fcnOpen);

    //color selection
    const ColorRgb& rgb = mData.bgcol_rgb;
    mBackgroundColor.setRgb(rgb.r(), rgb.g(), rgb.b());
    setColorIcon(ui.btnColor, mBackgroundColor);

    auto fcnColorSelection = [&] () {
        QColor result = QColorDialog().getColor(mBackgroundColor, this, QString("select background color"));
        if (result.isValid()) {
            mBackgroundColor = result;
            setColorIcon(ui.btnColor, mBackgroundColor);
            ui.radioColor->setChecked(true);
        }
    };
    connect(ui.btnColor, &QToolButton::clicked, this, fcnColorSelection);

    //limits
    ui.spinRadius->setMinimum(mData.limits.radsecMin);
    ui.spinRadius->setMaximum(mData.limits.radsecMax);
    ui.spinZoom->setMinimum(mData.limits.imZoomMin);
    ui.spinZoom->setMaximum(mData.limits.imZoomMax);

    //set statusbar
    statusBar()->showMessage(mDefaultMessage);

    //seek input
    connect(ui.imageInput, &ImageLabel::mouseClicked, this, &cuvistaGui::seek);

    //start stabilizing
    connect(ui.btnStart, &QPushButton::clicked, this, &cuvistaGui::stabilize);

    //cancel signal
    connect(mProgressWindow, &ProgressWindow::cancel, this, &cuvistaGui::cancelRequest);

    //info button
    connect(ui.btnInfo, &QPushButton::clicked, this, &cuvistaGui::showInfo);

    //set window to minimal size
    resize(minimumSize());

    //load file from command line argument
    QStringList lst = QCoreApplication::arguments();
    if (lst.size() > 1) {
        setInputFile(lst.at(1));
    }

    //label to show file link in status bar upon success
    statusLinkLabel = new QLabel();
    statusLinkLabel->setTextFormat(Qt::RichText);
    statusLinkLabel->setIndent(5);
    auto fileOpener = [] (const QString& file) {
        QUrl qurl = QUrl::fromLocalFile(file);
        QDesktopServices::openUrl(qurl);
    };
    connect(statusLinkLabel, &QLabel::linkActivated, this, fileOpener);
    statusBar()->addWidget(statusLinkLabel);
}

//open and read new input file
void cuvistaGui::setInputFile(const QString& filePath) {
    try {
        mReader.close();
        errorLogger.clearErrors();
        mInputCtx = mReader.open(filePath.toStdString());
        mInputYUV = ImageYuv(mInputCtx.h, mInputCtx.w);

        Stats status;
        mReader.read(mInputYUV, status); //read first image

        if (errorLogger.hasNoError()) {
            mReader.read(mInputYUV, status); //try to read again for second image
        }

        if (errorLogger.hasNoError() && status.endOfInput == false) {
            mReader.seek(0.1); //try to seek to 10%
        }

        if (errorLogger.hasError()) {
            throw AVException(errorLogger.getErrorMessage());
        }

        mInputReady = true;
        updateInputImage();
        statusBar()->showMessage({});
        StreamInfo info = mInputCtx.videoStreamInfo();
        std::string frameCount = mInputCtx.frameCount == 0 ? "unknown" : std::to_string(mInputCtx.frameCount);
        std::string str = std::format("video: {} x {} px @{:.3f} fps ({}:{})\ncodec: {}, duration: {}, frames: {}",
            mInputCtx.w, mInputCtx.h, mInputCtx.fps(), mInputCtx.fpsNum, mInputCtx.fpsDen, info.codec, info.durationString, frameCount);
        ui.texInput->setPlainText(QString::fromStdString(str));

    } catch (const AVException& ex) {
        ui.imageInput->setImage(mPixmapError);
        ui.statusbar->showMessage(QString(ex.what()));
        ui.texInput->setPlainText({});
        mInputReady = false;
    }
    mFileInput = QFileInfo(filePath);
    mInputDir = mFileInput.path();
    ui.fileOpen->setText(mFileInput.fileName());
    ui.fileOpen->setToolTip(filePath);
}

void cuvistaGui::seek(double frac) {
    if (mInputReady) {
        mReader.seek(frac);
        mReader.read(mInputYUV, mData.status);
        updateInputImage();
    }
}

void cuvistaGui::updateInputImage() {
    if (mInputReady)
        ui.imageInput->setImage(mInputYUV);
}

//-------------------------
// begin stabilization
//-------------------------

void cuvistaGui::stabilize() {
    //check if input is present
    if (mFileInput.fileName().isEmpty()) {
        statusBar()->showMessage(mDefaultMessage);
        return; //nothing to do
    }

    //get output file
    QFileDialog::Options op = ui.chkOverwrite->isChecked() ? QFileDialog::Option::DontConfirmOverwrite : QFileDialog::Options();
    QString fileFilter("MP4 Files (*.mp4);;All Files (*.*)");
    QString outFile = QFileDialog::getSaveFileName(this, QString("Select Video file to save"), mOutputDir, fileFilter, nullptr, op);
    if (outFile.isEmpty()) {
        statusBar()->showMessage(mDefaultMessage);
        return;
    }

    //check if input and output point to same file
    if (mFileInput.canonicalFilePath() == mFileOutput.canonicalFilePath()) {
        QMessageBox::critical(this, QString("Invalid File Selection"), QString("Please select different files\nfor input and output"), QMessageBox::Ok);
        return;
    }

    statusBar()->showMessage(QString("stabilizing..."));
    mFileOutput = QFileInfo(outFile);
    mOutputDir = outFile;

    //set parameters
    mData.inputCtx = mInputCtx;
    mData.fileOut = outFile.toStdString();
    mData.deviceRequested = true;
    mData.deviceNum = ui.comboDevice->currentIndex() - 1;
    EncoderSettings settings = encoderSettings[ui.comboEncoding->currentIndex()];
    mData.encodingDevice = settings.device;
    mData.videoCodec = settings.codec;
    mData.radsec = ui.spinRadius->value();
    mData.imZoom = ui.spinZoom->value();
    mData.bgmode = ui.radioBlend->isChecked() ? BackgroundMode::BLEND : BackgroundMode::COLOR;

    double val = ui.spinBlend->value() / 100.0;
    mData.blendInput.percent = ui.radioBlendLeft->isChecked() ? val : -val;

    using uchar = unsigned char;
    mData.bgcol_rgb = { (uchar) mBackgroundColor.red(), (uchar) mBackgroundColor.green(), (uchar) mBackgroundColor.blue() };

    //set up worker thread
    mThread = new StabilizerThread(mData, mReader);
    connect(mThread, &StabilizerThread::succeeded, this, &cuvistaGui::doneSuccess);
    connect(mThread, &StabilizerThread::failed, this, &cuvistaGui::doneFail);
    connect(mThread, &StabilizerThread::cancelled, this, &cuvistaGui::doneCancel);
    connect(mThread, &StabilizerThread::finished, mThread, &QObject::deleteLater);
    connect(mThread, &StabilizerThread::progress, mProgressWindow, &ProgressWindow::progress);
    connect(mThread, &StabilizerThread::updateInput, mProgressWindow, &ProgressWindow::updateInput);
    connect(mThread, &StabilizerThread::updateOutput, mProgressWindow, &ProgressWindow::updateOutput);

    //begin stabilizing
    mThread->start();
    mProgressWindow->updateInput(mPixmapWorking);
    mProgressWindow->updateOutput(mPixmapWorking);
    mProgressWindow->show();
}

void cuvistaGui::cancelRequest() {
    mThread->requestInterruption();
}

void cuvistaGui::doneSuccess(const std::string& fileString, const std::string& str) {
    mProgressWindow->hide();
    QString file = QString::fromStdString(fileString);
    QString url = QString("file:///") + file;
    QFontMetrics metrics(statusLinkLabel->font());
    QString fileElided = metrics.elidedText(file, Qt::ElideMiddle, 300);
    QString labelText = QString("<a href='%1'>%2</a> %3").arg(url).arg(fileElided).arg(QString::fromStdString(str));
    statusBar()->clearMessage(); //will show the label which was added in the constructor
    statusLinkLabel->setText(labelText);
    statusLinkLabel->show();
}

void cuvistaGui::doneFail(const std::string& str) {
    QString msg = QString::fromStdString(str);
    QMessageBox::critical(this, QString("Error"), msg, QMessageBox::Ok);
    showMessage(QString("Error: %1").arg(msg));
    errorLogger.clearErrors();
}

void cuvistaGui::doneCancel(const std::string& str) {
    showMessage(QString::fromStdString(str));
}

void cuvistaGui::showMessage(const QString& msg) {
    mProgressWindow->hide();
    statusBar()->showMessage(msg);
}

void cuvistaGui::showInfo() {
    int boxHeight = 175;
    int boxWidth = 400;

    QMessageBox msgBox(this);
    msgBox.setWindowTitle(QString("Cuvista Info"));
    msgBox.setIcon(QMessageBox::Information);
    QString author = "Copyright (c) 2023 Rainer Bitschi <a href='mailto:cuvista@a1.net'>Email: cuvista@a1.net</a>";
    QString license = "License GNU GPLv3+: GNU GPL version 3 or later";
    msgBox.setText(QString("CUVISTA - Cuda Video Stabilizer, %1<br>%2<br>%3").arg(version).arg(author).arg(license));
    msgBox.setTextFormat(Qt::RichText);

    std::stringstream ss;
    if (mData.cudaInfo.nvidiaDriverVersion > 0) {
        ss << "Nvidia Driver " << mData.cudaInfo.nvidiaDriver() << std::endl;
        ss << "Cuda Runtime " << mData.cudaInfo.cudaRuntime() << ", Cuda Driver " << mData.cudaInfo.cudaDriver() << std::endl;

    } else {
        ss << "Nvidia Driver Not Found" << std::endl;
    }

    ss << "FFmpeg libavformat " << LIBAVFORMAT_VERSION_MAJOR << "." << LIBAVFORMAT_VERSION_MINOR << "." << LIBAVFORMAT_VERSION_MICRO << std::endl;
    ss << std::endl;
    ss << "List of Cuda Devices:" << std::endl;
    for (int i = 0; i < mData.deviceProps.size(); i++) {
        const cudaDeviceProp& prop = mData.deviceProps[i].cudaProps;
        ss << "Device " << i << ": " << prop.name
            << " // " << prop.totalGlobalMem / 1024 / 1024 << " MB // Compute " << prop.major << "." << prop.minor
            << std::endl;
    }
    if (mData.deviceProps.empty()) {
        ss << "No Applicable Cuda Devices Found" << std::endl;
    }

    QPlainTextEdit* textBox = new QPlainTextEdit(this);
    QString qstr = QString::fromStdString(ss.str()).trimmed();
    textBox->setPlainText(qstr);
    textBox->setMinimumHeight(boxHeight);
    textBox->setReadOnly(true);

    QSpacerItem* spacer = new QSpacerItem(boxWidth, 0, QSizePolicy::Minimum, QSizePolicy::Minimum);
    QGridLayout* layout = (QGridLayout*) msgBox.layout();
    layout->addWidget(textBox, 1, 0, 1, layout->columnCount());
    layout->addItem(spacer, layout->rowCount(), 0, 1, layout->columnCount());
    msgBox.exec();
}
