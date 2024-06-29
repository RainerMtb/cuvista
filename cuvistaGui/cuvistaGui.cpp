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

#include "UserInputGui.hpp"
#include "MovieFrame.hpp"
#include "CpuFrame.hpp"
#include "OpenClFrame.hpp"
#include "CudaFrame.hpp"
#include "AvxFrame.hpp"
#include "progress.h"

cuvistaGui::cuvistaGui(QWidget *parent) : 
    QMainWindow(parent) 
{
    ui.setupUi(this);
    mPlayerWindow = new Player(this);
    mProgressWindow = new ProgressWindow(this);
    mMovieDir = QStandardPaths::locate(QStandardPaths::MoviesLocation, QString(), QStandardPaths::LocateDirectory);
    mInputDir = mMovieDir;
    mOutputDir = mMovieDir;
    QString version = QString("Version ") + QString::fromStdString(CUVISTA_VERSION);
    ui.labelVersion->setText(version);

    mData.console = &mData.nullStream;
    mData.probeCuda();
    mData.probeOpenCl();
    mData.collectDeviceInfo(); 

    //devices
    for (int i = 0; i < mData.deviceList.size(); i++) {
        ui.comboDevice->addItem(QString::fromStdString(mData.deviceList[i]->getName()));
    }

    //combo box for encoding options for selected device
    auto fcnEncoding = [&] (int index) {
        ui.comboEncoding->clear();

        std::vector<EncodingOption>& options = mData.deviceList[index]->encodingOptions;
        for (EncodingOption e : options) {
            QVariant qv = QVariant::fromValue(e);
            std::string str = std::format("{} - {}", mData.mapDeviceToString[e.device], mData.mapCodecToString[e.codec]);
            QString qs = QString::fromStdString(str);

            ui.comboEncoding->addItem(qs, qv);
        }
    };
    //set encoding options when device changes
    connect(ui.comboDevice, &QComboBox::currentIndexChanged, this, fcnEncoding);
    //trigger setting encoding options
    ui.comboDevice->setCurrentIndex(ui.comboDevice->count() - 1);

    //file for reading
    auto fcnOpen = [&] () {
        QString str = QFileDialog::getOpenFileName(this, QString("Select Video file to open"), mInputDir, "All Files (*.*)");
        if (!str.isEmpty()) setInputFile(str);
    };
    connect(ui.btnOpen, &QToolButton::clicked, this, fcnOpen);

    //color selection
    const im::ColorRgb& rgb = mData.bgcol_rgb;
    mBackgroundColor.setRgb(rgb.r(), rgb.g(), rgb.b());
    setColorIcon(ui.lblColor, mBackgroundColor);

    auto fcnColorSelection = [&] () {
        QColorDialog::ColorDialogOptions options = QColorDialog::ColorDialogOption::NoEyeDropperButton;
        QColor result = QColorDialog::getColor(mBackgroundColor, this, QString("Select Background Color"), options);
        if (result.isValid()) {
            mBackgroundColor = result;
            setColorIcon(ui.lblColor, mBackgroundColor);
            ui.radioColor->setChecked(true);
        }
    };
    connect(ui.lblColor, &ClickLabel::clicked, this, fcnColorSelection);

    //image sequences
    ui.comboImageType->addItem("BMP", QVariant::fromValue(OutputType::SEQUENCE_BMP));
    ui.comboImageType->addItem("JPG", QVariant::fromValue(OutputType::SEQUENCE_JPG));

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

    //info button
    connect(ui.btnInfo, &QPushButton::clicked, this, &cuvistaGui::showInfo);

    //progress handler
    connect(mProgressWindow, &ProgressWindow::cancel, &mInputHandler, &UserInputGui::cancel);
    connect(mProgressWindow, &ProgressWindow::sigProgress, mProgressWindow, &ProgressWindow::progress);
    connect(mProgressWindow, &ProgressWindow::sigUpdateInput, mProgressWindow, &ProgressWindow::updateInput);
    connect(mProgressWindow, &ProgressWindow::sigUpdateOutput, mProgressWindow, &ProgressWindow::updateOutput);

    //player window
    connect(mPlayerWindow, &Player::sigProgress, mPlayerWindow, &Player::progress);
    connect(mPlayerWindow, &Player::sigUpload, mPlayerWindow, &Player::upload);
    connect(mPlayerWindow, &Player::cancel, &mInputHandler, &UserInputGui::cancel);

    //set window to minimal size
    resize(minimumSize());

    //load files from command line argument
    //preset input file
    QStringList cmdArgs = QCoreApplication::arguments();
    if (cmdArgs.size() > 1) {
        QString inputFile = cmdArgs.at(1);
        if (QFileInfo(inputFile).exists()) {
            setInputFile(inputFile);
        }
    }
    //preset output file
    if (cmdArgs.size() > 2) {
        mOutputDir = cmdArgs.at(2);
        mOutputFilterSelected = "All Files (*.*)";
    }

    //label to show file link in status bar upon success
    mStatusLinkLabel = new QLabel();
    mStatusLinkLabel->setTextFormat(Qt::RichText);
    mStatusLinkLabel->setIndent(5);
    auto fileOpener = [] (const QString& file) {
        QUrl qurl = QUrl::fromLocalFile(file);
        QDesktopServices::openUrl(qurl);
    };
    connect(mStatusLinkLabel, &QLabel::linkActivated, this, fileOpener);
    statusBar()->addWidget(mStatusLinkLabel);
}

//open and read new input file
void cuvistaGui::setInputFile(const QString& filePath) {
    try {
        mReader.close();
        errorLogger.clearErrors();
        mReader.open(filePath.toStdString());
        mInputYUV = ImageYuv(mReader.h, mReader.w);

        mReader.read(mInputYUV); //read first image

        if (errorLogger.hasNoError()) {
            mReader.read(mInputYUV); //try to read again for second image
        }

        if (errorLogger.hasNoError() && mReader.endOfInput == false) {
            mReader.seek(0.1); //try to seek to 10%
        }

        if (errorLogger.hasError()) {
            throw AVException(errorLogger.getErrorMessage());
        }

        mInputReady = true;
        updateInputImage();
        statusBar()->showMessage({});
        StreamInfo info = mReader.videoStreamInfo();
        std::string frameCount = mReader.frameCount == 0 ? "unknown" : std::to_string(mReader.frameCount);
        std::string str = std::format("video: {} x {} px @{:.3f} fps ({}:{})\ncodec: {}, duration: {}, frames: {}",
            mReader.w, mReader.h, mReader.fps(), mReader.fpsNum, mReader.fpsDen, info.codec, info.durationString, frameCount);
        ui.texInput->setPlainText(QString::fromStdString(str));

    } catch (const AVException& ex) {
        ui.imageInput->setImage(mErrorImage);
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
        mReader.read(mInputYUV);
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

    //get output video file
    QString outFile;
    if (ui.chkEncode->isChecked() || ui.chkStack->isChecked()) {
        QFileDialog::Options op = ui.chkOverwrite->isChecked() ? QFileDialog::Option::DontConfirmOverwrite : QFileDialog::Options();
        QString fileFilter("Video Files (*.mp4; *.mkv);;All Files (*.*)");
        outFile = QFileDialog::getSaveFileName(this, QString("Select Video file to save"), mOutputDir, fileFilter, &mOutputFilterSelected, op);
        if (outFile.isEmpty()) {
            statusBar()->showMessage(mDefaultMessage);
            return;
        }
    }

    //get output sequence folder
    if (ui.chkSequence->isChecked()) {
        outFile = QFileDialog::getExistingDirectory(this, QString("Select Output Folder"), mOutputDir);
        if (outFile.isEmpty()) {
            statusBar()->showMessage(mDefaultMessage);
            return;
        }
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
    mData.fileOut = outFile.toStdString();
    mData.deviceRequested = true;
    mData.deviceSelected = ui.comboDevice->currentIndex();
    mData.requestedEncoding = ui.comboEncoding->currentData().value<EncodingOption>();
    mData.radsec = ui.spinRadius->value();
    mData.imZoom = ui.spinZoom->value();
    mData.bgmode = ui.radioBlend->isChecked() ? BackgroundMode::BLEND : BackgroundMode::COLOR;

    using uchar = unsigned char;
    mData.bgcol_rgb = { (uchar) mBackgroundColor.red(), (uchar) mBackgroundColor.green(), (uchar) mBackgroundColor.blue() };

    //rewind reader to beginning of input
    mReader.rewind();
    //check input parameters
    mData.validate(mReader);
    //reset input handler
    mInputHandler.reset();

    //select writer
    if (ui.chkStack->isChecked())
        mWriter = std::make_unique<StackedWriter>(mData, mReader, ui.slideStack->value() / 100.0);
    else if (ui.chkSequence->isChecked() && ui.comboImageType->currentData().value<OutputType>() == OutputType::SEQUENCE_BMP)
        mWriter = std::make_unique<BmpImageWriter>(mData, mReader);
    else if (ui.chkSequence->isChecked() && ui.comboImageType->currentData().value<OutputType>() == OutputType::SEQUENCE_JPG)
        mWriter = std::make_unique<JpegImageWriter>(mData, mReader);
    else if (ui.chkPlayer->isChecked())
        mWriter = std::make_unique<PlayerWriter>(mData, mReader, mPlayerWindow, mWorkingImage);
    else if (ui.chkEncode->isChecked() && mData.requestedEncoding.device == EncodingDevice::NVENC)
        mWriter = std::make_unique<CudaFFmpegWriter>(mData, mReader);
    else if (ui.chkEncode->isChecked())
        mWriter = std::make_unique<FFmpegWriter>(mData, mReader);
    else
        return;

    //open writer
    mWriter->open(mData.requestedEncoding);

    //select frame handler class
    DeviceType devtype = mData.deviceList[mData.deviceSelected]->type;
    if (devtype == DeviceType::CPU)
        mFrame = std::make_unique<CpuFrame>(mData, mReader, *mWriter);
    else if (devtype == DeviceType::AVX)
        mFrame = std::make_unique<AvxFrame>(mData, mReader, *mWriter);
    else if (devtype == DeviceType::CUDA)
        mFrame = std::make_unique<CudaFrame>(mData, mReader, *mWriter);
    else if (devtype == DeviceType::OPEN_CL)
        mFrame = std::make_unique<OpenClFrame>(mData, mReader, *mWriter);

    //set up output
    if (ui.chkPlayer->isChecked()) {
        mProgress = std::make_shared<PlayerProgress>(mData, *mFrame, mPlayerWindow);

    } else {
        mProgress = std::make_shared<ProgressGui>(mData, *mFrame, mProgressWindow);
        mProgressWindow->updateInput(mWorkingImage, "");
        mProgressWindow->updateOutput(mWorkingImage, "");
        mProgressWindow->show();
    }

    //set up worker thread
    auto fcn = [=] {
        //no secondary writers
        AuxWriters writers;
        //run loop
        mFrame->runLoop(DeshakerPass::COMBINED, *mProgress, mInputHandler, writers);
    };
    mThread = QThread::create(fcn);
    connect(mThread, &QThread::finished, this, &cuvistaGui::done);

    //begin stabilizing
    mData.timeStart();
    mThread->start();
}

void cuvistaGui::done() {
    mProgressWindow->hide();
    mPlayerWindow->hide();

    //stopwatch
    double secs = mData.timeElapsedSeconds();
    double fps = mWriter->frameEncoded / secs;

    //always destruct writer before frame
    mWriter.reset();
    mFrame.reset();
    mProgress.reset();

    //emit signals to report result back to main thread
    if (errorLogger.hasError())
        doneFail(errorLogger.getErrorMessage());
    else if (mInputHandler.mCurrentInput != UserInputEnum::CONTINUE)
        doneCancel("Operation was cancelled");
    else
        doneSuccess(mData.fileOut, std::format(" written in {:.1f} min at {:.1f} fps", secs / 60.0, fps));

    //delete stabilizer thread
    mThread->deleteLater();
}

void cuvistaGui::doneSuccess(const std::string& fileString, const std::string& str) {
    mProgressWindow->hide();
    QString file = QString::fromStdString(fileString);
    QString url = QString("file:///") + file;
    QFontMetrics metrics(mStatusLinkLabel->font());
    QString fileElided = metrics.elidedText(file, Qt::ElideMiddle, 300);
    QString labelText = QString("<a href='%1'>%2</a> %3").arg(url).arg(fileElided).arg(QString::fromStdString(str));
    statusBar()->clearMessage(); //will show the label which was added in the constructor
    mStatusLinkLabel->setText(labelText);
    mStatusLinkLabel->show();
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
    int boxHeight = 250;
    int boxWidth = 500;

    QMessageBox msgBox(this);
    msgBox.setWindowTitle(QString("Cuvista Info"));
    msgBox.setIcon(QMessageBox::Information);
    QString author = "Copyright (c) 2024 Rainer Bitschi <a href='mailto:cuvista@a1.net'>Email: cuvista@a1.net</a>";
    QString license = "License GNU GPLv3+: GNU GPL version 3 or later";
    QString version = QString::fromStdString(CUVISTA_VERSION);
    msgBox.setText(QString("CUVISTA - Cuda Video Stabilizer, Version %1<br>%2<br>%3").arg(version).arg(author).arg(license));
    msgBox.setTextFormat(Qt::RichText);

    std::stringstream ss;
    mData.showDeviceInfo(ss);

    QPlainTextEdit* textBox = new QPlainTextEdit(this);
    QString qstr = QString::fromStdString(ss.str());
    textBox->setPlainText(qstr);
    textBox->setMinimumHeight(boxHeight);
    textBox->setReadOnly(true);
    textBox->setFont(QFont("Consolas"));
    textBox->setLineWrapMode(QPlainTextEdit::LineWrapMode::NoWrap);

    QSpacerItem* spacer = new QSpacerItem(boxWidth, 0, QSizePolicy::Minimum, QSizePolicy::Minimum);
    QGridLayout* layout = (QGridLayout*) msgBox.layout();
    layout->addWidget(textBox, 1, 0, 1, layout->columnCount());
    layout->addItem(spacer, layout->rowCount(), 0, 1, layout->columnCount());
    msgBox.exec();
}

void cuvistaGui::setColorIcon(ClickLabel* btn, QColor& color) {
    QPixmap icon(30, 24);
    icon.fill(color);
    btn->setPixmap(icon);
}
