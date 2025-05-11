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

#include <QDebug>
#include <QStandardPaths>
#include <QMessageBox>
#include <QDialog>
#include <QColorDialog>
#include <QDesktopServices>
#include <QScrollBar>

#include "cuvistaGui.h"
#include "MessagePrinterGui.h"
#include "SelfTest.hpp"
#include "UserInputGui.hpp"
#include "MovieFrame.hpp"
#include "progress.h"
#include "CudaWriter.hpp"

template <class... Args> QString qformat(std::format_string<Args...> fmt, Args&&... args) {
    return QString::fromStdString(std::format(fmt, std::forward<Args>(args)...));
}

cuvistaGui::cuvistaGui(QWidget *parent) : 
    QMainWindow(parent) 
{
    ui.setupUi(this);
    mPlayerWindow = new PlayerWindow(this);
    mProgressWindow = new ProgressWindow(this);
    mMovieDir = QStandardPaths::locate(QStandardPaths::MoviesLocation, QString(), QStandardPaths::LocateDirectory);
    mInputDir = mMovieDir;
    mOutputDir = mMovieDir;
    ui.labelVersion->setText(qformat("Version {}", CUVISTA_VERSION));

    mData.console = &mData.nullStream;
    mData.printHeader = false;
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
            QString qs = qformat("{} - {}", mData.mapDeviceToString[e.device], mData.mapCodecToString[e.codec]);

            ui.comboEncoding->addItem(qs, qv);
        }
    };
    //initialize encoder with first device
    fcnEncoding(0);
    //set encoding options when device changes
    connect(ui.comboDevice, &QComboBox::currentIndexChanged, this, fcnEncoding);
    //trigger setting encoding options
    ui.comboDevice->setCurrentIndex(ui.comboDevice->count() - 1);

    //file for reading
    auto fcnOpen = [&] () {
        QString str = QFileDialog::getOpenFileName(this, QString("Select Video file to open"), mInputDir, "All Files (*.*)");
        if (!str.isEmpty()) setInputFile(str);
    };
    connect(ui.btnOpen, &QPushButton::clicked, this, fcnOpen);

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
            QString style = qformat("background-color: rgb({}, {}, {})", result.red(), result.green(), result.blue());
            ui.imageInput->setStyleSheet(style);
            mProgressWindow->setBackgroundColor(style);
        }
    };
    connect(ui.lblColor, &ClickLabel::clicked, this, fcnColorSelection);

    //image sequences
    ui.comboImageType->addItem("BMP", QVariant::fromValue(OutputType::SEQUENCE_BMP));
    ui.comboImageType->addItem("JPG", QVariant::fromValue(OutputType::SEQUENCE_JPG));

    //limits
    ui.spinRadius->setMinimum(mData.limits.radsecMin);
    ui.spinRadius->setMaximum(mData.limits.radsecMax);
    ui.spinZoomMin->setMinimum(mData.limits.imZoomMin * 100 - 100);
    ui.spinZoomMin->setMaximum(mData.limits.imZoomMax * 100 - 100);
    ui.spinZoomMin->setValue(std::round(mData.zoomMin * 100 - 100));
    ui.spinZoomMax->setMinimum(mData.limits.imZoomMin * 100 - 100);
    ui.spinZoomMax->setMaximum(mData.limits.imZoomMax * 100 - 100);
    ui.spinZoomMax->setValue(std::round(mData.zoomMax * 100 - 100));

    auto fcnEnable = [&] (Qt::CheckState state) { ui.spinZoomMax->setEnabled(state == Qt::CheckState::Checked); };
    connect(ui.chkDynamicZoom, &QCheckBox::checkStateChanged, this, fcnEnable);

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
    connect(mPlayerWindow, &PlayerWindow::sigProgress, mPlayerWindow, &PlayerWindow::progress);
    connect(mPlayerWindow, &PlayerWindow::cancel, &mInputHandler, &UserInputGui::cancel);

    //set default main window size
    resize(700, 650);

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
    mStatusLinkLabel = new QLabel(this);
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
    ui.comboAudioTrack->clear();
    mInputReady = false;
    
    try {
        mReader.close();
        errorLogger().clearErrors();
        mReader.open(filePath.toStdString());
        mInputYUV = ImageYuv(mReader.h, mReader.w);

        mReader.read(mInputYUV); //read first image

        if (errorLogger().hasNoError()) {
            mReader.read(mInputYUV); //try to read again for second image
        }

        if (errorLogger().hasNoError()) {
            mInputReady = true;
            seek(0.1);
        }

        if (errorLogger().hasError()) {
            throw AVException(errorLogger().getErrorMessage());
        }

        statusBar()->showMessage({});

        //info about streams
        std::string str;
        for (StreamContext& sc : mReader.mInputStreams) {
            StreamInfo info = sc.inputStreamInfo();
            str += std::format("- stream {}\ntype: {}, codec: {}, duration: {}\n", 
                sc.inputStream->index, info.streamType, info.codec, info.durationString);
            if (sc.inputStream->index == mReader.videoStream->index) {
                std::string frameCount = mReader.frameCount == 0 ? "unknown" : std::to_string(mReader.frameCount);
                str += std::format("video {} x {} px @{:.3f} fps ({}:{})\nvideo frames: {}\n",
                    mReader.w, mReader.h, mReader.fps(), mReader.fpsNum, mReader.fpsDen, frameCount);
            }
            if (info.mediaType == AVMEDIA_TYPE_AUDIO) {
                QString qstr = qformat("Track {}: {}", sc.inputStream->index, info.codec);
                ui.comboAudioTrack->addItem(qstr, QVariant::fromValue(&sc));
            }
        }
        ui.texInput->setPlainText(QString::fromStdString(str).trimmed());
        bool hasAudio = ui.comboAudioTrack->count() > 0;
        ui.chkPlayAudio->setEnabled(hasAudio);
        ui.chkPlayAudio->setChecked(hasAudio);

    } catch (const AVException& ex) {
        ui.imageInput->setImage(mErrorImage);
        ui.statusbar->showMessage(QString(ex.what()));
        ui.texInput->setPlainText("");
    }
    mFileInput = QFileInfo(filePath);
    mInputDir = mFileInput.path();
    ui.fileOpen->setText(mFileInput.fileName());
    ui.fileOpen->setToolTip(filePath);
}

void cuvistaGui::seek(double frac) {
    if (mInputReady && mReader.seek(frac) && mReader.read(mInputYUV)) {
        updateInputImage();
        ui.inputPosition->setValue(frac * 100.0);
    }

    //seeking may cause ffmpeg decoding error messages, ignore
    errorLogger().clearErrors(ErrorSource::FFMPEG);
}

void cuvistaGui::updateInputImage() {
    ui.imageInput->setImage(mInputYUV);
}

//-------------------------
// begin stabilization
//-------------------------

void cuvistaGui::stabilize() {
    //check if input is present
    if (mFileInput.fileName().isEmpty() || mInputReady == false) {
        statusBar()->showMessage(mDefaultMessage);
        return; //nothing to do
    }

    //get output video file
    QString outFile;
    if (ui.chkEncode->isChecked() || ui.chkStack->isChecked()) {
        QFileDialog::Options op = ui.chkOverwrite->isChecked() ? QFileDialog::Option::DontConfirmOverwrite : QFileDialog::Options();
        //file filter: different file endings separate by space, different filter entries separate by two semicolons
        QString fileFilter("Video Files (*.mp4 *.mkv);;All Files (*.*)");
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

        //check for overwrite
        std::string firstFile = ImageWriter::makeFilename(outFile.toStdString(), 0, ui.comboImageType->currentText().toLower().toStdString());
        QString qstr = QString::fromStdString(firstFile);
        if (QFile(qstr).exists() && ui.chkOverwrite->isChecked() == false) {
            QString msgTitle = "Confirm File Overwrite";
            QString msgText = QString("File %1 exists,\noverwrite this and subsequent files?").arg(qstr);
            if (QMessageBox::warning(this, msgTitle, msgText, QMessageBox::Yes | QMessageBox::No) == QMessageBox::No) {
                return;
            }
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
    mData.zoomMin = 1.0 + ui.spinZoomMin->value() / 100.0;
    mData.zoomMax = ui.chkDynamicZoom->isChecked() ? 1.0 + ui.spinZoomMax->value() / 100.0 : mData.zoomMin;
    mData.bgmode = ui.radioBlend->isChecked() ? BackgroundMode::BLEND : BackgroundMode::COLOR;
    if (ui.chkFrameLimit->isChecked()) mData.maxFrames = ui.spinFrameLimit->value();

    using uchar = unsigned char;
    mData.bgcol_rgb = { (uchar) mBackgroundColor.red(), (uchar) mBackgroundColor.green(), (uchar) mBackgroundColor.blue() };
    mData.bgcol_yuv = mData.bgcol_rgb.toNormalized();

    //rewind reader to beginning of input
    mReader.rewind();
    //check input parameters
    mData.validate(mReader);
    //reset input handler
    mInputHandler.mIsCancelled = false;
    //audio track to play
    int audioStreamIndex = -1;
    if (ui.chkPlayAudio->isChecked() && ui.comboAudioTrack->count() > 0) {
        audioStreamIndex = ui.comboAudioTrack->currentData().value<StreamContext*>()->inputStream->index;
    }

    //select writer
    if (ui.chkStack->isChecked())
        mWriter = std::make_shared<StackedWriter>(mData, mReader, ui.slideStack->value() / 100.0);
    else if (ui.chkSequence->isChecked() && ui.comboImageType->currentData().value<OutputType>() == OutputType::SEQUENCE_BMP)
        mWriter = std::make_shared<BmpImageWriter>(mData, mReader);
    else if (ui.chkSequence->isChecked() && ui.comboImageType->currentData().value<OutputType>() == OutputType::SEQUENCE_JPG)
        mWriter = std::make_shared<JpegImageWriter>(mData, mReader);
    else if (ui.chkPlayer->isChecked())
        mWriter = std::make_shared<PlayerWriter>(mData, mReader, mPlayerWindow, mWorkingImage, audioStreamIndex);
    else if (ui.chkEncode->isChecked() && mData.requestedEncoding.device == EncodingDevice::NVENC)
        mWriter = std::make_shared<CudaFFmpegWriter>(mData, mReader);
    else if (ui.chkEncode->isChecked())
        mWriter = std::make_shared<FFmpegWriter>(mData, mReader);
    else
        return;

    //open writer
    mWriter->open(mData.requestedEncoding);

    //select frame handler
    mFrame = std::make_shared<MovieFrameCombined>(mData, mReader, *mWriter);

    //select frame executor class
    mExecutor = mData.deviceList[mData.deviceSelected]->create(mData, *mFrame);

    //set up output
    if (ui.chkPlayer->isChecked()) {
        mProgress = std::make_shared<PlayerProgress>(mData, *mFrame, mPlayerWindow);

    } else {
        mProgress = std::make_shared<ProgressGui>(mData, *mFrame, mProgressWindow, *mExecutor);
        mProgressWindow->updateInput(mWorkingImage, "");
        mProgressWindow->updateOutput(mWorkingImage, "");
        QPoint p = pos();
        mProgressWindow->move(p.x() + 40, p.y() + 40); //on linux automatic positioning is bad
        mProgressWindow->show();
    }

    //set up worker thread
    auto fcn = [&] {
        //init writer on executor thread
        mWriter->start();
        //run loop
        mFrame->runLoop(mProgress, mInputHandler, mExecutor);
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
    mExecutor.reset();
    mFrame.reset();
    mProgress.reset();
    mThread->deleteLater();

    //emit signals to report result back to main thread
    if (errorLogger().hasError())
        doneFail(errorLogger().getErrorMessage());
    else if (mInputHandler.mIsCancelled)
        doneCancel("Operation was cancelled");
    else
        doneSuccess(mData.fileOut, std::format(" written in {:.1f} min at {:.1f} fps", secs / 60.0, fps));
}

void cuvistaGui::doneSuccess(const std::string& fileString, const std::string& str) {
    mProgressWindow->hide();
    QString file = QString::fromStdString(fileString);
    QUrl url = QUrl::fromLocalFile(file);
    QFontMetrics metrics(mStatusLinkLabel->font());
    QString fileElided = metrics.elidedText(file, Qt::ElideMiddle, 300);
    QString labelText = QString("<a href='%1'>%2</a> %3").arg(file).arg(fileElided).arg(QString::fromStdString(str));
    statusBar()->clearMessage(); //will show the label which was added in the constructor
    mStatusLinkLabel->setText(labelText);
    mStatusLinkLabel->show();
}

void cuvistaGui::doneFail(const std::string& str) {
    QString msg = QString::fromStdString(str);
    QMessageBox::critical(this, QString("Error"), msg, QMessageBox::Ok);
    showMessage(QString("Error: %1").arg(msg));
    errorLogger().clearErrors();
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
    int boxWidth = 450;

    InfoDialog msgBox(this);
    msgBox.setWindowTitle(QString("Cuvista Info"));
    std::string strEmail = "cuvista@a1.net";
    std::string strGitHub = "https://github.com/RainerMtb/cuvista";
    QString headerText = qformat(
        "CUVISTA - Cuda Video Stabilizer, Version {}<br>"
        "Copyright (c) 2024 Rainer Bitschi <a href='mailto:{}'>{}</a> <a href='{}'>{}</a><br>"
        "License GNU GPLv3+: GNU GPL version 3 or later<br>"
        "Gui compiled with Qt version {}, running on version {}",
        CUVISTA_VERSION, strEmail, strEmail, strGitHub, strGitHub, QT_VERSION_STR, qVersion());

    QLabel* header = new QLabel(&msgBox);
    header->setText(headerText);
    header->setTextFormat(Qt::RichText);

    std::stringstream ss;
    mData.showDeviceInfo(ss);

    ScrollingTextEdit* textBox = new ScrollingTextEdit(&msgBox);
    QString qstr = QString::fromStdString(ss.str());
    textBox->setPlainText(qstr);
    textBox->setMinimumHeight(boxHeight);
    textBox->setReadOnly(true);
    textBox->setFont(QFont("Consolas"));
    textBox->setLineWrapMode(QPlainTextEdit::LineWrapMode::NoWrap);

    QPushButton* btnTest = new QPushButton("Run Test", &msgBox);
    btnTest->setFixedWidth(100);
    btnTest->setFixedHeight(28);
    QPushButton* btnClose = new QPushButton("Close", &msgBox);
    btnClose->setFixedWidth(70);
    btnClose->setFixedHeight(28);

    //create widgets
    QGridLayout* layout = new QGridLayout(&msgBox);
    layout->addWidget(header, 0, 0, 1, 3);
    layout->addWidget(textBox, 1, 0, 1, 3);
    layout->addWidget(btnTest, 2, 1, 1, 1);
    layout->addWidget(btnClose, 2, 2, 1, 1);

    //calculate text width
    QFontMetrics fm = textBox->fontMetrics();
    for (auto& s : qstr.split('\n')) {
        int w = fm.horizontalAdvance(s);
        if (w > boxWidth) boxWidth = w;
    }
    textBox->setMinimumWidth(boxWidth + 40);

    msgBox.worker = QThread::create([&] {
        MessagePrinterGui printer;
        connect(&printer, &MessagePrinterGui::appendText, textBox, &ScrollingTextEdit::appendText);
        runSelfTest(printer, mData.deviceList);
    });
    auto fcn = [&] {
        btnTest->setEnabled(false);
        msgBox.worker->start();
    };
    connect(btnTest, &QPushButton::clicked, this, fcn);
    connect(btnClose, &QPushButton::clicked, &msgBox, &QDialog::close);
    
    msgBox.exec();
}

//show selected color
void cuvistaGui::setColorIcon(ClickLabel* btn, QColor& color) {
    QPixmap icon(30, 24);
    icon.fill(color);
    btn->setPixmap(icon);
}

//terminate test thread when closing info dialog
void InfoDialog::closeEvent(QCloseEvent* event) {
    worker->wait();
    worker->deleteLater();
    event->accept();
}