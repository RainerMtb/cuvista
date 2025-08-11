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
    ui.labelStatus->setText(qformat("Version {}", CUVISTA_VERSION));

    mData.console = &mData.nullStream;
    mData.printHeader = false;
    mData.probeCuda();
    mData.probeOpenCl();
    mData.collectDeviceInfo();

    QStringList cmdArgs = QCoreApplication::arguments();

    //modes list
    ui.comboMode->addItem(QString("Combined - Single Pass"));
    ui.comboMode->addItem(QString("Two Pass - Analyze then Write"));
    for (int i = 2; i <= 4; i++) {
        ui.comboMode->addItem(QString("Multi Pass - Analyze %1x").arg(i));
    }
    auto fcnModeCheck = [&] (int index) {
        //if (index > 0 && ui.chkPlayer->isChecked()) ui.chkPlayer->setChecked(false);
        ui.chkPlayer->setEnabled(index == 0);
        ui.chkPlayer->setCheckable(index == 0);
    };
    connect(ui.comboMode, &QComboBox::currentIndexChanged, this, fcnModeCheck);

    //available devices
    for (int i = 0; i < mData.deviceList.size(); i++) {
        ui.comboDevice->addItem(QString::fromStdString(mData.deviceList[i]->getName()));
    }

    //combo box for encoding options for selected device
    auto fcnEncoding = [&] (int index) {
        ui.comboEncoding->clear();

        std::vector<EncodingOption>& options = mData.deviceList[index]->encodingOptions;
        for (EncodingOption e : options) {
            QVariant qv = QVariant::fromValue(e);
            QString qs = qformat("{} - {}", mapDeviceToString[e.device], mapCodecToString[e.codec]);

            ui.comboEncoding->addItem(qs, qv);
        }
    };
    //initialize encoder with first device
    fcnEncoding(0);
    //set encoding options when device changes
    connect(ui.comboDevice, &QComboBox::currentIndexChanged, this, fcnEncoding);
    //trigger setting encoding options
    ui.comboDevice->setCurrentIndex(ui.comboDevice->count() - 1);

    //background color selection
    int colorRed = mSettings.value("qt/color/red", mData.backgroundColor.getChannel(0)).toInt();
    int colorGreen = mSettings.value("qt/color/green", mData.backgroundColor.getChannel(1)).toInt();
    int colorBlue = mSettings.value("qt/color/blue", mData.backgroundColor.getChannel(2)).toInt();
    QColor bg(colorRed, colorGreen, colorBlue);
    setBackgroundColor(bg);

    auto fcnColorSelection = [&] () {
        QColorDialog::ColorDialogOptions options = QColorDialog::ColorDialogOption::NoEyeDropperButton;
        QColor result = QColorDialog::getColor(mBackgroundColor, this, QString("Select Background Color"), options);
        if (result.isValid()) {
            setBackgroundColor(result);
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
    ui.spinZoomMin->setMinimum(mData.limits.imZoomMin * 100 - 100);
    ui.spinZoomMin->setMaximum(mData.limits.imZoomMax * 100 - 100);
    ui.spinZoomMin->setValue(std::round(mData.zoomMin * 100 - 100));
    ui.spinZoomMax->setMinimum(mData.limits.imZoomMin * 100 - 100);
    ui.spinZoomMax->setMaximum(mData.limits.imZoomMax * 100 - 100);
    ui.spinZoomMax->setValue(std::round(mData.zoomMax * 100 - 100));

    auto fcnEnable = [&] (Qt::CheckState state) { ui.spinZoomMax->setEnabled(state == Qt::CheckState::Checked); };
    connect(ui.chkDynamicZoom, &QCheckBox::checkStateChanged, this, fcnEnable);

    //seek input
    connect(ui.imageInput, &ImageLabel::mouseClicked, this, &cuvistaGui::seek);

    //start stabilizing
    connect(ui.btnStart, &QPushButton::clicked, this, &cuvistaGui::stabilize);

    //info button
    connect(ui.btnInfo, &QPushButton::clicked, this, &cuvistaGui::showInfo);

    //reset gui
    connect(ui.btnReset, &QPushButton::clicked, this, &cuvistaGui::resetGui);

    //progress handler
    connect(mProgressWindow, &ProgressWindow::cancel, &mInputHandler, &UserInputGui::cancel);
    connect(mProgressWindow, &ProgressWindow::sigProgress, mProgressWindow, &ProgressWindow::progress);
    connect(mProgressWindow, &ProgressWindow::sigUpdateInput, mProgressWindow, &ProgressWindow::updateInput);
    connect(mProgressWindow, &ProgressWindow::sigUpdateOutput, mProgressWindow, &ProgressWindow::updateOutput);
    connect(mProgressWindow, &ProgressWindow::sigUpdateStatus, mProgressWindow, &ProgressWindow::updateStatus);

    //player window
    connect(mPlayerWindow, &PlayerWindow::sigProgress, mPlayerWindow, &PlayerWindow::progress);
    connect(mPlayerWindow, &PlayerWindow::cancel, &mInputHandler, &UserInputGui::cancel);

    //status bar
    connect(this, &cuvistaGui::sigShowStatusMessage, this, &cuvistaGui::showStatusMessage);

    //load recent files before activating combo box action
    for (int idx = 0; idx < 6; idx++) {
        QString key = QString("qt/input%1").arg(idx);
        QString file = mSettings.value(key, "").toString();
        if (file.isEmpty() == false) ui.comboInputFile->addItem(file);
    }

    //select input from list of recent files
    auto fcnSelectFile = [&] (const QString& str) {
        //qDebug() << "combo" << str;
        setInputFile(str);
        ui.comboInputFile->setToolTip(str);
    };
    connect(ui.comboInputFile, &QComboBox::currentTextChanged, this, fcnSelectFile);

    //file for reading
    auto fcnOpenFile = [&] () {
        QString str = QFileDialog::getOpenFileName(this, QString("Select Video file to open"), mInputDir, "All Files (*.*)");
        addInputFile(str);
    };
    connect(ui.btnOpen, &QPushButton::clicked, this, fcnOpenFile);

    //load files from command line argument
    //load input file
    if (cmdArgs.size() > 1) {
        addInputFile(cmdArgs.at(1));
    }
    //preset output file
    if (cmdArgs.size() > 2) {
        mOutputDir = cmdArgs.at(2);
        mOutputFilterSelected = "All Files (*.*)";
    }
    
    //label to show file link in status bar upon success
    auto fileOpener = [] (const QString& file) {
        QUrl qurl = QUrl::fromLocalFile(file);
        QDesktopServices::openUrl(qurl);
    };
    connect(ui.labelStatus, &QLabel::linkActivated, this, fileOpener);

    //stored settings
    //window position and size
    int windowPosX = mSettings.value("qt/window/posx", 50).toInt();
    int windowPosY = mSettings.value("qt/window/posy", 50).toInt();
    int windowWidth = mSettings.value("qt/window/width", 700).toInt();
    int windowHeight = mSettings.value("qt/window/height", 700).toInt();
    move(windowPosX, windowPosY);
    resize(windowWidth, windowHeight);

    //player window position
    int playerPosX = mSettings.value("qt/player/posx", 50).toInt();
    int playerPosY = mSettings.value("qt/player/posy", 50).toInt();
    int playerWidth = mSettings.value("qt/player/width", 640).toInt();
    int playerHeight = mSettings.value("qt/player/height", 480).toInt();
    mPlayerWindow->move(playerPosX, playerPosY);
    mPlayerWindow->resize(playerWidth, playerHeight);

    //other settings
    ui.chkOverwrite->setChecked(mSettings.value("qt/overwrite", false).toBool());
    ui.spinRadius->setValue(mSettings.value("qt/radius", mData.radsec).toDouble());
    ui.spinZoomMin->setValue(mSettings.value("qt/zoom/min", std::round(mData.zoomMin * 100 - 100)).toInt());
    ui.spinZoomMax->setValue(mSettings.value("qt/zoom/max", std::round(mData.zoomMax * 100 - 100)).toInt());
    ui.chkDynamicZoom->setChecked(mSettings.value("qt/zoom/dynamic", true).toBool());
    ui.chkFrameLimit->setChecked(mSettings.value("qt/limit/enable", false).toBool());
    ui.spinFrameLimit->setValue(mSettings.value("qt/limit/value", 500).toInt());
}

//-------------------------
// set input file
//-------------------------

void cuvistaGui::setInputFile(const QString& inputPath) {
    //qDebug() << "input" << inputPath;
    ui.comboAudioTrack->clear();
    ui.comboAudioTrack->addItem("No Audio");
    mInputReady = false;
    
    try {
        mReader.close();
        errorLogger().clearErrors();
        mReader.open(inputPath.toStdString());
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

        ui.comboAudioTrack->setCurrentIndex(ui.comboAudioTrack->count() > 1 ? 1 : 0);
        ui.texInput->setPlainText(QString::fromStdString(str).trimmed());

    } catch (const AVException& ex) {
        ui.imageInput->setImage(mErrorImage);
        ui.labelStatus->setText(QString(ex.what()));
        ui.texInput->setPlainText("");
    }

    mFileInput = QFileInfo(inputPath);
    mInputDir = mFileInput.path();
}

void cuvistaGui::addInputFile(const QString& inputPath) {
    if (inputPath.isEmpty() == false) {
        int idx = ui.comboInputFile->findText(inputPath);
        ui.comboInputFile->insertItem(0, inputPath);
        ui.comboInputFile->setCurrentIndex(0);
        if (idx > -1) {
            ui.comboInputFile->removeItem(idx + 1);
        }
    }
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
        ui.labelStatus->setText({});
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
            ui.labelStatus->setText({});
            return;
        }
    }

    //get output sequence folder
    if (ui.chkSequence->isChecked()) {
        outFile = QFileDialog::getExistingDirectory(this, QString("Select Output Folder"), mOutputDir);
        if (outFile.isEmpty()) {
            ui.labelStatus->setText({});
            return;
        }

        //check for overwrite
        std::string firstFile = ImageWriter::makeFilename(outFile.toStdString(), 0, ui.comboImageType->currentText().toLower().toStdString());
        QString qstr = QString::fromStdString(firstFile);
        if (QFile(qstr).exists() && ui.chkOverwrite->isChecked() == false) {
            QString msgTitle = "Confirm File Overwrite";
            QString msgText = QString("File %1 exists,\noverwrite this and subsequent files?").arg(qstr);
            if (QMessageBox::warning(this, msgTitle, msgText, QMessageBox::Yes | QMessageBox::No) == QMessageBox::No) {
                ui.labelStatus->setText({});
                return;
            }
        }
    }

    //check if input and output point to same file
    if (mFileInput.canonicalFilePath() == mFileOutput.canonicalFilePath()) {
        QMessageBox::critical(this, QString("Invalid File Selection"), QString("Please select different files\nfor input and output"), QMessageBox::Ok);
        ui.labelStatus->setText({});
        return;
    }

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
    uchar rgb[] = { (uchar) mBackgroundColor.red(), (uchar) mBackgroundColor.green(), (uchar) mBackgroundColor.blue() };
    mData.backgroundColor = Color::rgb(rgb[0], rgb[1], rgb[2]);
    mData.backgroundColor.toYUVfloat(&mData.bgcolorYuv.y, &mData.bgcolorYuv.u, &mData.bgcolorYuv.v);

    //rewind reader to beginning of input
    mReader.rewind();
    //check input parameters
    mData.validate(mReader);
    //reset input handler
    mInputHandler.mIsCancelled = false;
    //audio track to play
    int audioStreamIndex = -1;
    if (ui.comboAudioTrack->currentIndex() > 0) {
        audioStreamIndex = ui.comboAudioTrack->currentData().value<StreamContext*>()->inputStream->index;
    }
    //crop setting for stack
    mData.stackCrop = { ui.spinStackLeft->value(), ui.spinStackRight->value() };

    //select writer
    if (ui.chkStack->isChecked())
        mWriter = std::make_shared<StackedWriter>(mData, mReader);
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
    mData.mode = ui.comboMode->currentIndex();
    if (mData.mode == 0) {
        mFrame = std::make_shared<MovieFrameCombined>(mData, mReader, *mWriter);
    } else {
        mFrame = std::make_shared<MovieFrameConsecutive>(mData, mReader, *mWriter);
    }

    //select frame executor class
    mExecutor = mData.deviceList[mData.deviceSelected]->create(mData, *mFrame);

    //set up output
    if (ui.chkPlayer->isChecked()) {
        mProgress = std::make_shared<PlayerProgress>(mData, mPlayerWindow, *mExecutor);

    } else {
        mProgress = std::make_shared<ProgressGui>(mData, mProgressWindow, *mExecutor);
        mProgressWindow->updateInput(mWorkingImage, "");
        mProgressWindow->updateOutput(mWorkingImage, "");
        QPoint p = pos();
        mProgressWindow->move(p.x() + 40, p.y() + 40); //position relativ to parent window
        mProgressWindow->show();
    }

    //set up worker thread
    auto fcn = [&] {
        sigShowStatusMessage("stabilizing...");
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
    QFontMetrics metrics(ui.labelStatus->font());
    QString fileElided = metrics.elidedText(file, Qt::ElideMiddle, 300);
    QString labelText = QString("<a href='%1'>%2</a> %3").arg(file).arg(fileElided).arg(QString::fromStdString(str));
    ui.labelStatus->setText(labelText);
}

void cuvistaGui::doneFail(const std::string& str) {
    QString msg = QString::fromStdString(str);
    QMessageBox::critical(this, QString("Error"), msg, QMessageBox::Ok);
    ui.labelStatus->setText(QString("Error: %1").arg(msg));
    errorLogger().clearErrors();
}

void cuvistaGui::doneCancel(const std::string& str) {
    ui.labelStatus->setText(QString::fromStdString(str));
}

void cuvistaGui::showStatusMessage(const std::string& msg) {
    ui.labelStatus->setText(QString::fromStdString(msg));
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
void cuvistaGui::setBackgroundColor(const QColor& color) {
    mBackgroundColor = color;
    
    QString style = qformat("background-color: rgb({}, {}, {})", color.red(), color.green(), color.blue());
    ui.imageInput->setStyleSheet(style);
    mProgressWindow->setBackgroundColor(style);

    QPixmap icon(30, 24);
    icon.fill(color);
    ui.lblColor->setPixmap(icon);
}

//terminate test thread when closing info dialog
void InfoDialog::closeEvent(QCloseEvent* event) {
    worker->wait();
    worker->deleteLater();
    event->accept();
}

//reset gui options to defaults
void cuvistaGui::resetGui() {
    setBackgroundColor(QColor::fromRgb(0, 150, 0));

    if (ui.comboInputFile->count() > 0) {
        QString item = ui.comboInputFile->currentText();
        ui.comboInputFile->insertItem(0, item);
        ui.comboInputFile->setCurrentIndex(0);
        while (ui.comboInputFile->count() > 1) ui.comboInputFile->removeItem(1);
    }

    mPlayerWindow->move(50, 50);
    mProgressWindow->resize(640, 480);

    ui.chkOverwrite->setChecked(false);
    ui.spinRadius->setValue(0.5);
    ui.spinZoomMin->setValue(5);
    ui.spinZoomMax->setValue(15);
    ui.chkDynamicZoom->setChecked(true);
    ui.chkFrameLimit->setChecked(false);
    ui.spinFrameLimit->setValue(500);
}

//destructor stores settings
cuvistaGui::~cuvistaGui() {
    mSettings.setValue("qt/color/red", mBackgroundColor.red());
    mSettings.setValue("qt/color/green", mBackgroundColor.green());
    mSettings.setValue("qt/color/blue", mBackgroundColor.blue());

    //main window
    mSettings.setValue("qt/window/posx", x());
    mSettings.setValue("qt/window/posy", y());
    mSettings.setValue("qt/window/width", width());
    mSettings.setValue("qt/window/height", height());

    //player window
    mSettings.setValue("qt/player/posx", mPlayerWindow->x());
    mSettings.setValue("qt/player/posy", mPlayerWindow->y());
    mSettings.setValue("qt/player/width", mPlayerWindow->width());
    mSettings.setValue("qt/player/height", mPlayerWindow->height());

    mSettings.setValue("qt/overwrite", ui.chkOverwrite->isChecked());
    mSettings.setValue("qt/radius", ui.spinRadius->value());
    mSettings.setValue("qt/zoom/min", ui.spinZoomMin->value());
    mSettings.setValue("qt/zoom/max", ui.spinZoomMax->value());
    mSettings.setValue("qt/zoom/dynamic", ui.chkDynamicZoom->isChecked());
    mSettings.setValue("qt/limit/enable", ui.chkFrameLimit->isChecked());
    mSettings.setValue("qt/limit/value", ui.spinFrameLimit->value());

    //recent files
    for (int idx = 0; idx < 6; idx++) {
        QString key = QString("qt/input%1").arg(idx);
        mSettings.setValue(key, ui.comboInputFile->itemText(idx));
    }
}
