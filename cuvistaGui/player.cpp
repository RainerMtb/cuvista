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
#include <QCloseEvent>
#include <QMediaDevices>
#include <QAudioDevice>
#include <QAudioOutput>
#include <QVideoFrameFormat>
#include <QVideoFrameInput>
#include <algorithm>

#include "player.h"
#include "MovieFrame.hpp"
#include "MovieReader.hpp"
#include "Image2.hpp"


//-------------------------------------------------
//------------- Player Window ---------------------
//-------------------------------------------------

PlayerWindow::PlayerWindow(QWidget* parent) :
    QMainWindow(parent) 
{
    ui.setupUi(this);
    setWindowModality(Qt::ApplicationModal);
    connect(ui.btnPause, &QPushButton::clicked, this, &PlayerWindow::pause);
    connect(ui.btnPlay, &QPushButton::clicked, this, &PlayerWindow::play);
}

QVideoWidget* PlayerWindow::videoWidget() {
    return ui.videoWidget;
}

void PlayerWindow::progress(QString str, QString status) {
    ui.lblFrame->setText(str);
    ui.lblStatus->setText(status);
}

void PlayerWindow::pause() {
    isPaused = true;
    ui.lblStatus->setText("Pausing...");
}

void PlayerWindow::play() {
    isPaused = false;
    ui.lblStatus->setText("Playing...");
}

void PlayerWindow::closeEvent(QCloseEvent* event) {
    cancel();
    isPaused = false;

    //hide only after output is terminated in main window
    event->ignore();
}


//-------------------------------------------------
//------------- Image FFmpeg Buffer ---------------
//-------------------------------------------------

ImageYuvFFmpeg::ImageYuvFFmpeg(AVFrame* av_frame) :
    av_frame { av_frame } {}

uint8_t* ImageYuvFFmpeg::addr(size_t idx, size_t r, size_t c) {
    return av_frame->data[idx] + r * av_frame->linesize[idx] + c;
}

const uint8_t* ImageYuvFFmpeg::addr(size_t idx, size_t r, size_t c) const {
    return av_frame->data[idx] + r * av_frame->linesize[idx] + c;
}

uint8_t* ImageYuvFFmpeg::plane(size_t idx) {
    return av_frame->data[idx];
}

const uint8_t* ImageYuvFFmpeg::plane(size_t idx) const {
    return av_frame->data[idx];
}

int ImageYuvFFmpeg::strideInBytes() const {
    assert(av_frame->linesize[0] == av_frame->linesize[1] && av_frame->linesize[0] == av_frame->linesize[2]);
    return av_frame->linesize[0];
}

int ImageYuvFFmpeg::height() const {
    return av_frame->height;
}

int ImageYuvFFmpeg::width() const {
    return av_frame->width;
}

int ImageYuvFFmpeg::planes() const {
    return 3;
}

void ImageYuvFFmpeg::setIndex(int64_t frameIndex) {
    index = frameIndex;
}

bool ImageYuvFFmpeg::saveAsBMP(const std::string& filename, uint8_t scale) const {
    int h = height();
    int w = width();
    Matc y = Matc::fromRowData(h, w, strideInBytes(), plane(0));
    Matc u = Matc::fromRowData(h, w, strideInBytes(), plane(1));
    Matc v = Matc::fromRowData(h, w, strideInBytes(), plane(2));
    return ImageYuvMat8(h, w, w, y.data(), u.data(), v.data()).saveAsBMP(filename, scale);
}

//-------------------------------------------------
//------------- Writer Class ----------------------
//-------------------------------------------------

PlayerWriter::PlayerWriter(MainData& data, MovieReader& reader, PlayerWindow* player, QImage imageWorking, int audioStreamIndex) :
    FFmpegWriter(data, reader, 0),
    QObject(nullptr),
    mPlayer { player },
    mImageWorking { imageWorking },
    mAudioStreamIndex { audioStreamIndex },
    mBufferDevice(2ull * data.w * data.h * data.radius) {}

int PlayerWriter::writeBuffer(void* opaque, const uint8_t* buf, int bufsiz) {
    PlayerWriter* ptr = static_cast<PlayerWriter*>(opaque);

    //qDebug() << ">>writing" << bufsiz;
    //static std::ofstream outfile("f:/file.dat", std::ios::binary);
    //outfile.write(reinterpret_cast<const char*>(buf), bufsiz);
    return bufsiz;
}

void PlayerWriter::open(EncodingOption videoCodec) {
    av_log_set_callback(ffmpeg_log);
    int result;
    
    //setup output format
    const AVOutputFormat* ofmt = av_guess_format("asf", NULL, NULL);
    result = avformat_alloc_output_context2(&fmt_ctx, ofmt, NULL, NULL);
    if (result < 0)
        throw AVException(av_make_error(result));

    //io context for memory buffer
    int bufsiz = 16 * 1024;
    mBuffer = (unsigned char*) av_malloc(bufsiz);
    m_av_avio = avio_alloc_context(mBuffer, bufsiz, 1, this, nullptr, &PlayerWriter::writeBuffer, nullptr);
    m_av_avio->seekable = 0;
    fmt_ctx->pb = m_av_avio;
    fmt_ctx->flags |= AVFMT_FLAG_CUSTOM_IO;

    for (StreamContext& sc : mReader.inputStreams) {
        if (sc.inputStream->index == mReader.videoStream->index) {
            sc.handling = StreamHandling::STREAM_STABILIZE;

        } else if (sc.inputStream->index == mAudioStreamIndex) {
            int codecSupported = avformat_query_codec(fmt_ctx->oformat, sc.inputStream->codecpar->codec_id, FF_COMPLIANCE_STRICT);
            if (codecSupported == 1) {
                sc.handling = StreamHandling::STREAM_COPY;
            }

        } else {
            sc.handling = StreamHandling::STREAM_IGNORE;
        }
    }

    //open ffmpeg
    AVCodecID id = AV_CODEC_ID_FFVHUFF;
    FFmpegFormatWriter::open(id);
    FFmpegWriter::open(id, AV_PIX_FMT_YUV444P, mData.h, mData.w, mData.cpupitch);

    //yuv image refering to ffmpeg frame buffer
    mImageFrame = ImageYuvFFmpeg(av_frame);

    mPlayer->show();
    mMediaPlayer.setVideoOutput(mPlayer->videoWidget());

    //mediaPlayer.setAudioOutput(&mAudioOutput);
    //mediaPlayer.setSourceDevice(&playerDevice);
    //mediaPlayer.play();

    imageBuffer.emplace_back(mData.h, mData.w, mData.cpupitch);
}

//this runs on a background thread
void PlayerWriter::prepareOutput(FrameExecutor& executor) {
    executor.getOutput(frameIndex, mImageFrame);
}

void PlayerWriter::write(const FrameExecutor& executor) {
    av_frame->pts = mImageFrame.index;
    sendFFmpegFrame(av_frame);
    writeFFmpegPacket(av_frame);
    frameIndex++;
}

//close video player
bool PlayerWriter::flush() {
    //wait some time after the last frame is displayed before closing the player
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    return false; 
}

//terminate encoding
bool PlayerWriter::startFlushing() { 
    sendFFmpegFrame(nullptr);
    writeFFmpegPacket(nullptr);
    return true;
}

PlayerWriter::~PlayerWriter() {
    avio_context_free(&m_av_avio);
}


//-------------------------------------------------
//------------- Progress Class --------------------
//-------------------------------------------------

void PlayerProgress::update(bool force) {
    int64_t idx = frame.mWriter.frameIndex - 1;
    auto opstr = frame.mReader.ptsForFrameAsString(idx);

    //frame stats
    QString str = "";
    if (opstr.has_value()) str = QString("%1 (%2)").arg(idx).arg(QString::fromStdString(*opstr));

    //player state
    QString status = "Playing...";
    if (mPlayer->isPaused) status = "Pausing...";
    else if (idx < 0) status = "Buffering...";
    else if (frame.mWriter.frameIndex == frame.mReader.frameIndex) status = "Ending...";
}