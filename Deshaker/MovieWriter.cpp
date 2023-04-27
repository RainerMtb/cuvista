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

#include "MovieWriter.hpp"
#include "Stats.hpp"

#include <io.h>
#include <fcntl.h>

OutputContext MovieWriter::getOutputData() {
	return { true, false, &outputFrame, nullptr, 0 };
}

std::string ImageWriter::makeFilename(const std::string& pattern, int64_t index) {
	const int siz = 256;
	char fname[siz];
	std::snprintf(fname, siz, pattern.c_str(), index);
	return fname;
}

std::string ImageWriter::makeFilename() const {
	return makeFilename(data.fileOut, status.frameWriteIndex);
}

//-----------------------------------------------------------------------------------
// BMP Images
//-----------------------------------------------------------------------------------

void BmpImageWriter::write() {
	outputFrame.toBGR(image).saveAsBMP(makeFilename());
	status.outputBytesWritten += image.dataSizeInBytes();
}

//-----------------------------------------------------------------------------------
// JPG Images through ffmpeg
//-----------------------------------------------------------------------------------

void JpegImageWriter::open() {
	const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
	ctx = avcodec_alloc_context3(codec);
	ctx->width = data.w;
	ctx->height = data.h;
	ctx->time_base = { 1, 1 };
	ctx->framerate = { 1, 1 };
	ctx->codec_type = AVMEDIA_TYPE_VIDEO;
	ctx->pix_fmt = AV_PIX_FMT_YUV444P;
	ctx->color_range = AVCOL_RANGE_JPEG;
	int ret = avcodec_open2(ctx, codec, NULL);
	if (ret < 0)
		throw AVException(av_make_error(ret, "cannot open mjpeg codec"));

	frame = av_frame_alloc();
	frame->format = ctx->pix_fmt;
	frame->width = data.w;
	frame->height = data.h;

	frame->linesize[0] = outputFrame.stride;
	frame->linesize[1] = outputFrame.stride;
	frame->linesize[2] = outputFrame.stride;
	frame->data[0] = outputFrame.plane(0);
	frame->data[1] = outputFrame.plane(1);
	frame->data[2] = outputFrame.plane(2);

	packet = av_packet_alloc();
}

void JpegImageWriter::write() {
	frame->pts = status.frameWriteIndex;
	int result = avcodec_send_frame(ctx, frame);
	if (result < 0)
		errorLogger.logError(av_make_error(result, "error sending frame"));

	result = avcodec_receive_packet(ctx, packet);
	if (result < 0)
		errorLogger.logError(av_make_error(result, "error receiving packet"));

	std::string fname = makeFilename();
	std::ofstream file(fname, std::ios::binary);
	file.write(reinterpret_cast<char*>(packet->data), packet->size);

	status.outputBytesWritten += packet->size;
	av_packet_unref(packet);
}

JpegImageWriter::~JpegImageWriter() {
	av_packet_free(&packet);
	avcodec_free_context(&ctx);
}

//-----------------------------------------------------------------------------------
// YUV444 without stride
//-----------------------------------------------------------------------------------

void RawWriter::packYuv() {
	unsigned char* yuv = outputFrame.data();
	char* dest = yuvPacked.data();

	for (int i = 0; i < 3ull * outputFrame.h; i++) {
		std::copy(yuv, yuv + outputFrame.w, dest);
		yuv += outputFrame.stride;
		dest += outputFrame.w;
	}
}

//-----------------------------------------------------------------------------------
// send to command line pipe
//-----------------------------------------------------------------------------------

void PipeWriter::open() {
    //set stdout to binary mode
    int result = _setmode(_fileno(stdout), _O_BINARY);
    if (result < 0) 
		throw AVException("Pipe: error setting stdout to binary");
}

void PipeWriter::write() {
	packYuv();
	size_t siz = fwrite(yuvPacked.data(), 1, yuvPacked.size(), stdout);
	if (siz != yuvPacked.size()) {
		errorLogger.logError("Pipe: error writing data");
	}
	status.outputBytesWritten += siz;

	//static std::ofstream out("f:/test.yuv", std::ios::binary);
	//out.write(yuvPacked.data(), yuvPacked.size());
}

PipeWriter::~PipeWriter() {
    int result = _setmode(_fileno(stdout), _O_TEXT);
	if (result < 0) {
		errorLogger.logError("Pipe: error setting stdout back to text");
	}
}

//-----------------------------------------------------------------------------------
// send to TCP address
//-----------------------------------------------------------------------------------

void TCPWriter::open() {
	WSADATA wsaData {};
	int retval = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (retval < 0) 
		throw AVException("cannot start TCP");

	SOCKET mSock = socket(AF_INET, SOCK_STREAM, 0);
	if (mSock < 0) 
		throw AVException("cannot create TCP socket");

	sockaddr_in sockAddr;
	sockaddr* sockaddr_ptr = (sockaddr*) &sockAddr;
	int addrSize = sizeof(sockAddr);
	memset(&sockAddr, 0, addrSize);
	sockAddr.sin_family = AF_INET;
	inet_pton(AF_INET, data.tcp_address.c_str(), &sockAddr.sin_addr.s_addr);
	sockAddr.sin_port = htons(data.tcp_port);

	bind(mSock, sockaddr_ptr, addrSize);
	listen(mSock, 3);
	*data.console << "listening for TCP connection at " << data.tcp_address << ":" << data.tcp_port << std::endl;
	mConn = accept(mSock, sockaddr_ptr, &addrSize); //blocking call, wait for connecting client
	if (mConn < 0) throw AVException("cannot connect");
	*data.console << "established TCP connection" << std::endl;
}

void TCPWriter::write() {
	packYuv();
	int retval = send(mConn, yuvPacked.data(), (int) yuvPacked.size(), 0);
	if (retval < 0 || retval != yuvPacked.size()) {
		errorLogger.logError("error sending TCP data: " + WSAGetLastError());
	}
	status.outputBytesWritten += retval;
}

TCPWriter::~TCPWriter() {
	closesocket(mConn);
	closesocket(mSock);
	WSACleanup();
}

//so GUID can be used as key in a map
bool CudaFFmpegWriter::FunctorLess::operator () (const GUID& g1, const GUID& g2) const {
	return std::tie(g1.Data1, g1.Data2, g1.Data3, *g1.Data4) < std::tie(g2.Data1, g2.Data2, g2.Data3, *g2.Data4);
}
