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

#include <filesystem>
#include "MovieWriter.hpp"
#include "MovieFrame.hpp"
#include "MovieReader.hpp"

SimpleYuvWriter::SimpleYuvWriter(const std::string& file) {
	os = std::ofstream(file, std::ios::binary);
}

void SimpleYuvWriter::write(ImageYuv& image) {
	const unsigned char* src = image.data();
	for (int i = 0; i < 3ull * image.h; i++) {
		os.write(reinterpret_cast<const char*>(src), image.w);
		src += image.stride;
	}
}


//-----------------------------------------------------------------------------------
// Writer Collection
//-----------------------------------------------------------------------------------

MovieWriterCollection::MovieWriterCollection(MainData& data, MovieReader& reader, std::shared_ptr<MovieWriter> mainWriter) :
	NullWriter(data, reader),
	mainWriter { mainWriter } {}

void MovieWriterCollection::addWriter(std::shared_ptr<MovieWriter> writer) {
	auxWriters.push_back(writer);
}

void MovieWriterCollection::open(OutputOption outputOption) {
	mainWriter->open(outputOption);
	for (auto& writer : auxWriters) writer->open(outputOption);
}

void MovieWriterCollection::start() {
	mainWriter->start();
	for (auto& writer : auxWriters) writer->start();
}

void MovieWriterCollection::updateStats() {
	std::unique_lock<std::mutex> lock(mStatsMutex);
	frameIndex = mainWriter->frameIndex;
	frameEncoded = mainWriter->frameEncoded;
	encodedBytesTotal = mainWriter->encodedBytesTotal;
	outputBytesWritten = mainWriter->outputBytesWritten;
}

void MovieWriterCollection::writeInput(const FrameExecutor& executor) {
	mainWriter->writeInput(executor);
	for (auto& writer : auxWriters) writer->writeInput(executor);
	updateStats();
}

void MovieWriterCollection::writeOutput(const FrameExecutor& executor) {
	mainWriter->writeOutput(executor);
	for (auto& writer : auxWriters) writer->writeOutput(executor);
	updateStats();
}

bool MovieWriterCollection::flush() {
	bool retval = mainWriter->flush();
	for (auto& writer : auxWriters) writer->flush();
	updateStats();
	return retval;
}

void MovieWriterCollection::close() {
	mainWriter->close();
	for (auto& writer : auxWriters) writer->close();
	updateStats();
}


//-----------------------------------------------------------------------------------
// Raw Format Writers
//-----------------------------------------------------------------------------------

void OutputWriter::writeOutput(const FrameExecutor& executor) {
	executor.getOutputYuv(frameIndex, outputFrame);
	this->frameIndex++;
}

void RawMemoryStoreWriter::writeOutput(const FrameExecutor& executor) {
	if (doWriteOutput) {
		{
			ImageYuv& frameYuv = outputFramesYuv.emplace_back(executor.mData.h, executor.mData.w);
			executor.getOutputYuv(frameIndex, frameYuv);
			while (outputFramesYuv.size() > maxFrameCount) outputFramesYuv.pop_front();
		}

		{
			ImageRGBA& frameRgba = outputFramesRgba.emplace_back(executor.mData.h, executor.mData.w);
			executor.getOutputImage(frameIndex, frameRgba);
			while (outputFramesRgba.size() > maxFrameCount) outputFramesRgba.pop_front();
		}

		{
			ImageBGRA& frameBgra = outputFramesBgra.emplace_back(executor.mData.h, executor.mData.w);
			executor.getOutputImage(frameIndex, frameBgra);
			while (outputFramesBgra.size() > maxFrameCount) outputFramesBgra.pop_front();
		}
	}
	this->frameIndex++;
}

void RawMemoryStoreWriter::writeInput(const FrameExecutor& executor) {
	results.push_back(executor.mFrame.mResultPoints);
	
	if (doWriteInput) {
		ImageYuv image(executor.mData.h, executor.mData.w);
		executor.getInput(inputFrameIndex, image);
		image.index = inputFrameIndex;
		inputFrames.push_back(image);
		while (inputFrames.size() > maxFrameCount) inputFrames.pop_front();
		this->inputFrameIndex++;
	}
}

void RawMemoryStoreWriter::writeYuvFiles(const std::string& inputFile, const std::string& outputFile) {
	std::ofstream osin(inputFile, std::ios::binary);
	for (ImageYuv& image : outputFramesYuv) {
		static ImageNV12 nv12(image.h, image.w, image.w);
		std::string str = std::format(" frame {:04} ", image.index);
		image.writeText(str, 0, image.h);
		image.toNV12(nv12);
		osin.write(reinterpret_cast<char*>(nv12.addr(0, 0, 0)), nv12.sizeInBytes());
	}

	std::ofstream osout(outputFile, std::ios::binary);
	for (ImageYuv& image : inputFrames) {
		static ImageNV12 nv12(image.h, image.w, image.w);
		std::string str = std::format(" frame {:04} ", image.index);
		image.writeText(str, 0, image.h);
		image.toNV12(nv12);
		osout.write(reinterpret_cast<char*>(nv12.addr(0, 0, 0)), nv12.sizeInBytes());
	}
}


//-----------------------------------------------------------------------------------
// Image Helpers
//-----------------------------------------------------------------------------------

std::string ImageWriter::makeFilename(const std::string& pattern, int64_t index, const std::string& extension) {
	namespace fs = std::filesystem;
	fs::path out;

	if (pattern.empty() == false && fs::is_directory(pattern)) {
		//file in the given directory
		std::string file = std::format("im{:04d}.{}", index, extension);
		out = fs::path(pattern) / fs::path(file);

	} else {
		//apply pattern as is
		const int siz = 512;
		char fname[siz];
		std::snprintf(fname, siz, pattern.c_str(), index);
		out = fs::path(fname);
	}
	return out.make_preferred().string();
}

std::string ImageWriter::makeFilenameSamples(const std::string& pattern, const std::string& extension) {
	std::string samples = "";
	std::string file = makeFilename(pattern, 0, extension);
	int idx = 0;
	while (samples.size() + file.size() < 100 && idx < 3) {
		samples += file;
		samples += ", ";
		idx++;
		file = makeFilename(pattern, idx, extension);
	}
	samples += "...";
	return samples;
}

std::string ImageWriter::makeFilename(const std::string& extension) const {
	return makeFilename(mData.fileOut, this->frameIndex, extension);
}


//-----------------------------------------------------------------------------------
// BMP Images
//-----------------------------------------------------------------------------------

void BmpImageWriter::writeOutput(const FrameExecutor& executor) {
	worker.join();
	executor.getOutputImage(frameIndex, image);
	std::string fname = makeFilename("bmp");
	worker = std::jthread([&, fname] { 
		image.saveAsColorBMP(fname);
		this->outputBytesWritten += 3ll * image.w * image.h;
		this->encodedBytesTotal += std::filesystem::file_size(std::filesystem::path(fname));
	});
	this->frameIndex++;
}

void BmpImageWriter::close() {
	worker.join();
}


//-----------------------------------------------------------------------------------
// JPG Images via ffmpeg
//-----------------------------------------------------------------------------------

void JpegImageWriter::open(OutputOption outputOption) {
	const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
	ctx = avcodec_alloc_context3(codec);
	ctx->width = mData.w;
	ctx->height = mData.h;
	ctx->time_base = { 1, 1 };
	ctx->framerate = { 1, 1 };
	ctx->codec_type = AVMEDIA_TYPE_VIDEO;
	ctx->pix_fmt = AV_PIX_FMT_YUV444P;
	ctx->color_range = AVCOL_RANGE_JPEG;
	ctx->flags |= AV_CODEC_FLAG_QSCALE;
	ctx->global_quality = FF_QP2LAMBDA * mData.selectedCrf; //values for crf from 31 (worst) to 1 (best)
	int retval;
	retval = avcodec_open2(ctx, codec, NULL);
	if (retval < 0)
		throw AVException(av_make_error(retval, "cannot open mjpeg codec"));

	av_frame = av_frame_alloc();
	av_frame->format = ctx->pix_fmt;
	av_frame->width = mData.w;
	av_frame->height = mData.h;
	av_frame->quality = ctx->global_quality; //quality must be set both to AVContext and AVFrame

	av_frame->linesize[0] = outputFrame.stride;
	av_frame->linesize[1] = outputFrame.stride;
	av_frame->linesize[2] = outputFrame.stride;
	av_frame->data[0] = outputFrame.plane(0);
	av_frame->data[1] = outputFrame.plane(1);
	av_frame->data[2] = outputFrame.plane(2);

	packet = av_packet_alloc();
}

void JpegImageWriter::writeOutput(const FrameExecutor& executor) {
	executor.getOutputYuv(frameIndex, outputFrame);

	av_frame->pts = this->frameIndex;
	int result = avcodec_send_frame(ctx, av_frame);
	if (result < 0)
		errorLogger().logError(av_make_error(result, "error sending frame"), ErrorSource::WRITER);

	result = avcodec_receive_packet(ctx, packet);
	if (result < 0)
		errorLogger().logError(av_make_error(result, "error receiving packet"), ErrorSource::WRITER);

	std::string fname = makeFilename("jpg");
	std::ofstream file(fname, std::ios::binary);
	if (file)
		file.write(reinterpret_cast<char*>(packet->data), packet->size);
	else
		errorLogger().logError("error opening output file '" + fname + "'", ErrorSource::WRITER);

	this->outputBytesWritten += packet->size;
	av_packet_unref(packet);

	this->encodedBytesTotal += std::filesystem::file_size(std::filesystem::path(fname));
	this->frameIndex++;
}

JpegImageWriter::~JpegImageWriter() {
	av_packet_free(&packet);
	avcodec_free_context(&ctx);
}


//-----------------------------------------------------------------------------------
// NV12 raw video
//-----------------------------------------------------------------------------------

//open file
void RawNv12Writer::open(OutputOption outputOption) {
	file = std::ofstream(mData.fileOut, std::ios::binary);
	nv12 = ImageNV12(mData.h, mData.w, mData.w);
}

void RawNv12Writer::writeOutput(const FrameExecutor& executor) {
	executor.getOutputYuv(frameIndex, outputFrame);
	outputFrame.toNV12(nv12, executor.mPool);
	file.write(reinterpret_cast<const char*>(nv12.data()), nv12.sizeInBytes());

	std::unique_lock<std::mutex> lock(mStatsMutex);
	this->outputBytesWritten += nv12.sizeInBytes();
	this->encodedBytesTotal = outputBytesWritten;
	this->frameIndex++;
}


//-----------------------------------------------------------------------------------
// YUV444 packed without striding pixels
//-----------------------------------------------------------------------------------

//open file
void RawYuvWriter::open(OutputOption outputOption) {
	file = std::ofstream(mData.fileOut, std::ios::binary);
}

void RawYuvWriter::writeOutput(const FrameExecutor& executor) {
	executor.getOutputYuv(frameIndex, outputFrame);
	file.write(reinterpret_cast<const char*>(outputFrame.data()), outputFrame.sizeInBytes());

	std::unique_lock<std::mutex> lock(mStatsMutex);
	this->outputBytesWritten += 3ll * outputFrame.h * outputFrame.w;
	this->encodedBytesTotal = outputBytesWritten;
	this->frameIndex++;
}


//-----------------------------------------------------
// Write raw data to Pipe
//-----------------------------------------------------

void RawPipeWriter::open(OutputOption outputOption) {
	PipeWriter::openPipe();
}

//get data via cuvista ... | ffmpeg -f rawvideo -pix_fmt yuv444p -video_size ww:hh -r xx -i pipe:0 -pix_fmt yuv420p outfile.mp4
void RawPipeWriter::writeOutput(const FrameExecutor& executor) {
	executor.getOutputYuv(frameIndex, outputFrame);

	size_t bytes = fwrite(outputFrame.data(), 1, outputFrame.sizeInBytes(), stdout);
	if (bytes != outputFrame.sizeInBytes()) {
		errorLogger().logError("Pipe: error writing data", ErrorSource::WRITER);
	}

	std::unique_lock<std::mutex> lock(mStatsMutex);
	this->outputBytesWritten += bytes;
	this->encodedBytesTotal = outputBytesWritten;
	this->frameIndex++;
}

void RawPipeWriter::close() {
	fflush(stdout);
	PipeWriter::closePipe();
}


//-----------------------------------------------------
// Write asf data to Pipe
//-----------------------------------------------------

void AsfPipeWriter::open(OutputOption outputOption) {
	PipeWriter::openPipe();
	av_log_set_callback(ffmpeg_log);

	//setup output context
	const AVOutputFormat* ofmt = av_guess_format("asf", NULL, NULL);
	AVFormatContext* fmt = nullptr;
	int result = avformat_alloc_output_context2(&fmt, ofmt, NULL, NULL);
	if (result < 0)
		throw AVException(av_make_error(result, "cannot allocate output format"));

	//custom avio
	int bufsiz = 4 * mData.h * mData.cpupitch;
	mBuffer = (unsigned char*) av_malloc(bufsiz);
	av_avio = avio_alloc_context(mBuffer, bufsiz, 1, this, nullptr, &AsfPipeWriter::writeBuffer, nullptr);
	av_avio->seekable = 0; //no seek allowed
	fmt->pb = av_avio;
	fmt->flags |= AVFMT_FLAG_CUSTOM_IO;

	//open ffmpeg
	AVCodecID id = AV_CODEC_ID_FFVHUFF;
	FFmpegFormatWriter::openFormat(id, fmt, 1);
	FFmpegWriter::open({}, id, AV_PIX_FMT_YUV444P, mData.h, mData.w, mData.cpupitch);

	//allocate yuv frame based on av_Frame
	outputFrame = ImageYuv(mData.h, mData.w, mData.cpupitch);
}

//for ffmpeg 7
int AsfPipeWriter::writeBuffer(void* opaque, unsigned char* buf, int siz) {
	return writeBuffer(opaque, (const unsigned char*) buf, siz);
}

//for ffmpeg 8
int AsfPipeWriter::writeBuffer(void* opaque, const unsigned char* buf, int siz) {
	//AsfPipeWriter* ptr = static_cast<AsfPipeWriter*>(opaque);
	size_t bytes = fwrite(buf, 1, siz, stdout);
	if (bytes != siz) {
		errorLogger().logError("Pipe: error writing data", ErrorSource::WRITER);
	}
	return (int) bytes;
}

void AsfPipeWriter::writeOutput(const FrameExecutor& executor) {
	executor.getOutputYuv(frameIndex, outputFrame);
	
	av_frame->data[0] = outputFrame.plane(0);
	av_frame->data[1] = outputFrame.plane(1);
	av_frame->data[2] = outputFrame.plane(2);
	av_frame->linesize[0] = outputFrame.stride;
	av_frame->linesize[1] = outputFrame.stride;
	av_frame->linesize[2] = outputFrame.stride;
	av_frame->pts = frameIndex;

	//generate and write packet through ffmpeg writer
	int result = sendFFmpegFrame(av_frame);
	while (result >= 0) {
		result = writeFFmpegPacket(av_frame);
	}

	this->frameIndex++;
}

void AsfPipeWriter::close() {
	FFmpegFormatWriter::close();
	fflush(stdout);
	PipeWriter::closePipe();
}

AsfPipeWriter::~AsfPipeWriter() {
	avio_context_free(&av_avio);
}


//-----------------------------------------------------------------------------------
// Computed Transformation Values
//-----------------------------------------------------------------------------------

std::map<int64_t, TransformValues> TransformsFile::readTransformMap(const std::string& trajectoryFile) {
	std::map<int64_t, TransformValues> transformsMap;
	std::ifstream file(trajectoryFile, std::ios::binary);
	if (file.is_open()) {
		//read and check signature
		std::string str = "    ";
		file.get(str.data(), 5);
		if (str != id) {
			errorLogger().logError("transforms file '" + trajectoryFile + "' is not valid", ErrorSource::WRITER);

		} else {
			while (!file.eof()) {
				int64_t frameIdx = 0;
				double s = 0, dx = 0, dy = 0, da = 0;
				file.read(reinterpret_cast<char*>(&frameIdx), sizeof(frameIdx));
				file.read(reinterpret_cast<char*>(&s), sizeof(s));
				file.read(reinterpret_cast<char*>(&dx), sizeof(dx));
				file.read(reinterpret_cast<char*>(&dy), sizeof(dy));
				file.read(reinterpret_cast<char*>(&da), sizeof(da));

				transformsMap[frameIdx] = { s, dx, dy, da / 60.0 * std::numbers::pi / 180.0 };
			}
		}

	} else {
		errorLogger().logError("cannot open transforms file '" + trajectoryFile + "'", ErrorSource::WRITER);
	}
	return transformsMap;
}

void TransformsFile::open(const std::string& trajectoryFile) {
	file = std::ofstream(trajectoryFile, std::ios::binary);
	if (file.is_open()) {
		//write signature
		file << id;

	} else {
		throw AVException("error opening outout file '" + trajectoryFile + "'");
	}
}

void TransformsFile::writeTransform(const Affine2D& transform, int64_t frameIndex) {
	writeValue(frameIndex);
	writeValue(transform.scale());
	writeValue(transform.dX());
	writeValue(transform.dY());
	writeValue(transform.rotMinutes());
}

void TransformsWriter::start() {
	TransformsFile::open(mData.trajectoryFile);
}

void TransformsWriter::writeInput(const FrameExecutor& executor) {
	writeTransform(executor.mFrame.mFrameResult.getTransform(), frameIndex);
	this->frameIndex++;
}


//-----------------------------------------------------------------------------------
// Optical Flow Video
//-----------------------------------------------------------------------------------

OpticalFlowWriter::OpticalFlowWriter(MainData& data, MovieReader& reader) :
	FFmpegWriter(data, reader, 1),
	imageInterpolated(data.h, data.w),
	imageResults(data.iyCount, data.ixCount)
{}

void OpticalFlowWriter::vectorToColor(double dx, double dy, unsigned char* r, unsigned char* g, unsigned char* b) {
	const double f = 20.0;
	double hue = std::atan2(dy, dx) / std::numbers::pi * 180.0 + 180.0;
	double val = std::clamp(std::sqrt(dx * dx + dy * dy) / f, 0.0, 1.0);
	im::hsv_to_rgb(hue, 1.0, val, r, g, b);
}

void OpticalFlowWriter::start() {
	start(mData.fileOut, AV_PIX_FMT_YUV420P);
}

void OpticalFlowWriter::start(const std::string& sourceName, AVPixelFormat pixfmt) {
	//simplified ffmpeg video setup
	av_log_set_callback(ffmpeg_log);

	//setup output context
	int result = avformat_alloc_output_context2(&fmt_ctx, NULL, NULL, sourceName.c_str());
	if (result < 0)
		throw AVException(av_make_error(result));

	result = avio_open(&fmt_ctx->pb, fmt_ctx->url, AVIO_FLAG_WRITE);
	if (result < 0)
		throw AVException("error opening output file '" + mData.fileOut + "'");

	for (StreamContext& sc : mReader.mInputStreams) {
		auto osc = std::make_shared<OutputStreamContext>();
		osc->inputStream = sc.inputStream;

		if (sc.inputStream->index == mReader.videoStream->index) {
			osc->handling = StreamHandling::STREAM_STABILIZE;
			osc->outputStream = videoStream = createNewStream(fmt_ctx, sc.inputStream);

		} else {
			osc->handling = StreamHandling::STREAM_IGNORE;
		}

		outputStreams.push_back(osc);
		sc.outputStreams.push_back(osc);
	}

	FFmpegWriter::open({}, AV_CODEC_ID_H264, pixfmt, mData.h, mData.w, mData.cpupitch);

	//setup scaler to accept RGB
	sws_scaler_ctx = sws_getContext(mData.w, mData.h, AV_PIX_FMT_RGBA, mData.w, mData.h, pixfmt, SWS_BILINEAR, NULL, NULL, NULL);
	if (!sws_scaler_ctx)
		throw AVException("Could not get scaler context");

	//setup legend image showing flow colors
	legendScale = std::max(1, std::min(mData.w, mData.h) / 512);
	legendSize = legendSizeBase * legendScale;
	legend = ImageRGBA(legendSize, legendSize);

	double r = legendSize / 2.0;
	for (int y = 0; y < legendSize; y++) {
		double dy = r - y;
		for (int x = 0; x < legendSize; x++) {
			double dx = r - x;
			double dist = std::sqrt(dx * dx + dy * dy);
			if (dist < r) {
				legend.at(3, y, x) = 0xFF;
				vectorToColor(dx, -dy, legend.addr(0, y, x), legend.addr(1, y, x), legend.addr(2, y, x));

			} else {
				legend.at(3, y, x) = 0;
			}
		}
	}
}

void OpticalFlowWriter::writeFlow(const MovieFrame& frame) {
	std::vector<unsigned char> colorGray = { 128, 128, 128, 255 };

	//convert vectors to hsv color values and then to rgb planar image
	std::vector<PointResult> results = frame.mResultPoints;
	for (const PointResult& pr : results) {
		if (pr.isValid()) {
			vectorToColor(-pr.u, pr.v, imageResults.addr(0, pr.iy0, pr.ix0), imageResults.addr(1, pr.iy0, pr.ix0), imageResults.addr(2, pr.iy0, pr.ix0));

		} else {
			//invalid result is gray pixel
			imageResults.setPixel(pr.iy0, pr.ix0, colorGray);
		}
	}

	//interpolate image
	auto fcn = [&] (size_t iy) {
		int div = 1 << mData.zMax;
		double y = 1.0 * iy / div - 1.0 - mData.ir;
		for (size_t ix = 0; ix < mData.w; ix++) {
			double x = 1.0 * ix / div - 1.0 - mData.ir;
			for (int i = 0; i < 3; i++) {
				unsigned char col = colorGray.at(i);
				if (x >= -0.5 && x < mData.ixCount + 0.5 && y >= -0.5 && y < mData.iyCount + 0.5) col = imageResults.sample(i, x, y);
				imageInterpolated.at(i, iy, ix) = col;
			}
		}
	};
	frame.mPool.addAndWait(fcn, 0, mData.h);
}

void OpticalFlowWriter::writeAVFrame(AVFrame* av_frame) {
	int result = avcodec_send_frame(codec_ctx, av_frame);
	if (result < 0)
		ffmpeg_log_error(result, "error encoding flow #1", ErrorSource::WRITER);

	while (result >= 0) {
		result = avcodec_receive_packet(codec_ctx, videoPacket);
		if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) {
			//do not report error here, need more frame data or end of file

		} else if (result < 0) {
			//report error, something wrong
			ffmpeg_log_error(result, "error encoding flow #2", ErrorSource::WRITER);

		} else {
			av_packet_rescale_ts(videoPacket, codec_ctx->time_base, videoStream->time_base);
			int siz = videoPacket->size;
			result = av_interleaved_write_frame(fmt_ctx, videoPacket);
			if (result < 0) {
				ffmpeg_log_error(result, "error writing flow packet", ErrorSource::WRITER);

			} else {
				encodedBytesTotal += siz;
			}

			std::unique_lock<std::mutex> lock(mStatsMutex);
			outputBytesWritten = avio_tell(fmt_ctx->pb);
			frameEncoded++;
		}
	}
}

void OpticalFlowWriter::writeInput(const FrameExecutor& executor) {
	writeFlow(executor.mFrame);
	//imageInterpolated.saveAsBMP("f:/im.bmp");

	//stamp color legend onto image
	legend.copyTo(imageInterpolated, 0ull + mData.h - 10 - legendSize, 10);
	int tx = 10 + legendSize / 2;
	int ty = 0ull + mData.h - 10 - legendSize / 2;
	imageInterpolated.writeText(std::to_string(frameIndex), tx, ty, legendScale, legendScale, im::TextAlign::MIDDLE_CENTER);

	//encode rgba image
	uint8_t* src[] = { imageInterpolated.data(), nullptr, nullptr, nullptr };
	int strides[] = { imageInterpolated.stride, 0, 0, 0 };
	int sliceHeight = sws_scale(sws_scaler_ctx, src, strides, 0, imageInterpolated.h, av_frame->data, av_frame->linesize);

	av_frame->pts = frameIndex;
	writeAVFrame(av_frame);
	frameIndex++;
}

void OpticalFlowWriter::writeOutput(const FrameExecutor& executor) {}

bool OpticalFlowWriter::flush() {
	writeAVFrame(nullptr);
	return false;
}


//-----------------------------------------------------------------------------------
// Computed Results per Point
//-----------------------------------------------------------------------------------

void  ResultDetailsWriter::start() {
	mFile = std::ofstream(mData.resultsFile);
	if (mFile.is_open()) {
		mFile << "frameIdx" << mDelim << "ix0" << mDelim << "iy0"
			<< mDelim << "x" << mDelim << "y" << mDelim << "u" << mDelim << "v"
			<< mDelim << "isValid" << mDelim << "isConsens" << mDelim << "direction" << std::endl;

	} else {
		throw AVException("cannot open file '" + mData.resultsFile + "'");
	}
}

void ResultDetailsWriter::write(std::span<PointResult> results, int64_t frameIndex) {
	//for better performace first write into buffer string
	std::stringstream ss;
	for (auto& item : results) {
		ss << frameIndex << mDelim << item.ix0 << mDelim << item.iy0 
			<< mDelim << item.x << mDelim << item.y << mDelim << item.u << mDelim << item.v 
			<< mDelim << item.resultValue() << mDelim << item.isConsens << mDelim << item.direction << std::endl;
	}
	//write buffer to file
	mFile << ss.str();
}

void ResultDetailsWriter::writeInput(const FrameExecutor& executor) {
	write(executor.mFrame.mResultPoints, frameIndex);
	this->frameIndex++;
}

void ResultDetailsWriter::write(std::span<PointResult> results, const std::string& filename) {
	MainData data;
	data.resultsFile = filename;

	ResultDetailsWriter writer(data);
	writer.start();
	writer.write(results, 0);
}


//-----------------------------------------------------------------------------------
// Result Images
//-----------------------------------------------------------------------------------

void ResultImageWriter::writeImage(const AffineTransform& trf, std::span<PointResult> res, int64_t idx, const ImageYuv& yuv, ThreadPoolBase& pool) {
	using namespace im;

	//copy Y plane of YUV to all planes in bgr image making it grayscale bgr
	for (int z = 0; z < 3; z++) {
		for (int r = 0; r < bgr.h; r++) {
			for (int c = 0; c < bgr.w; c++) {
				bgr.at(z, r, c) = yuv.at(0, r, c);
			}
		}
	}

	//draw lines
	//draw blue lines first
	auto func1 = [&] (size_t idx) {
		const PointResult& pr = res[idx];
		if (pr.isConsidered) {
			double px = pr.x + mData.w / 2.0;
			double py = pr.y + mData.h / 2.0;
			double x2 = px + pr.u;
			double y2 = py + pr.v;

			//blue line to computed transformation
			auto [tx, ty] = trf.transform(pr.x, pr.y);
			bgr.drawLine(px, py, tx + bgr.w / 2.0, ty + bgr.h / 2.0, Color::BLUE, 0.5);
		}
	};
	pool.addAndWait(func1, 0, res.size());

	//draw on top
	//green line if point is consens
	//red line if point is not consens
	int numConsidered = 0, numConsens = 0;
	Color col;
	for (const PointResult& pr : res) {
		if (pr.isConsidered) {
			numConsidered++;
			double px = pr.x + mData.w / 2.0;
			double py = pr.y + mData.h / 2.0;
			double x2 = px + pr.u;
			double y2 = py + pr.v;

			if (pr.isConsens) {
				col = Color::GREEN;
				numConsens++;

			} else {
				col = Color::RED;
			}
			bgr.drawLine(px, py, x2, y2, col);
			bgr.drawMarker(x2, y2, col, 1.4);
		}
	}

	//write text info
	double frac = numConsidered == 0 ? 0.0 : 100.0 * numConsens / numConsidered;
	std::string s2 = std::format("transform dx={:.1f}, dy={:.1f}, scale={:.5f}, rot={:.1f}", trf.dX(), trf.dY(), trf.scale(), trf.rotMinutes());
	Size s = bgr.writeText(s2, 0, bgr.h);
	std::string s1 = std::format("index {}, consensus {}/{} ({:.1f}%)", idx, numConsens, numConsidered, frac);
	bgr.writeText(s1, 0, bgr.h - s.h);
}

void ResultImageWriter::writeInput(const FrameExecutor& executor) {
	//get input image from buffers
	executor.getInput(frameIndex, yuv);
	std::string fname = ImageWriter::makeFilename(mData.fileOut, frameIndex, "bmp");
	writeImage(executor.mFrame.getTransform(), executor.mFrame.mResultPoints, frameIndex, yuv, executor.mPool);

	//save image to file
	bool result = bgr.saveAsColorBMP(fname);
	if (result == false) {
		errorLogger().logError("cannot write file '" + fname + "'", ErrorSource::WRITER);
	}

	frameIndex++;
	encodedBytesTotal += 3ll * mData.h * mData.w;
	outputBytesWritten += std::filesystem::file_size(std::filesystem::path(fname));
}

void ResultImageWriter::writeImage(const AffineTransform& trf, std::span<PointResult> res, int64_t idx, const ImageYuv& yuv, const std::string& outFile) {
	writeImage(trf, res, idx, yuv, ThreadPool::defaultPool);
	bgr.saveAsColorBMP(outFile);
}
