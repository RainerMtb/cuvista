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

#include "AVException.hpp"
#include "Stats.hpp"
#include "Image2.hpp"
#include <optional>

class MovieReader : public ReaderStats {

public:
	std::list<VideoPacketContext> packetList;
	std::vector<StreamContext> inputStreams;
	bool storePackets = true;

	virtual ~MovieReader() = default;

	virtual void open(std::string_view source) = 0;
	virtual void read(ImageYuv& inputFrame) = 0;
	virtual std::future<void> readAsync(ImageYuv& inputFrame);
	virtual void close() {}
	virtual void rewind() {}
	virtual void seek(double fraction) {}

	std::optional<std::string> dtsForFrameString(int64_t frameIndex);
	std::optional<int64_t> dtsForFrameMillis(int64_t frameIndex);
};

//main class to decode input
class FFmpegReader : public MovieReader {

private:
	AVFormatContext* av_format_ctx = nullptr;
	AVCodecContext* av_codec_ctx = nullptr;
	AVFrame* av_frame = nullptr;
	AVPacket* av_packet = nullptr;
	SwsContext* sws_scaler_ctx = nullptr;

public:
	~FFmpegReader() override;
	void open(std::string_view source) override;

	void read(ImageYuv& frame) override;
	void close() override;
	void rewind() override;
	void seek(double fraction);
};


class NullReader : public MovieReader {

public:
	void open(std::string_view source) override {};
	void read(ImageYuv& frame) override;
};