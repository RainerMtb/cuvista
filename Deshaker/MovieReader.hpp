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

#include "Stats.hpp"
#include "FFmpegUtil.hpp"
#include "FrameExecutor.hpp"

#include <optional>
#include <span>
#include <mutex>
#include <future>

//-----------------------------------------------------------------------------------
// reader must increment frame counter when beeing called to read
//-----------------------------------------------------------------------------------

struct StreamContext;
struct OutputStreamContext;
namespace im { class Image8; }

class MovieReader : public ReaderStats {

public:
	std::mutex mVideoPacketMutex;
	std::list<VideoPacketContext> mVideoPacketList;
	bool mStoreSidePackets = true;
	std::string mSource;

	virtual ~MovieReader() = default;

	virtual void open(const std::string& source) = 0;
	virtual void open(std::span<unsigned char> movieData) {}
	virtual void start() {}
	virtual bool read(im::Image8& inputFrame) = 0;
	virtual bool read(FrameExecutor& executor) { return false; };
	virtual std::future<void> readAsync(FrameExecutor& executor);
	virtual void close() {}

	virtual void rewind() {}
	virtual bool seek(double fraction) { return true; }
	virtual int openAudioDecoder(std::shared_ptr<OutputStreamContextBase> sposc) { return -1; }

	virtual size_t inputStreamCount() const { return 0; }
	virtual std::shared_ptr<StreamContext> inputStream(size_t index) const { return nullptr; }
	virtual std::shared_ptr<StreamContextBase> inputStreamBase(size_t index) const { return nullptr; }

	std::optional<std::string> ptsForFrameAsString(int64_t frameIndex);
	std::optional<int64_t> ptsForFrameAsMillis(int64_t frameIndex);
	double ptsForFrame(int64_t frameIndex);
	std::string videoStreamSummary() const;

protected:
	int sideDataMaxSize = 20 * 1024 * 1024;
};
