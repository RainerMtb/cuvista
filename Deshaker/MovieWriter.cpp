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
#include "MovieFrame.hpp"
#include "ErrorLogger.hpp"
#include <filesystem>


SimpleYuvWriter::SimpleYuvWriter(const std::string& file) {
	os = std::ofstream(file, std::ios::binary);
}

void SimpleYuvWriter::write(ImageVuyx& image) {
	const unsigned char* src = image.data();
	for (int i = 0; i < image.h(); i++) {
		os.write(reinterpret_cast<const char*>(src), image.w() * 4ull);
		src += image.stride();
	}
}


//-----------------------------------------------------------------------------------
// Writer Collection
//-----------------------------------------------------------------------------------

MovieWriterCollection::MovieWriterCollection(MainData& data, MovieReader& reader, std::vector<std::shared_ptr<MovieWriter>> writers) :
	NullWriter(data, reader),
	writers { writers },
	hasFrames(writers.size())
{
	if (writers.empty()) {
		throw AVException("no output was specified");
	}
}

void MovieWriterCollection::open(OutputOption outputOption) {
	for (auto& writer : writers) writer->open(outputOption);
}

void MovieWriterCollection::start() {
	for (auto& writer : writers) writer->start();
}

void MovieWriterCollection::updateStats() {
	//take values from first writer
	std::shared_ptr<MovieWriter> mainWriter = writers.front();
	frameIndex.store(mainWriter->frameIndex);
	frameEncoded.store(mainWriter->frameEncoded);
	encodedBytesTotal.store(mainWriter->encodedBytesTotal);
	outputBytesWritten.store(mainWriter->outputBytesWritten);
}

void MovieWriterCollection::writeInput(const FrameExecutor& executor) {
	for (auto& writer : writers) writer->writeInput(executor);
	updateStats();
}

void MovieWriterCollection::writeOutput(const FrameExecutor& executor) {
	for (auto& writer : writers) writer->writeOutput(executor);
	updateStats();
}

bool MovieWriterCollection::flush() {
	for (int i = 0; i < writers.size(); i++) {
		hasFrames[i] = writers[i]->flush();
	}
	updateStats();
	return std::accumulate(hasFrames.begin(), hasFrames.end(), 0) > 0;
}

void MovieWriterCollection::close() {
	for (auto& writer : writers) writer->close();
	updateStats();
}


//-----------------------------------------------------------------------------------
// primitive output writer class
//-----------------------------------------------------------------------------------

const ImageVuyx& OutputWriter::getOutputFrame() { 
	return outputFrame; 
}

void OutputWriter::writeOutput(const FrameExecutor& executor) {
	executor.getOutput(frameIndex, outputFrame);
	this->frameIndex++;
}


//-----------------------------------------------------------------------------------
// Raw Format Writers
//-----------------------------------------------------------------------------------

void RawMemoryStoreWriter::writeOutput(const FrameExecutor& executor) {
	if (doWriteOutput) {
		{
			ImageVuyx& frameYuv = outputFramesYuv.emplace_back(executor.mData.h, executor.mData.w);
			executor.getOutput(frameIndex, frameYuv);
			while (outputFramesYuv.size() > maxFrameCount) outputFramesYuv.pop_front();
		}

		{
			ImageRGBA& frameRgba = outputFramesRgba.emplace_back(executor.mData.h, executor.mData.w);
			executor.getOutput(frameIndex, frameRgba);
			while (outputFramesRgba.size() > maxFrameCount) outputFramesRgba.pop_front();
		}

		{
			ImageBGRA& frameBgra = outputFramesBgra.emplace_back(executor.mData.h, executor.mData.w);
			executor.getOutput(frameIndex, frameBgra);
			while (outputFramesBgra.size() > maxFrameCount) outputFramesBgra.pop_front();
		}
	}
	this->frameIndex++;
}

void RawMemoryStoreWriter::writeInput(const FrameExecutor& executor) {
	results.push_back(executor.mFrame.mResultPoints);
	
	if (doWriteInput) {
		ImageVuyx image(executor.mData.h, executor.mData.w);
		executor.getInput(inputFrameIndex, image);
		image.index = inputFrameIndex;
		inputFrames.push_back(image);
		while (inputFrames.size() > maxFrameCount) inputFrames.pop_front();
		this->inputFrameIndex++;
	}
}

void RawMemoryStoreWriter::writeYuvFiles(const std::string& inputFile, const std::string& outputFile, int maxFrames) {
	writeInputFile(inputFile, maxFrames);
	writeOutputFile(outputFile, maxFrames);
}

void RawMemoryStoreWriter::writeInputFile(const std::string& inputFile, int maxFrames, ThreadPoolBase& pool) {
	std::ofstream osin(inputFile, std::ios::binary);
	int idx = 0;
	for (ImageVuyx& image : inputFrames) {
		static ImageNV12 nv12(image.h(), image.w(), image.w());
		std::string str = std::format(" frame {:04} ", idx);
		image.writeText(str, 0, image.h());
		image.convertTo(nv12, pool);
		osin.write(reinterpret_cast<char*>(nv12.addr(0, 0, 0)), nv12.sizeInBytes());
		idx++;
		if (idx == maxFrames) break;
	}
}

void RawMemoryStoreWriter::writeOutputFile(const std::string& outputFile, int maxFrames, ThreadPoolBase& pool) {
	std::ofstream osout(outputFile, std::ios::binary);
	int idx = 0;
	for (ImageVuyx& image : outputFramesYuv) {
		static ImageNV12 nv12(image.h(), image.w(), image.w());
		std::string str = std::format(" frame {:04} ", idx);
		image.writeText(str, 0, image.h());
		image.convertTo(nv12, pool);
		osout.write(reinterpret_cast<char*>(nv12.addr(0, 0, 0)), nv12.sizeInBytes());
		idx++;
		if (idx == maxFrames) break;
	}
}


//-----------------------------------------------------------------------------------
// BMP Images
//-----------------------------------------------------------------------------------

void BmpImageWriter::writeOutput(const FrameExecutor& executor) {
	worker.join();
	executor.getOutput(frameIndex, imageBgra);
	std::string fname = makeFilename("bmp");
	worker = std::jthread([&, fname] {
		imageBgra.saveBmpColor(fname);
		this->outputBytesWritten += 3ull * mData.h * mData.w;
		this->encodedBytesTotal += std::filesystem::file_size(std::filesystem::path(fname));
	});
	this->frameIndex++;
}

void BmpImageWriter::close() {
	worker.join();
}


//-----------------------------------------------------------------------------------
// NV12 raw video
//-----------------------------------------------------------------------------------

//open file
void RawNv12Writer::open(OutputOption outputOption) {
	file = std::ofstream(mData.fileOut, std::ios::binary);
}

void RawNv12Writer::writeOutput(const FrameExecutor& executor) {
	executor.getOutput(frameIndex, nv12, 0, nullptr);

	for (int r = 0; r < nv12.rows(); r++) {
		uchar* ptr = nv12.row(r);
		file.write(reinterpret_cast<const char*>(ptr), nv12.w());
	}

	this->outputBytesWritten += nv12.rows() * nv12.cols();
	this->encodedBytesTotal.store(outputBytesWritten);
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
	executor.getOutput(frameIndex, yuv);

	for (int r = 0; r < yuv.rows(); r++) {
		uchar* ptr = yuv.row(r);
		file.write(reinterpret_cast<const char*>(ptr), yuv.cols());
	}

	this->outputBytesWritten += yuv.rows() * yuv.cols();
	this->encodedBytesTotal.store(outputBytesWritten);
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
	executor.getOutput(frameIndex, output);

	size_t bytes = 0;
	for (int r = 0; r < output.rows(); r++) {
		uchar* ptr = output.row(r);
		bytes += fwrite(ptr, 1, output.cols(), stdout);
	}
	if (bytes != output.rows() * output.cols()) {
		errorLogger().logError("Pipe: error writing data", ErrorSource::WRITER);
	}

	this->outputBytesWritten += bytes;
	this->encodedBytesTotal.store(outputBytesWritten);
	this->frameIndex++;
}

void RawPipeWriter::close() {
	fflush(stdout);
	PipeWriter::closePipe();
}


//-----------------------------------------------------------------------------------
// Computed Transformation Values
//-----------------------------------------------------------------------------------

std::map<int64_t, TransformValues> TransformsFile::readTransformMap(const std::string& trajectoryFile) {
	std::map<int64_t, TransformValues> transformsMap;
	std::ifstream infile(trajectoryFile, std::ios::binary);
	if (infile.is_open()) {
		//read and check signature
		std::string str = "    ";
		infile.get(str.data(), 5);
		if (str != id) {
			errorLogger().logError("transforms file '" + trajectoryFile + "' is not valid", ErrorSource::WRITER);

		} else {
			while (!infile.eof()) {
				int64_t frameIdx = 0;
				double s = 0, dx = 0, dy = 0, da = 0;
				infile.read(reinterpret_cast<char*>(&frameIdx), sizeof(frameIdx));
				infile.read(reinterpret_cast<char*>(&s), sizeof(s));
				infile.read(reinterpret_cast<char*>(&dx), sizeof(dx));
				infile.read(reinterpret_cast<char*>(&dy), sizeof(dy));
				infile.read(reinterpret_cast<char*>(&da), sizeof(da));

				transformsMap[frameIdx] = { s, dx, dy, da / 60.0 * std::numbers::pi / 180.0 };
			}
		}

	} else {
		errorLogger().logError("cannot open transforms file '" + trajectoryFile + "'", ErrorSource::WRITER);
	}
	return transformsMap;
}

void TransformsFile::open(const std::string& trajectoryFile) {
	mFile = std::ofstream(trajectoryFile, std::ios::binary);
	if (mFile.is_open()) {
		//write signature
		mFile << id;

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
	outputBytesWritten = mFile.tellp();
}

void TransformsWriter::writeInput(const FrameExecutor& executor) {
	writeTransform(executor.mFrame.mFrameResult.getTransform(), frameIndex);
	this->frameIndex++;
	outputBytesWritten = mFile.tellp();
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

	outputBytesWritten = mFile.tellp();
}

void ResultDetailsWriter::write(std::span<PointResult> results, int64_t frameIndex) {
	std::stringstream ss;
	for (auto& item : results) {
		ss << frameIndex << mDelim << item.ix0 << mDelim << item.iy0 
			<< mDelim << item.x << mDelim << item.y << mDelim << item.u << mDelim << item.v 
			<< mDelim << item.resultValue() << mDelim << item.isConsens << mDelim << item.direction << std::endl;
	}

	//write buffer to file
	std::string str = ss.str();
	mFile << ss.str();

	outputBytesWritten = mFile.tellp();
}

void ResultDetailsWriter::write(std::span<PointResult> results, const std::string& filename) {
	MainData data;
	data.resultsFile = filename;

	ResultDetailsWriter writer(data);
	writer.start();
	writer.write(results, 0);
}

void ResultDetailsWriter::writeInput(const FrameExecutor& executor) {
	write(executor.mFrame.mResultPoints, frameIndex);
	this->frameIndex++;
}


//-----------------------------------------------------------------------------------
// Result Images
//-----------------------------------------------------------------------------------

void ResultImageWriter::writeImage(const FrameResultData& resultData, std::span<PointResult> res, int64_t idx, Image8& dest, ThreadPoolBase& pool, bool drawTransformed) {
	int h = dest.h();
	int w = dest.w();
	Color col = Color::web("#F09B59");

	//draw transform indicator lines first
	if (drawTransformed) {
		for (int thridx = 0; thridx < pool.size(); thridx++) {
			auto func1 = [&] (size_t thridx) {
				for (size_t idx = thridx; idx < res.size(); idx += pool.size()) {
					const PointResult& pr = res[idx];
					if (pr.isConsidered) {
						double px = pr.x + w / 2.0;
						double py = pr.y + h / 2.0;
						double x2 = px + pr.u;
						double y2 = py + pr.v;

						//blue line to computed transformation
						auto [tx, ty] = resultData.transform.transform(pr.x, pr.y);
						dest.drawLine(px, py, tx + w / 2.0, ty + h / 2.0, col, 0.1);
					}
				}
			};
			pool.addAndWait(func1, 0, pool.size());
		}
	}

	//green line if point is consens, red line if point is not consens
	int numConsidered = 0;
	int numConsens = 0;
	std::mutex mutex;
	auto func1 = [&] (size_t thridx) {
		Color col;
		for (size_t idx = thridx; idx < res.size(); idx += pool.size()) {
			const PointResult& pr = res[idx];
			if (pr.isConsidered) {
				double px = pr.x + w / 2.0;
				double py = pr.y + h / 2.0;
				double x2 = px + pr.u;
				double y2 = py + pr.v;

				int delta = 0;
				if (pr.isConsens) {
					col = Color::GREEN;
					delta = 1;

				} else {
					col = Color::RED;
				}
				dest.drawLine(px, py, x2, y2, col);
				dest.drawMarker(x2, y2, col, 1.4);

				std::unique_lock<std::mutex> lock(mutex);
				numConsidered++;
				numConsens += delta;
			}
		}
	};
	pool.addAndWait(func1, 0, pool.size());

	//write text info
	double frac = numConsidered == 0 ? 0.0 : 100.0 * numConsens / numConsidered;
	const AffineTransform& trf = resultData.transform;
	std::string s2 = std::format(" transform dx={:.1f} px, dy={:.1f} px, scale={:.5f}, rot={:.5f} deg ", trf.dX(), trf.dY(), trf.scale(), trf.rotDegrees());
	Size s = dest.writeText(s2, 0, h);
	std::string s1 = std::format(" frame {}, consensus {}/{} ({:.1f}%) ", idx, numConsens, numConsidered, frac);
	dest.writeText(s1, 0, h - s.h);

	//cluster info
	auto sizes = resultData.getClusterSizes();
	if (sizes.size() > 0) {
		std::string str = std::format(" clusters: {} {} ", sizes.size(), util::collectionToString(sizes, 10));
		dest.writeText(str, 0, 0, TextAlign::TOP_LEFT);
	}

	//adjusted gamma
	if (resultData.gamma > 0.0) {
		std::string str = std::format(" gamma={:.3f} ", resultData.gamma);
		dest.writeText(str, 0, h - 2 * s.h);
	}
}

void ResultImageWriter::writeInput(const FrameExecutor& executor) {
	//get input image from buffers
	executor.getInput(frameIndex, bgra);
	bgra.gray();
	std::string fname = ImageWriter::makeFilename(mData.fileOut, frameIndex, "bmp");
	writeImage(executor.mFrame.getResultData(), executor.mFrame.mResultPoints, frameIndex, bgra, executor.mPool);

	//save image to file
	bgra.saveBmpColor(fname);

	this->outputBytesWritten += std::filesystem::file_size(std::filesystem::path(fname));
	this->encodedBytesTotal += 3ll * mData.h * mData.w;
	this->frameIndex++;
}


//-----------------------------------------------------------------------------------
// Result Video
//-----------------------------------------------------------------------------------

void ResultVideoWriter::open(OutputOption outputOption) {
	file = std::ofstream(mData.fileOut, std::ios::binary);
	bgra = ImageBGRA(mData.h, mData.w);
	nv12 = ImageNV12(mData.h, mData.w, mData.w);
}

void ResultVideoWriter::writeInput(const FrameExecutor& executor) {
	executor.getInput(frameIndex, bgra);
	bgra.gray(executor.mPool);
	ResultImageWriter::writeImage(executor.mFrame.getResultData(), executor.mFrame.mResultPoints, frameIndex, bgra, executor.mPool, false);
	bgra.convertTo(nv12, executor.mPool);
	file.write(reinterpret_cast<const char*>(nv12.data()), nv12.sizeInBytes());

	this->outputBytesWritten += nv12.sizeInBytes();
	this->encodedBytesTotal.store(outputBytesWritten);
	this->frameIndex++;
}
