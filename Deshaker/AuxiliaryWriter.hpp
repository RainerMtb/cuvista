#pragma once

#include "MainData.hpp"
#include "Trajectory.hpp"
#include "Stats.hpp"

class MovieFrame;

//-------------- superclass for secondary writers -----------------------------------
class AuxiliaryWriter {

protected:
	const MainData& mAuxData; //also contained in MovieWriter

public:
	int64_t frameIndex = 0;

	AuxiliaryWriter(MainData& data) :
		mAuxData { data } {}

	virtual void open() = 0;
	virtual void write(const MovieFrame& frame) = 0;
	virtual ~AuxiliaryWriter() = default;
};

//--------------- write point results as large text file ----------------------------
class ResultDetailsWriter : public AuxiliaryWriter {

private:
	std::string delimiter = ";";
	std::ofstream file;
	void write(const std::vector<PointResult>& results, int64_t frameIndex);

public:
	ResultDetailsWriter(MainData& data) :
		AuxiliaryWriter(data) {}

	virtual void open() override;
	virtual void write(const MovieFrame& frame) override;
};


//--------------- write individual images to show point results ---------------------
class ResultImageWriter : public AuxiliaryWriter {

private:
	ImageBGR bgr;
	void write(const FrameResult& fr, int64_t idx, const ImageYuv& yuv, const std::string& fname);

public:
	ResultImageWriter(MainData& data) :
		AuxiliaryWriter(data),
		bgr(data.h, data.w) {}

	virtual void open() override {}
	virtual void write(const MovieFrame& frame) override;
};


//--------------- collection of auxiliary writer instances
class AuxWriters : public std::vector<std::unique_ptr<AuxiliaryWriter>> {

public:
	void writeAll(const MovieFrame& frame);
};