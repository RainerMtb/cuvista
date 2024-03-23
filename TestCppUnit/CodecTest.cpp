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

#include "pch.h"
#include "CppUnitTest.h"
#include "NvEncoder.hpp"
#include "Image2.hpp"
#include "FFmpegUtil.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace CudaTest {

	TEST_CLASS(CodecTest) {

private:

	std::wstring toWString(const CUresult& result) {
		const char* pstr = nullptr;
		cuGetErrorString(result, &pstr);
		std::string str(pstr);
		return std::wstring(str.cbegin(), str.cend());
	}

	//encode sample image through cuda
	//decode and compare with original image
	void runEncoder(int w, int h, uint8_t y, uint8_t u, uint8_t v, const std::string& msg) {
		CUresult res;

		//set up cuda encoder
		NvEncoder nvenc(w, h);

		uint8_t crf = 10;
		nvenc.createEncoder(10, 1, 5, crf, NV_ENC_CODEC_HEVC_GUID, 0);

		//set up a frame in nv12 format
		ImageYuv inputFrame(h, w, nvenc.cudaPitch);
		inputFrame.setValues({ y, u, v });
		inputFrame.writeText("test", 20, 20, 5, 5, ColorYuv::BLACK, ColorYuv::WHITE);

		size_t siz = h * nvenc.cudaPitch * 3 / 2;
		std::vector<unsigned char> inputNV12(siz);
		inputFrame.toNV12(inputNV12, nvenc.cudaPitch);
		CUdeviceptr devptr = (CUdeviceptr) nvenc.getNextInputFramePtr();
		res = cuMemcpyHtoD_v2(devptr, inputNV12.data(), siz);
		Assert::IsTrue(res == CUDA_SUCCESS, toWString(res).c_str());

		//encode into packet
		std::list<NvPacket> packets;
		nvenc.encodeFrame(packets);
		nvenc.endEncode();
		packets.push_back(nvenc.getBufferedFrame());
		Assert::IsTrue(packets.size() == 1); //only one packet
		std::vector<uint8_t> encPacket = packets.front().packet;
		Assert::IsTrue(encPacket.size() > 100); //packet must be some bytes long

		//std::ofstream file("f:/codec.h264", std::ios::binary);
		//file.write(reinterpret_cast<char*>(encPacket.data()), encPacket.size());

		//set up decoder
		const AVCodec* dec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
		Assert::IsNotNull(dec);
		AVCodecContext* ctx = avcodec_alloc_context3(dec);
		Assert::IsNotNull(ctx);
		int retval = avcodec_open2(ctx, dec, NULL);
		Assert::IsTrue(retval == 0);
		AVPacket* pkt = av_packet_alloc();
		Assert::IsNotNull(pkt);
		AVFrame* av_frame = av_frame_alloc();
		Assert::IsNotNull(av_frame);

		//decode packet into ffmpeg frame in format YUV420
		pkt->data = packets.front().packet.data();
		pkt->size = (int) packets.front().packet.size();
		retval = avcodec_send_packet(ctx, pkt);
		Assert::IsTrue(retval == 0);
		retval = avcodec_send_packet(ctx, NULL);
		Assert::IsTrue(retval == 0);
		retval = avcodec_receive_frame(ctx, av_frame);
		Assert::IsTrue(retval == 0);

		//scale to YUV444
		ImageYuv outputFrame(h, w, nvenc.cudaPitch);
		SwsContext* scaler = sws_getContext(w, h, ctx->pix_fmt, w, h, AV_PIX_FMT_YUV444P, SWS_BILINEAR, NULL, NULL, NULL);
		uint8_t* frame_buffer[] = {(uint8_t*) outputFrame.plane(0), (uint8_t*) outputFrame.plane(1), (uint8_t*) outputFrame.plane(2), nullptr};
		int stride = (int) nvenc.cudaPitch;
		int linesizes[] = { stride, stride, stride, 0 };
		sws_scale(scaler, av_frame->data, av_frame->linesize, 0, av_frame->height, frame_buffer, linesizes);

		//inputFrame.writeToBMPcolor("f:/test_in.bmp");
		//outputFrame.writeToBMPcolor("f:/test_out.bmp");

		//compare input and output
		double avg = inputFrame.compareTo(outputFrame);
		std::wstring err = L"difference " + ToString(msg) + L"=" + ToString(avg);
		Assert::IsTrue(avg < 0.01, err.c_str());

		//tear down encoder and decoder
		nvenc.destroyEncoder();
		sws_freeContext(scaler);
		avcodec_free_context(&ctx);
		av_frame_free(&av_frame);
		av_packet_free(&pkt);
	}

public:
	TEST_METHOD(cuEncodeDecode) {
		CUresult res;
		res = cuInit(0);
		Assert::IsTrue(res == CUDA_SUCCESS);

		runEncoder(1920, 1080, 130, 130, 80, "A");
		runEncoder(1920, 1080, 40, 75, 100, "B");
		runEncoder(600, 400, 130, 130, 130, "C");
		runEncoder(6144, 6144, 130, 130, 130, "D");
	}

	};
}