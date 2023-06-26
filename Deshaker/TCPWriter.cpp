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
#include <ws2tcpip.h>
#include <WinSock2.h>

 //undef conflicting macro
#undef max
#undef min

struct Sockets {
	SOCKET mSock {};
	SOCKET mConn {};
};

TCPWriter::TCPWriter(MainData& data) : RawWriter(data) {
	sockets = std::make_unique<Sockets>();
}

void TCPWriter::open(OutputCodec videoCodec) {
	WSADATA wsaData {};
	int retval = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (retval < 0)
		throw AVException("cannot start TCP");

	sockets->mSock = socket(AF_INET, SOCK_STREAM, 0);
	if (sockets->mSock < 0)
		throw AVException("cannot create TCP socket");

	sockaddr_in sockAddr;
	sockaddr* sockaddr_ptr = (sockaddr*) &sockAddr;
	int addrSize = sizeof(sockAddr);
	memset(&sockAddr, 0, addrSize);
	sockAddr.sin_family = AF_INET;
	inet_pton(AF_INET, mData.tcp_address.c_str(), &sockAddr.sin_addr.s_addr);
	sockAddr.sin_port = htons(mData.tcp_port);

	bind(sockets->mSock, sockaddr_ptr, addrSize);
	listen(sockets->mSock, 3);
	*mData.console << "listening for TCP connection at " << mData.tcp_address << ":" << mData.tcp_port << std::endl;

	//blocking call, wait for connecting client
	sockets->mConn = accept(sockets->mSock, sockaddr_ptr, &addrSize);
	if (sockets->mConn < 0) {
		throw AVException("cannot connect to TCP");
	}

	*mData.console << "established TCP connection" << std::endl;
}

void TCPWriter::write() {
	packYuv();
	int retval = send(sockets->mConn, yuvPacked.data(), (int) yuvPacked.size(), 0);
	if (retval < 0 || retval != yuvPacked.size()) {
		errorLogger.logError(std::format("error sending TCP data #{}", WSAGetLastError()));
	}
	mStatus.outputBytesWritten += retval;
}

TCPWriter::~TCPWriter() {
	closesocket(sockets->mConn);
	closesocket(sockets->mSock);
	WSACleanup();
}