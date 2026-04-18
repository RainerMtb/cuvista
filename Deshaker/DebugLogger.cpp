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

#include <iostream>
#include <regex>
#include "SystemStuff.hpp"
#include "AVException.hpp"

using namespace util;

void DebugLoggerConsole::log(const std::string& msg) {
	std::lock_guard<std::mutex> lock(mutex);
	std::cout << time() << msg << std::endl;
}

std::string DebugLoggerConsole::str() { return ""; }

void DebugLoggerString::log(const std::string& msg) {
	std::lock_guard<std::mutex> lock(mutex);
	ss << time() << msg << std::endl;
}

std::string DebugLoggerString::str() { return ss.str(); }


DebugLoggerFile::DebugLoggerFile(const std::string& filename) :
	filename { filename },
	os { std::ofstream(filename) }
{}

void DebugLoggerFile::log(const std::string& msg) {
	os << msg << std::endl;
}

std::string DebugLoggerFile::str() {
	int siz = os.tellp();
	return std::format("file {} {} bytes", filename, siz); 
}

std::shared_ptr<DebugLogger> DebugLogger::create(const std::string& logger) {
	std::regex patternTcp("^tcp://(.+):(\\d+)$");

	if (logger.starts_with("file://")) {
		return std::make_shared<DebugLoggerFile>(logger.substr(7));

	} else if (std::smatch matcher; std::regex_match(logger, matcher, patternTcp)) {
		return std::make_shared<DebugLoggerTcp>(matcher[1].str(), std::stoi(matcher[2]));

	} else {
		throw AVException("invalid log parameter '" + logger + "'");
	}
}

//---------------- system specific ---------------------------

#if defined(_WIN64)
#include <WinSock2.h> //include before windows.h

SOCKET sock = INVALID_SOCKET;

//send debug messages over tcp on windows
DebugLoggerTcp::DebugLoggerTcp(const std::string& ip, int port) {
	WSADATA wsaData = {};
	int retval = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (retval < 0) return;

	sock = socket(PF_INET, SOCK_STREAM, 0);
	sockaddr_in servAddr = {};
	servAddr.sin_family = PF_INET;
	servAddr.sin_addr.s_addr = inet_addr(ip.c_str());
	servAddr.sin_port = htons(port);
	mIsConnected = connect(sock, (SOCKADDR*) &servAddr, sizeof(servAddr));
}

void DebugLoggerTcp::log(const std::string& msg) {
	std::lock_guard<std::mutex> lock(mutex);
	if (mIsConnected == 0) {
		std::string t = time();
		send(sock, t.c_str(), t.size(), 0);
		send(sock, msg.c_str(), msg.size(), 0);
		send(sock, "\n", 1, 0);
	}
}

std::string DebugLoggerTcp::str() {
	return "tcp";
}

DebugLoggerTcp::~DebugLoggerTcp() {
	log("shutdown");
	closesocket(sock);
	WSACleanup();
}

#elif defined(__linux__)

extern "C" {
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
}

int sock = -1;

DebugLoggerTcp::DebugLoggerTcp(const std::string& ip, int port) {
	sock = socket(AF_INET, SOCK_STREAM, 0);
	sockaddr_in servAddr = {};
	servAddr.sin_family = AF_INET;
	servAddr.sin_addr.s_addr = inet_addr(ip.c_str());
	servAddr.sin_port = htons(port);
	mIsConnected = connect(sock, (sockaddr*) &servAddr, sizeof(servAddr));
}

void DebugLoggerTcp::log(const std::string& msg) {
	std::lock_guard<std::mutex> lock(mutex);
	if (mIsConnected == 0) {
		std::string t = time();
		send(sock, t.c_str(), t.size(), 0);
		send(sock, msg.c_str(), msg.size(), 0);
		send(sock, "\n", 1, 0);
	}
}

std::string DebugLoggerTcp::str() {
	return "tcp";
}

DebugLoggerTcp::~DebugLoggerTcp() {
	log("shutdown");
	close(sock);
}

#else

DebugTcpWriter::DebugTcpWriter(int port) {}
DebugTcpWriter::~DebugTcpWriter() {}
void DebugTcpWriter::log(const std::string& msg) {}
std::string DebugLoggerTcp::str() { return "tcp"; }

#endif
