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
#include <chrono>

#include "DebugLogger.hpp"
#include "SystemStuff.hpp"
#include "AVException.hpp"

using namespace util;

std::mutex mutex;

void DebugLoggerConsole::log(const std::string& msg) {
	std::lock_guard<std::mutex> lock(mutex);
	std::cout << time() << msg << std::endl;
}

std::string DebugLoggerConsole::str() { return ""; }

DebugLoggerType DebugLoggerConsole::type() { return DebugLoggerType::CONSOLE; }

void DebugLoggerString::log(const std::string& msg) {
	std::lock_guard<std::mutex> lock(mutex);
	ss << time() << msg << std::endl;
}

std::string DebugLoggerString::str() { return ss.str(); }

DebugLoggerType DebugLoggerString::type() { return DebugLoggerType::STRING; }


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

DebugLoggerType DebugLoggerFile::type() { return DebugLoggerType::FILE; }

//---------------------------------------------------------------

std::shared_ptr<DebugLogger> DebugLogger::create(const std::string& logger) {
	std::regex patternTcp("^tcp://(.+):(\\d+)$");
	std::shared_ptr<DebugLogger> out;

	if (logger.starts_with("file://")) {
		out = std::make_shared<DebugLoggerFile>(logger.substr(7));

	} else if (std::smatch matcher; std::regex_match(logger, matcher, patternTcp)) {
		out = std::make_shared<DebugLoggerTcp>(matcher[1].str(), std::stoi(matcher[2]));

	} else if (logger == "console://") {
		out = std::make_shared<DebugLoggerConsole>();

	} else {
		out = std::make_shared<DebugLoggerNull>();
		throw AVException("invalid log parameter '" + logger + "'");
	}

	out->format(">>> starting log {} on thread {} <<<", std::chrono::current_zone()->to_local(std::chrono::system_clock::now()), threadId());
	return out;
}

void DebugLogger::open(const std::string& logger) {
	debugLoggerPtr = create(logger);
}

//---------------- system specific ---------------------------

#if defined(_WIN64)
#include <WinSock2.h> //include before windows.h
#include <Ws2tcpip.h>

SOCKET sock = INVALID_SOCKET;

//open tcp logger
DebugLoggerTcp::DebugLoggerTcp(const std::string& ip, int port) {
	std::lock_guard<std::mutex> lock(mutex);
	WSADATA wsaData = {};
	int retval = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (retval < 0) return;

	sock = socket(PF_INET, SOCK_STREAM, 0);
	sockaddr_in servAddr = {};
	servAddr.sin_family = PF_INET;
	inet_pton(AF_INET, ip.c_str(), &servAddr.sin_addr.s_addr);
	servAddr.sin_port = htons(port);
	mIsConnected = connect(sock, (SOCKADDR*) &servAddr, sizeof(servAddr));
}

//send debug messages over tcp on windows
void DebugLoggerTcp::log(const std::string& msg) {
	std::lock_guard<std::mutex> lock(mutex);
	if (mIsConnected == 0) {
		std::string str = time() + msg + "\n";
		int result = send(sock, str.c_str(), str.size(), 0);
	}
}

std::string DebugLoggerTcp::str() {
	return "tcp";
}

DebugLoggerType DebugLoggerTcp::type() { 
	return DebugLoggerType::TCP; 
}

DebugLoggerTcp::~DebugLoggerTcp() {
	std::lock_guard<std::mutex> lock(mutex);
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
	std::lock_guard<std::mutex> lock(mutex);
	sock = socket(AF_INET, SOCK_STREAM, 0);
	sockaddr_in servAddr = {};
	servAddr.sin_family = AF_INET;
	servAddr.sin_port = htons(port);
	inet_pton(AF_INET, ip.c_str(), &servAddr.sin_addr.s_addr);
	mIsConnected = connect(sock, (sockaddr*) &servAddr, sizeof(servAddr));
}

void DebugLoggerTcp::log(const std::string& msg) {
	std::lock_guard<std::mutex> lock(mutex);
	if (mIsConnected == 0) {
		std::string str = time() + msg + "\n";
		int result = send(sock, str.c_str(), str.size(), 0);
	}
}

std::string DebugLoggerTcp::str() {
	return "tcp";
}

DebugLoggerType DebugLoggerTcp::type() { 
	return DebugLoggerType::TCP; 
}

DebugLoggerTcp::~DebugLoggerTcp() {
	std::lock_guard<std::mutex> lock(mutex);
	shutdown(sock, SHUT_RDWR);
}

#else

DebugTcpWriter::DebugTcpWriter(int port) {}
DebugTcpWriter::~DebugTcpWriter() {}
void DebugTcpWriter::log(const std::string& msg) {}
std::string DebugLoggerTcp::str() { return "tcp"; }

#endif
