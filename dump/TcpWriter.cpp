//-----------------------------------------------------------------------------------
class TCPWriter : public RawWriter {

protected:
	SOCKET mSock {};
	SOCKET mConn {};

public:
	TCPWriter(CpuData& data);

	virtual void write(FrameStatus& status) override;
	virtual void close(FrameStatus& status) override;
};


//-----------------------------------------------------------------------------------

TCPWriter::TCPWriter(CpuData& data) : RawWriter(data) {
	WSADATA wsaData {};
	int retval = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (retval < 0) throw AVException("cannot start TCP");

	SOCKET mSock = socket(AF_INET, SOCK_STREAM, 0);
	if (mSock < 0) throw AVException("cannot create TCP socket");

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

void TCPWriter::write(FrameStatus& status) {
	packYuv();
	int retval = send(mConn, yuvPacked.data(), (int) yuvPacked.size(), 0);
	if (retval < 0 || retval != yuvPacked.size()) {
		status.err << "error sending TCP data: " << WSAGetLastError();
	}
	status.outputBytesWritten += retval;
}

void TCPWriter::close(FrameStatus& status) {
	closesocket(mConn);
	closesocket(mSock);
	WSACleanup();
}