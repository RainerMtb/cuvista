#include "CudaInfo.hpp"

std::string CudaInfo::nvidiaDriver() const {
	return std::to_string(nvidiaDriverVersion / 100) + "." + std::to_string(nvidiaDriverVersion % 100);
}

std::string CudaInfo::cudaRuntime() const {
	return std::to_string(cudaRuntimeVersion / 1000) + "." + std::to_string(cudaRuntimeVersion % 1000 / 10);
}

std::string CudaInfo::cudaDriver() const {
	return std::to_string(cudaDriverVersion / 1000) + "." + std::to_string(cudaDriverVersion % 1000 / 10);
}

bool CudaInfo::isSupported(int device, OutputCodec codec) const {
	const std::vector<OutputCodec>& cudaCodecs = supportedCodecs[device];
	return std::find(cudaCodecs.cbegin(), cudaCodecs.cend(), codec) != cudaCodecs.cend();
}