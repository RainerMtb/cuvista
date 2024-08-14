![GitHub Release](https://img.shields.io/github/v/release/RainerMtb/cuvista)
![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/RainerMtb/cuvista/total)
# CUVISTA - Gpu accelerated Video Stabilizer
Check the projects [GitHub Page](https://rainermtb.github.io/cuvista)

A small and easy to use Application to stabilize shaky video footage using GPU acceleration via Cuda or OpenCL platforms when available. The software will run on CPU power alone, which will be significantly slower but produces the exact same outcome.

I put in great effort to ensure that on all platforms the stabilization results are absolutely equal. By that I mean floating point equal, to the bit. So far I tested on Windows 11 and Ubuntu 24.04 with MSVC and GCC compilers.

Have a look at a sample video comparing an original recording side by side to the stabilized version:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=kD84VqBurZc" target="_blank">
<img src="http://img.youtube.com/vi/kD84VqBurZc/mqdefault.jpg" alt="Cuvista Demo" width="320" height="160" border="10"/>
</a>

## GPU Support
For Cuda acceleration a device with Compute Version 5 or later is required. For OpenCL the device must support at least version 2.

## Typical Performance

On a RTX 3060 graphics card a typical video in FullHD resolution (1920 x 1080) should be processed at around 100 frames per second, including decoding, stabilizing and encoding.

# Using CUVISTA
For Windows you can get the latest version from the [Releases](https://github.com/RainerMtb/cuvista/releases) page. Just download, unzip and run either ```cuvista.exe``` on the command line or ```cuvistaGui.exe``` for a windowed user interface. 

Have a look at available options on the command line via ```cuvista -h``` or ```cuvista -help```, in the GUI version a subset of options is available

In a fresh Windows environment the the Microsoft Visual C/C++ Runtime Libraries might be missing, so when prompted with a message like ```MSVCP140.dll not found```, download and install the latest redistributable https://aka.ms/vs/17/release/vc_redist.x64.exe

# Building CUVISTA
## Building on Windows
### Main Dependencies

- Get Cuda https://developer.nvidia.com/cuda-downloads
- Get Qt https://www.qt.io/download-qt-installer, only the essential packages are required
- Get or build ffmeg with shared libraries https://www.ffmpeg.org/download.html#build-windows


### Building
in a command line window, starting from the project main directory, where this readme and the top level CMakeLists.txt is located, exectute the following commands

It is recommended to build in a subdirectory:
```
mkdir build
cd build
```
prepare the locations of Cuda, Qt, FFMPEG and provide them to cmake as outlined. Better use forward slashes ```/``` to separate folders. Adapt locations to your system:
```
cmake .. -DCMAKE_PREFIX_PATH=C:/Qt/6.7.2/msvc2019_64;C:/CUDA/v12.5 -DFFMPEG_PATH=C:/ffmpeg
cmake --build . --config Release
```
Upon successfull completion you will get the files
```
cuvistaCli/Release/cuvista.exe
cuvistaGui/Release/cuvistaGui.exe
```
Those programs will run right away if all necessary runtime dependencies are on the system path. Optionally continue with:
```
cmake --install .
```
This will put together the libraries for Qt and ffmpeg, you will then get everything packed together in the subfolder ```install```


## Building on Linux
Tested on Ubuntu 24.04
### Main Dependencies

- Get Cuda https://developer.nvidia.com/cuda-downloads 
- Get Qt https://www.qt.io/download-qt-installer 
- It is possible to install both packages purely from the command line, see their respective instructions

### More Libraries
Execute commands in a Linux Terminal to download and install additional components if not already available on your system

Cmake, the build tool:
```
sudo apt install -y cmake
```
FFmpeg shared libraries:
```
sudo apt install -y libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev libswresample-dev libswscale-dev
```
Nvidia Video encoder - I wonder why ***that*** library carries the name ffmpeg...
```
sudo apt install -y libffmpeg-nvenc-dev
```
Google Cpu Features:
```
sudo apt install -y libcpu-features-dev
```
Seems to be needed for the Qt stuff:
```
sudo apt install -y libxkbcommon-dev
```
### Building
Finally, in a terminal session, starting from the project main directory, where this readme and the top level CMakeLists.txt is located

It is recommended to build in a subdirectory:
```
mkdir build
cd build
```
tell cmake where to find Cuda and Qt, adapt the respective locations to your system - alternatively those paths can be provided directly to cmake via ```-D``` option:
```
export CMAKE_PREFIX_PATH=~/Qt/6.7.2/gcc_64:/usr/local/cuda
```
Execute the build process:
```
cmake ..
cmake --build .
```
Upon successfull completion you will find the executables
```
cuvistaCli/cuvista
cuvistaGui/cuvistaGui
```

## Future Ideas
- Improve performance on all devices
- Improve quality and robustness of stabilization - likely degrading performace
- Look into more advanced algorithms like 3D stabilization - but I currently lack information on the fundamental math of such an approach
- Use this codebase to remove duplicate frames from videos
- Maybe improve performace by using multiple GPUs - dont know if anyone would need that though

## Tested Versions
- Windows 11 64bit
- Ubuntu 24.04
- Visual Studio 2022
- Nvidia Cuda 12.5
- Nvidia Video Codec SDK 12.2.72
- FFmpeg 7.0
- Qt 6.7.2
- Cmake 3.28.3
