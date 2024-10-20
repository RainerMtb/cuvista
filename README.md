![GitHub Release](https://img.shields.io/github/v/release/RainerMtb/cuvista)
![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/RainerMtb/cuvista/total)
# CUVISTA - Gpu accelerated Video Stabilizer
Check the projects [GitHub Page](https://rainermtb.github.io/cuvista)

A simple, easy to use Application to stabilize shaky video footage preferably using GPU acceleration via Cuda or OpenCL platforms when available. The software will run on CPU power alone, which will be significantly slower but produces the exact same outcome.

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

Have a look at available options on the command line via ```cuvista -h``` or ```cuvista -help```, in the GUI version a subset of most important options is available.

In a fresh Windows environment the the Microsoft Visual C/C++ Runtime Libraries might be missing, so when prompted with a message like ```MSVCP140.dll not found```, download and install the latest redistributable https://aka.ms/vs/17/release/vc_redist.x64.exe

# Building CUVISTA
To start, get the repository and submodules to your computer, then proceed below according to your system


## Building on Windows
Tested on Windows 11

### Core Dependencies
Get Cuda https://developer.nvidia.com/cuda-downloads  
Get ffmeg with shared libraries https://www.ffmpeg.org/download.html#build-windows  

### Building the CLI executable
In a command line window, starting from the project main directory, where this readme and the top level CMakeLists.txt is located, exectute the following commands.  
It is recommended to build in a subdirectory:
```
mkdir build
cd build
```
prepare the locations of Cuda and FFMPEG and provide them to cmake. Better use forward slashes ```/``` to separate folders. Adapt locations to your system:
```
cmake .. -D CMAKE_PREFIX_PATH=C:/CUDA/version -D FFMPEG_PATH=C:/ffmpeg
cmake --build . --config Release
```
Upon successfull completion you will get the file
```
cuvistaCli/Release/cuvista.exe
```
Optionally continue with:
```
cmake --install . --config Release
```
This will put together the libraries for ffmpeg and the executable in the subfolder ```install```

### Including the GUI executable in the build
In addition to the Core Dependencies above get Qt https://www.qt.io/download-qt-installer, only the essential packages are required  
Provide the location of Qt to cmake via the ```CMAKE_PREFIX_PATH``` and proceed similar to the steps above. Adapt locations to your system:
```
cmake .. -D CMAKE_PREFIX_PATH=C:/Qt/6.8.0/msvc2022_64;C:/CUDA/version -D FFMPEG_PATH=C:/ffmpeg -D GUI=1
cmake --build . --config Release
```
Upon successfull completion you will get
```
cuvistaCli/Release/cuvista.exe
cuvistaGui/Release/cuvistaGui.exe
```
Optionally continue to pack everything together into the ```install``` subfolder with:
```
cmake --install . --config Release
```

## Building on Linux
Tested on Ubuntu 24.04

### Core Dependencies
Get Cuda https://developer.nvidia.com/cuda-downloads and follow instructions there  
Execute commands in a Linux Terminal to download and install additional components if not already available on your system

Cmake, the build tool:
```
sudo apt install -y cmake
```
FFmpeg shared libraries:
```
sudo apt install -y libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev libswresample-dev libswscale-dev
```
Nvidia Video encoder - I wonder why ***that library*** carries the name ffmpeg...
```
sudo apt install -y libffmpeg-nvenc-dev
```
Google Cpu Features:
```
sudo apt install -y libcpu-features-dev
```
In case PkgConfig is missing:
```
sudo apt install -y pkg-config
```

### Building the CLI executable
In a terminal session, starting from the project main directory, where this readme and the top level CMakeLists.txt is located

It is recommended to build in a subdirectory:
```
mkdir build
cd build
```
Adapt the location of cuda to your system and execute the build process
```
cmake .. -D CMAKE_PREFIX_PATH=/usr/local/cuda
cmake --build .
```
Upon successfull completion you will find the executable which should execute right away
```
cuvistaCli/cuvista
```

### Including the GUI executable in the build
In addition to the Core Libraries above get Qt https://www.qt.io/download-qt-installer, only the essential packages are required.  
You might want to ```wget``` the installer from https://download.qt.io/official_releases/online_installers/

You probably need
```
sudo apt install -y libxkbcommon-dev
```
Proceed similar to the steps above, also providing the location of Qt to cmake, adapt locations to your system
```
cmake .. -D CMAKE_PREFIX_PATH=~/Qt/6.8.0/gcc_64:/usr/local/cuda -D GUI=1
cmake --build .
```
Upon successfull completion you will find the executables which should execute right away
```
cuvistaCli/cuvista
cuvistaGui/cuvistaGui
```

## Future Ideas
- Improve performance and robustness of stabilization - very likely mutually exclusive goals
- Look into more advanced algorithms like 3D stabilization - but I currently lack information on the fundamental math of suchs approaches
- Use this codebase to remove duplicate frames from videos
- Possibly improve performace by using multiple GPUs - dont know if anyone would need that though
- Thinks in the Gui

## Built and Tested on following Tools and versions
- Windows 11 64bit
- Ubuntu 24.04
- Visual Studio 2022
- Nvidia Cuda 12.6
- Nvidia Video Codec SDK 12.2.72
- FFmpeg 7.1
- Qt 6.8.0
- Cmake 3.28.3
