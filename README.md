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

## Limitations
Stabilization works best with *natural video footage* like a camera mounted on a vehicle. It does not really work with sudden scene changes like a movie cut. Sudden changes in brightness can also throw off computations. In such cases even more shaking can be introduced.

# Using CUVISTA
For Windows you can get the latest version from the [Releases](https://github.com/RainerMtb/cuvista/releases) page. Just download either the msi file to install, or the zip file to unzip and run either ```cuvista.exe``` on the command line or ```cuvistaGui.exe``` for a graphical user interface. 

Have a look at available options on the command line via ```cuvista -h``` or ```cuvista -help```, in the GUI version a subset of most important options is available.

In a fresh Windows environment the the Microsoft Visual C/C++ Runtime Libraries might be missing, so when prompted with a message like ```MSVCP140.dll not found```, download and install the latest redistributable https://aka.ms/vs/17/release/vc_redist.x64.exe



# Building CUVISTA

To start, get the repository and submodules to your computer
```
git clone --recursive https://github.com/RainerMtb/cuvista.git
```
This repo comes with a cmake script. As the most bare version only the command line executable is built without Cuda support.

The build script will look for Cuda and include Cuda when found. Tell cmake where to find Cuda by setting ```CMAKE_PREFIX_PATH=path/to/cuda``` or via the system path. Explicitly disable Cuda in the build via option ```-D BUILD_CUDA=0```. Explicitly enable Cuda via ```-D BUILD_CUDA=1```, this way skipping the search process

The build script will look for Qt6 and include the Gui executable when found. Tell cmake where to find Qt6 by setting ```CMAKE_PREFIX_PATH=path/to/qt``` or via the system path. Explicitly disable building the gui via option ```-D BUILD_GUI=0```



## Building on Windows
Tested on Windows 11

### Dependencies
Get ffmeg with shared libraries https://www.ffmpeg.org/download.html#build-windows  
Optionally get Cuda https://developer.nvidia.com/cuda-downloads  
Optionally get Qt6 https://www.qt.io/download-qt-installer, install at least the essentials and Qt Multimedia

### Building the full experience
In a command line window, starting from the project main directory, where this readme and the top level CMakeLists.txt is located, exectute the following commands.  
It is recommended to build in a subdirectory:
```
mkdir build
cd build
```
prepare the locations of Cuda, Qt and FFMPEG and provide them to cmake. Also see notes above. Better use forward slashes ```/``` to separate folders. Adapt locations to your system:
```
cmake .. -D CMAKE_PREFIX_PATH=C:/CUDA/version;C:/Qt/6.8.2/msvc2022_64 -D FFMPEG_PATH=C:/ffmpeg --fresh
cmake --build . --config Release
```
Upon successfull completion you will get the files
```
cuvistaCli/Release/cuvista.exe
cuvistaGui/Release/cuvistaGui.exe
```
Optionally continue with:
```
cmake --install . --config Release
```
This will put together the dependent libraries and the executables in the subfolder ```install```



## Building on Linux
Tested on Ubuntu 24.04

### Dependencies

#### Cuda
Optionally get Cuda https://developer.nvidia.com/cuda-downloads and follow instructions there  

With Cuda you will also need Nvidia Video encoder - I wonder why ***that library*** carries the name ffmpeg...
```
sudo apt install -y libffmpeg-nvenc-dev
```

#### Qt6
Optionally get Qt6, see https://www.qt.io/download-qt-installer. 
To get the necessary components on the command line
```
wget https://download.qt.io/official_releases/online_installers/qt-unified-linux-x64-online.run
chmod +x qt-unified-linux-x64-online.run
./qt-unified-linux-x64-online.run install qt.qt6.682.linux_gcc_64 qt.qt6.682.addons.qtmultimedia
```
You might also need
```
sudo apt install -y libxkbcommon-dev libglu1-mesa-dev
```

#### More Libraries
Get additional components if not already available on your system

Cmake, the build tool:
```
sudo apt install -y cmake
```
FFmpeg shared libraries:
```
sudo apt install -y libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev libswresample-dev libswscale-dev
```
In case PkgConfig is missing:
```
sudo apt install -y pkg-config
```

### Building the full experience
In a terminal session, starting from the project main directory, where this readme and the top level CMakeLists.txt is located

It is recommended to build in a subdirectory:
```
mkdir build
cd build
```
Adapt the location of Cuda and  Qt to your system and execute the build process. Also see notes above.
```
export CMAKE_PREFIX_PATH=~/Qt/6.8.2/gcc_64:/usr/local/cuda
cmake .. --fresh
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
- Maybe porting to Mac and Apple Silicon

 
## Built and Tested on following Tools and versions
- Windows 11 64bit
- Ubuntu 24.04
- Visual Studio 2022
- Nvidia Cuda 12.8
- Nvidia Video Codec SDK 12.2.72
- FFmpeg 7.1
- Qt 6.8.2
- Cmake 3.28.3
