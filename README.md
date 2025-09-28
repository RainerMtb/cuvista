![GitHub Release](https://img.shields.io/github/v/release/RainerMtb/cuvista)
![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/RainerMtb/cuvista/total)
# CUVISTA - Gpu accelerated Video Stabilizer
Check the projects [GitHub Page](https://rainermtb.github.io/cuvista)

A simple, easy to use Application to stabilize shaky video footage preferably using GPU acceleration via Cuda or OpenCL. The software will run on CPU power alone, which will be significantly slower but produces the exact same outcome.

I put in great effort to ensure that on all platforms the stabilization results are absolutely equal. By that I mean floating point equal, to the bit.

There is no AI involved, just math, algorithms and brain power.

Have a look at a sample video comparing an original recording side by side to the stabilized version:


<a href="http://www.youtube.com/watch?feature=player_embedded&v=kD84VqBurZc" target="_blank">
<img src="http://img.youtube.com/vi/kD84VqBurZc/mqdefault.jpg" alt="Cuvista Demo" width="340" height="170" border="10"/>
</a>

<a href="http://www.youtube.com/watch?feature=player_embedded&v=kBkYwDKidPA" target="_blank">
<img src="http://img.youtube.com/vi/kBkYwDKidPA/mqdefault.jpg" alt="Cuvista Demo" width="340" height="170" border="10"/>
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

In a fresh Windows environment the the Microsoft Visual C/C++ Runtime Libraries might be missing, so when using the zip archive you might be prompted with a message like ```MSVCP140.dll not found```, then download and install the latest redistributable https://aka.ms/vs/17/release/vc_redist.x64.exe. The MSI installer already comes with the necessary system files.



# Building CUVISTA

To start, get the repository and submodules to your computer
```
git clone --recursive https://github.com/RainerMtb/cuvista.git
```
This repo comes with a cmake script. As the most bare version only the command line executable is built without Cuda support.

The build script will look for Cuda and include Cuda when found. Tell cmake where to find Cuda by setting ```CMAKE_PREFIX_PATH=path/to/cuda``` or via the system path. Explicitly disable Cuda in the build via option ```-D BUILD_CUDA=0```. Explicitly enable Cuda via ```-D BUILD_CUDA=1```, this way skipping the search process

The build script will look for Qt6 and include the Gui executable when found. Tell cmake where to find Qt6 by setting ```CMAKE_PREFIX_PATH=path/to/qt``` or via the system path. Explicitly disable building the gui via option ```-D BUILD_GUI=0```



## Building on Windows
Tested on Windows 11 and MSVC

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
cmake .. -D CMAKE_PREFIX_PATH=C:/CUDA/version;C:/Qt/6.9.2/msvc2022_64 -D FFMPEG_PATH=C:/ffmpeg --fresh
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

When done, run ```cuvista -info``` to see available devices and run a test



## Building on Linux
Tested on Ubuntu 25.04

### Dependencies

#### Cuda
Optionally get Cuda https://developer.nvidia.com/cuda-downloads and follow instructions there  

When using Cuda you will also need Nvidia Video encoder - I wonder why ***that library*** carries the name ffmpeg...
```
sudo apt install -y libffmpeg-nvenc-dev
```

#### Qt6
Optionally get Qt6 to build a Gui, see https://www.qt.io/download-qt-installer. 
To get the necessary components on the command line
```
wget https://download.qt.io/official_releases/online_installers/qt-online-installer-linux-x64-online.run
chmod +x qt-online-installer-linux-x64-online.run
./qt-online-installer-linux-x64-online.run install qt.qt6.692.linux_gcc_64 qt.qt6.692.addons.qtmultimedia
```
This will then require
```
sudo apt install -y libxkbcommon-dev libglu1-mesa-dev
```

#### More Libraries
Cmake, the build tool:
```
sudo apt install -y cmake
```
FFmpeg shared libraries:
```
sudo apt install -y libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavutil-dev libswresample-dev libswscale-dev
```
PkgConfig is used to check ffmpeg components:
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
export CMAKE_PREFIX_PATH=~/Qt/6.9.2/gcc_64:/usr/local/cuda
cmake .. --fresh
cmake --build . --parallel
```
Upon successfull completion you will find the executables which should execute right away
```
cuvistaCli/cuvista
cuvistaGui/cuvistaGui
```

When done, run ```cuvista -info``` to see available devices and run a test


## Future Ideas
- Improve performance and robustness of stabilization - very likely mutually exclusive goals
- Maybe supporting multiple GPUs for shared execution - dont know if anyone would need that though
- Maybe look into more advanced algorithms for stabilization
- Maybe provide some API to the stabilization
- Maybe porting to Mac and Apple Silicon

 
## Built and Tested on following Tools and versions
- Windows 11
- Ubuntu 25.04
- Visual Studio 2022
- Nvidia Cuda 13.0.1
- Nvidia Video Codec SDK 13.0.19
- FFmpeg 8.0
- Qt 6.9.2
- Cmake later than 3.28
