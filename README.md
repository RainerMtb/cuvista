# CUVISTA - Gpu accelerated Video Stabilizer
Check the projects [GitHub Page](https://rainermtb.github.io/cuvista)

A small and easy to use Application to stabilize shaky video footage using GPU acceleration via the Cuda platform or OpenCL when available.

There is the option to only use CPU power to do the all the stabilizing computations, which will be significantly slower but produces the exact same outcome

Have a look at a sample video:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=kD84VqBurZc" target="_blank">
<img src="http://img.youtube.com/vi/kD84VqBurZc/mqdefault.jpg" alt="Cuvista Demo" width="320" height="160" border="10"/>
</a>

## GPU Support
For Cuda acceleration a device with Compute Version 5 or later is required

For OpenCL the device must support at least version 2

## Getting started
Get the latest version from the [Releases](https://github.com/RainerMtb/cuvista/releases) page, download, unzip and run either ```cuvista.exe``` on the command line or ```cuvistaGui.exe``` for a windowed user interface. 

In a fresh Windows environment the the Microsoft Visual C/C++ Runtime Libraries might be missing, so when prompted with a message like ```MSVCP140.dll not found```, download and install the latest redistributable https://aka.ms/vs/17/release/vc_redist.x64.exe
## Building
To work on the code yourself and build the application, get the source code and start by opening ```Deshaker.sln``` in Visual Studio

## Resources and Dependencies
- Windows 11 64bit
- Visual Studio 2022
- Nvidia Cuda 12.3
- Nvidia Video Codec SDK 12.1.14
- Nvidia NvApi https://github.com/NVIDIA/nvapi.git
- Google Cpu Features https://github.com/google/cpu_features.git
- FFmpeg 6.0

For the GUI
- Qt 6.6.1
