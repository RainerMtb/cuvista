A small and simple Application to stabilize shaky video footage using the Cuda platform when available.

There also is the option to use CPU power only in order to do the all the stabilizing computations, which will be significantly slower but produces the exact same outcome.

Have a look at some [sample video footage](https://youtu.be/kD84VqBurZc) that shows original and stabilized video side by side for comparison

## Just using it
Get the latest version from the [Releases](https://github.com/RainerMtb/cuvista/releases) page, download, unzip and run either ```cuvista.exe``` on the command line or ```cuvistaGui.exe``` for a windowed user interface
## Building the application yourself
To work on the code yourself and build the application, get the source code and start by opening ```Deshaker.sln```. Use ```Build Solution``` to get everything compiled and built

The following resources are used
- Windows 11 64bit
- Visual Studio 2022
- Nvidia Cuda 11.8
- Nvidia Video Codec SDK 11.0.10
- Nvidia NvApi R530
- FFmpeg 6.0

For the GUI
- Qt 6.5.0
