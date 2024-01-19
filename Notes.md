# General program flow

1. Create MainData structure
1. Optional: probe cuda and/or opencl devices
1. Collect device info
1. Create subclass of MovieReader and open input
1. Validate MainData variables
1. Create subclass of MovieWriter and open output
1. Create subclass of MovieFrame handler class
1. Set up progress information handler class
1. Decide on the type of loop to run
1. Run the loop in the frame class
1. Close Writer
1. Close Reader
1. Close Frame

# Dependencies

## Cpu Features
https://github.com/google/cpu_features.git
used to get name and vector features of Cpu
build commands in repo readme do not work

test executable
> cmake -S. -Bbuild -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
> cmake --build build --config Release -j --target ALL_BUILD
> .\build\Release\list_cpu_features

This will show CPU features of current device

To build library
> cmake -B build -D BUILD_TESTING=OFF --install-prefix=f:\repos\cpu_features\install
> cmake --build build --config Release
> cmake --install build

## Nvidia NvApi
https://github.com/NVIDIA/nvapi.git
used only to get and show Nvidia Driver Version
just clone, repo already comes with headers and .lib

## FFMPEG
always used to decode video
used to encode when not via nvenc
get shared build from https://www.gyan.dev/ffmpeg/builds/
or build with Media Auto Build https://github.com/m-ab-s/media-autobuild_suite
but M-AB-S rarely ever succeeds, always some problem with some package

# Misc Notes
Cuda 11.8 is the highest version to support Compute 3.5
Video Codec 11.0.10 is the highest encoder version to go along with it