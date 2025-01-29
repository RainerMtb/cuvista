# Theory of Operation

1. Create MainData structure
1. Optional: probe cuda devices
1. Optional: probe opencl devices
1. Collect device info
1. Create subclass of MovieReader and open input
1. Validate MainData variables with input
1. Optional: create one or more Auxiliary Writers - those are called per input frame
1. Create subclass of MovieWriter and open output - this is called per output frame
1. Create subclass of MovieFrame handler class - this selects the loop mechanism
1. Create subclass of FrameExecutor - this selects the device for computations
1. Create subclass of progress handler - this informs about progress during loop
1. Run the loop in the frame class
1. Close Writer
1. Close Reader
1. Close Executor
1. Close Frame

# Dependencies

## FFMPEG
always used to decode video
used to encode when not via nvenc
get shared build from https://www.gyan.dev/ffmpeg/builds/
or build with Media Auto Build https://github.com/m-ab-s/media-autobuild_suite

# Misc Notes
Cuda 11.8 is the highest version to support Compute 3.5
Video Codec 11.0.10 is the highest encoder version to go along with it

# Files To Deploy
in program folder:
```
   avcodec-61.dll
   avformat-61.dll
   avutil-59.dll
   D3Dcompiler_47.dll
   opengl32sw.dll
   Qt6Core.dll
   Qt6Gui.dll
   Qt6Multimedia.dll
   Qt6MultimediaWidgets.dll
   Qt6Network.dll
   Qt6Widgets.dll
   swresample-5.dll
   swscale-8.dll
   
---generic
       qtuiotouchplugin.dll
       
---multimedia
       ffmpegmediaplugin.dll
       windowsmediaplugin.dll
       
---networkinformation
       qnetworklistmanager.dll
       
---platforms
       qwindows.dll
       
---styles
       qmodernwindowsstyle.dll
       
---tls
        qcertonlybackend.dll
        qopensslbackend.dll
        qschannelbackend.dll
```