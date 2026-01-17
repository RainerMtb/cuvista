# Theory of Operation

1. Create MainData structure
1. Optional: probe cuda devices
1. Optional: probe opencl devices
1. Collect device info
1. Create subclass of MovieReader and open input
1. Validate MainData variables with input
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
in program folder for qt gui:
```
   avcodec.dll
   avformat.dll
   avutil.dll
   swresample.dll
   swscale.dll

   D3Dcompiler_47.dll
   opengl32sw.dll
   Qt6Core.dll
   Qt6Gui.dll
   Qt6Multimedia.dll
   Qt6MultimediaWidgets.dll
   Qt6Network.dll
   Qt6Widgets.dll

   
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

# update git submodules to latest commit
on the main project folder
```
git submodule update --recursive --remote --rebase
```