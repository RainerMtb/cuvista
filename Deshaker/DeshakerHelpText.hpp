/*
 * This file is part of CUVISTA - Cuda Video Stabilizer
 * Copyright (c) 2023 Rainer Bitschi cuvista@a1.net
 *
 * This program is free software : you can redistribute it and /or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.If not, see < http://www.gnu.org/licenses/>.
 */

const char* helpString = R"(
usage: cuvista [-i inputfile -o outputfile] [options...]

input/output options:
-i file         input file, anything that ffmpeg can handle as input
-o file.mp4     output file in .MP4 format
-o file.mkv     output file in .MKV format
-o file.yuv     output file, write uncompressed YUV444P video data
                this will produce a huge file, you have been warned

-o pipe:        output raw YUV444P data to pipe
                use for example to connect to ffmpeg encoder
                for further processing, input must then be specified as
                -f rawvideo -video_size w:h -framerate n -pix_fmt yuv444p 
                refer to pipe mechanism of your OS for further information
-o fmt.jpg      output a sequence of images in jpeg format
                format see below for bmp images
-o fmt.bmp      output a sequence of images in 8bit color bmp format

fmt specifier   fmt must specify the output file and should contain a number
                formatting sequence, otherwise the same file will just
                be written repeatedly
                example: -o im%03d.bmp will produce 3 digits with leading 0
                im000.bmp, im001.bmp, im002.bmp, ...

-o null         do not write any output, for whatever reason this may be

computing options:
-device n       device to use for stabilization computations
                use -info to list available devices and run a small test
                default: highest index available
-device cpu     same as -device 0
-encoder xxx    device to use for video encoding
                options: auto, nvenc, ffmpeg
                default: auto; prefer nvenc when available
-codec xxx      video codec to use for encoding
                options: auto, h264, h265
                default: auto; which is h265 on gpu / h264 on cpu

mode of operation:
-mode 0         read and write in one single pass, fast, requires large buffer
-mode 1         consecutively run read and write pass, requires less memory
-mode n         multiple analysis passes, potentially improving stabilization
                default: 0
                maximum: 6

other output options:
-trf file       text file containing transformation data per frame
-res file       write detailed results of the compute step to text file
-resim fmt.bmp  write images containing transform vectors in bmp file format 
                images are grayscaled, lines are colored
                green: point is consens / red: point is not consens
                blue: calculated transform
                see above for filename pattern
-resim folder   write images to specified folder using default filename pattern
-flow file      produce video of calculated optical flow

-frames n       maximum number of frames to encode
                default: all the frames there are
-stack l:r      horizontally stack source and stabilized frames for comparison
                parameter l:r defines pixels to crop from left and right
                provide this option after setting the output file via -o xxx

quality and performance settings:
-radius sec     the temporal radius within which frames will be considered for 
                stabilization, given in seconds
                default: 0.5
-zoom n         fixed zoom value to apply to the frames after stabilization
                value n is given in percent
-zoom n:m       dynamic zoom between values n and m, minimizing to fill frame
                default zoom setting is dynamic between 5% and 15%
-bgmode mode    background mode, how to fill void when frame is out of center
                options: blend, color
                blend: use preceding frames to blend into current frame
                color: use defined color, see -bgcolor
                default: blend
-bgcolor name   some predifined colors to use for background fill like
                red, green, blue, white, black, ...
                default: green
-bgcolor rgb    background colors separated by colons in format RRR:GGG:BBB
                color values must be between 0 and 255
-bgcolor web    background color given as a web color string in format #RRGGBB
-cputhreads n   number of threads on the cpu to use for computations
                default: 3/4 of hardware threads
-crf n          constant rate factor used for encoding
                lower value represents higher quality and bigger file size
                valid range and effect depends on codec in use
                default: encoder specific default value

misc options:
-y              overwrite output file without asking
-n              never overwrite output, quit application instead
-h, -help, -?   display help text
-info           display information about software and hardware detected
                run a test on available devices
-copyframes     just copy input to output, do not stabilize
                useful for testing decoding and encoding stuff
-progress 0     quiet mode, do not output progress information
-progress 1     default mode shows progress for input, output, encoding
-progress 2     frequently rewrite single line on the output console
-progress 3     print new line for every frame
-progress 4     simple graph indicator
-noheader       do not display program info at start
-showheader     display program info at start
-nosummary      do not display summary satistics at end of program
-showsummary    display summary statistics 
-quiet          same as '-progress 0 -noheader -nosummary'
                do not produce any output except error messages
-version        display version identifier

advanced computation parameters:
-levels         number of pyramid levels, between 1 and 6
                default: 3
-ir             integration radius, between 1 and 3
                default: 3
-roicrop h:v    specify region of interest for analyzation
                number of pixels to crop separated by colon
                representing cropHorizontal:cropVertical
                default 0:0
                
keyboard input options at runtime:
key [e]         stop reading input, write pending output, then terminate
key [q]         stop reading and writing, gracefully terminate output
key [x]         stop reading and writing immediately
)";
