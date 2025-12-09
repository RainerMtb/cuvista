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
-o file.nv12    output file, write raw video data in NV12 format
                half the size of YUV444P, still be warned

-o fmt.jpg      output a sequence of images in jpeg format
                format see below
-o fmt.bmp      output a sequence of images in 8bit color bmp format
                format see below
-o null         do not write any output file

fmt specifier   fmt must specify the output file and should contain a number
                formatting sequence, otherwise the same file will just
                be written repeatedly
                example: -o im%03d.bmp will produce 3 digits with leading 0
                im000.bmp, im001.bmp, im002.bmp, ...

following options may be provided after -o parameter
-enc xx:yy      specify output encoding options
                get a list of available video encoding options via -info
                default: auto:auto, based on -o parameter
-enc pipe:asf   write to pipe in asf container format
                video data arrives in YUV444P format, other streams are copied
                example: cuvista [...] -enc pipe:asf | ffmpeg -i pipe:0 [...]
                refer to pipe mechanism of your OS for further information
-enc pipe:raw   output raw YUV444P data to pipe
                use for example to connect to ffmpeg encoder
                for further processing, input must then be specified as
                -f rawvideo -video_size w:h -framerate n -pix_fmt yuv444p 
                refer to pipe mechanism of your OS for further information
                piping does not work in Windows PowerShell, use Command Line
-resvid         write video in nv12 format showing transform vectors
-resim          write bmp images showing transform vectors
                green: point is consens / red: point is not consens
-flow           produce video of calculated optical flow
-stack l:r      horizontally stack source and stabilized frames for comparison
                parameter l:r number of pixels to crop from left and right

computing options:
-device n       device to use for stabilization computations
                use -info to list available devices and run a small test
                default: max index value
-device cpu     same as -device 0

mode of operation:
-mode 0         read and write in one single pass, fastest, needs larger buffer
-mode 1         consecutively run read and write pass, needs smaller buffer
-mode n         multiple analysis passes, potentially improving stabilization
                default: 0, min: 0, max: 6

other output options:
-trf file       write file containing transformation data per frame
-res file       write detailed results of the compute step to text file
-frames n       maximum number of frames to encode, useful for a test sample
                default: all the frames there are

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
-bgcolor web    background color given as web color string in format "#RRGGBB"
-cputhreads n   number of threads on the cpu to use for various tasks
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
-nosummary      do not display summary satistics at end of program run
-showsummary    display summary statistics 
-quiet          same as '-progress 0 -noheader -nosummary'
                do not produce any output except error messages
-version        display version identifier

advanced computation parameters:
-levels         number of pyramid levels, between 1 and 6
                default: 3
-ir             integration radius, between 1 and 3
                default: 3
-roicrop h:v    specify region of interest for stabilization computations
                number of pixels to crop away horizontally and vertically
                may be used to ignore text along the edge of input
                default 0:0
                
keyboard input options at runtime:
key [e]         stop reading input, write pending output, then terminate
key [q]         stop reading and writing, gracefully terminate output
key [x]         stop reading and writing immediately
)";
