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
-i file         input file, anything that ffmpeg can read will do
-o file         output file, recommended formats are .MP4 and .MKV
-o pipe:0       output to pipe in raw YUV444P format
                forward to other software, for example ffmpeg through
                -f rawvideo -framerate n -pix_fmt yuv444p -video_size w:h
                note: does not work on windows PowerShell, use cmd shell
-o null         do not write any output
-o fmt.jpg      output a sequence of images in jpeg format
                format see below for bmp images
-o fmt.bmp      output a sequence of images in 8bit bmp format
                fmt must contain a number formatting sequence
                example: -o im%03d.bmp will produce 3 digits with leading 0

computing options:
-device n       device to use for stabilization computations
                use -info to list installed cuda devices
                default: highest index available
-device cpu     compute everything on the cpu, do not use cuda devices
                this will be comparatively very slow
-encoder xxx    device to use for video encoding
                options: auto, nvenc, cpu
                default: auto; which is selected based on computing device
-codec xxx      video codec to use for encoding
                options: auto, h264, h265
                default: auto; which is h265 on gpu / h264 on cpu

other output options:
-trf file       text file containing transformation data per frame
                use in conjunction with single pass mode, see -pass
-res file       write detailed results of the compute step to file
-resim files    write images containing transform vectors in bmp file format 
                images are grayscaled, lines are colored
                green: point is consens / red: point is not consens
                blue: calculated transform
                see -o options for filename pattern
-resim folder   write images to specified folder using default filename pattern
-pass n         mode of operation
-pass 0         read and write in one single pass, requires large buffer
-pass 12        consecutively run first and then second pass
                reads input twice, requires less buffer memory
                default: 0
-frames n       maximum number of frames to encode
                program will terminate when this number is reached
                default: max int value 
-stack x        stack part of source and stabilized frames side by side
                parameter value must be between -1.0 and 1.0 and
                specifies which part of images is shown
                final output video is 3/2 the input width
                3/4 of input and 3/4 of output are displayed
                -1 takes leftmost part, 0 shows the middle, +1 takes the right
                default: 0
-flow file      produce video of calculated optical flow

quality and performance settings:
-radius sec     the temporal radius where frames will be considered for 
                stabilization, given in seconds
                default: 0.5
-zoom n         fixed zoom value to apply to the frames after stabilization
                value n is given in percent
-zoom n:m       dynamic zoom between values n and m, minimizing to fill frame
                default zoom setting is dynamic between 5 % and 20 %
-bgmode mode    background mode, how to fill void when frame is out of center
                blend: use preceding frames to blend into current frame
                color: use defined color, see -bgcol
                default: blend
-bgcolor name   some predifined colors to use for background fill like
                red, green, blue, white, black, ...
                default: green
-bgcolor rgb    background colors separated by colons in format R:G:B
                values must be 0..255
-bgcolor web    background color given as a web color string in format #RRGGBB
-cputhreads n   number of threads on the cpu to use for computations
                default: based on hardware
-crf n          constant rate factor used for encoding
                lower value represents higher quality and bigger file size
                default: encoder specific default value

misc options:
-y              overwrite output file without asking
-n              never overwrite output, quit application instead
-h, -help, -?   display help text
-info           display information about software and hardware detected
                run a test on available devices
-copyframes     just copy input to output, do not stabilize
                useful for testing decoding and encoding stuff
-progress n     show progress in different ways
-progress 0     quiet mode, do not output progress information
-progress 1     default mode
-progress 2     frequently rewrite line on the output console
-progress 3     print new line for every frame
-progress 4     graph indicator
-noheader       do not display program info at start
-showheader     display program info at start
-nosummary      do not display summary satistics at end of program
-showsummary    display summary statistics 
-quiet          same as '-progress 0 -noheader -nosummary'
                do not produce any output except error messages

advanced computation parameters:
-levels         number of pyramid levels
                default: 3
-ir             integration radius
                default: 3, maximum: 3
-version        display version identifier

keyboard input options at runtime:
key [e]         stop reading input, write pending output, then terminate
key [q]         stop reading and writing, gracefully terminate output
key [x]         stop reading and writing immediately
)";