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
-o file         output file, will contain only video data, 
                other streams from input will be omitted
-o pipe:0       output to pipe in raw YUV444P format
                forward to other software, for example ffmpeg through
                -f rawvideo -framerate n -pix_fmt yuv444p -video_size w:h
                on windows only run in the command line, NOT in PowerShell
-o tcp://aa:p   output to tcp address in raw YUV444P format
                example: -o tcp://100.101.102.103:1234
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
-device -1      same as -device cpu
-encdev xxx     device to use for video encoding
                options: auto, gpu, cpu
                default: auto; which is selected based on computing device
-codec xxx      video codec to use for encoding
                options: auto, h264, h265
                default: auto; which is h265 on gpu / h264 on cpu

other output options:
-trf file       text file containing transformation data per frame
                use in conjunction with single pass mode, see -pass
-res file       write detailed results of the compute step to file
-resim fmt      write images containing lines to show calculations 
                images are grayscaled, lines are colored
                blue: calculated transform
                green: point is consens / red: point is not consens
                see -o options for filename pattern
-pass n         mode of operation
-pass 1         only reads video file and computes transformations
-pass 2         only applies transformation to generate output video
-pass 0         combination of both steps at once, requires larger buffer
-pass 12        consecutively run first and then second pass
                default: 0
-frames n       maximum number of frames to encode
                program will terminate when this number is reached
-blendsource x  blend input image over stabilized footage
                value must be between -1.0 and 1.0
                positive value puts input to left, negative to right side
                example 0.25: left 25% of video will show input
                default: 0

quality and performance settings:
-radius sec     the temporal radius where frames will be considered for 
                stabilization, given in seconds
                default: 0.5
-zoom value     zoom value to apply to the frames after stabilization
                default: 1.05
-bgmode mode    background mode, how to fill void when frame is out of center
                blend: use preceding frames to blend into current frame
                color: use defined color, see -bgcol
                default: blend
-bgcolor name   some predifined colors to use for background fill like
                red, green, blue, white, black, ...
                default: yellow
-bgcolor rgb    color triplet separated by colons in format R:G:B
                values must be 0..255
-cputhreads n   number of threads on the cpu to use for computations
                default: based on hardware
-crf n          constant rate factor used for encoding
                default: 22

misc options:
-y              overwrite output file
-h, -help       display this help text
-info           display information about software and hardware detected
-copyframes     just copy input to output, do not stabilize
                useful for testing decoding and encoding stuff
-progress n     show progress in different ways
-progress 0     quiet mode, no progress
-progress 1     frequently rewrite line on the output console
-progress 2     graph indicator
-progress 3     more detailed report
                default: 1

advanced computation parameters:
-levels         number of pyramid levels
                default: 4
-ir             integration radius, window is then 2*ir+1
                default: 3

keyboard input options at runtime:
key [e]         stop reading input, write pending output, then terminate
key [q]         stop reading and writing, flush output buffer
key [x]         stop reading and writing immediately
)";