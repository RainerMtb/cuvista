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

#include <pch.h>
#include <ShObjIdl.h>
#include "MainWindow.xaml.h"
#undef min
#undef max

#include "AppUtil.hpp"

#include <winrt/Windows.Graphics.Imaging.h>
#include <winrt/Windows.Storage.Streams.h>
#include <winrt/Microsoft.UI.Xaml.Media.Imaging.h>


void debugPrint(winrt::hstring str) {
    OutputDebugString(L"-------- ");
    OutputDebugString(str.c_str());
    OutputDebugString(L"\n");
}

void debugPrint(const std::string& str) {
    debugPrint(winrt::to_hstring(str));
}

winrt::Windows::Foundation::IInspectable box_string(const std::string& str) {
    return winrt::box_value(winrt::to_hstring(str));
}


//input file
winrt::hstring selectFileOpen(HWND hwnd) {
    winrt::hstring out = L"";
    IFileDialog* ifd = nullptr;

    HRESULT result = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_INPROC_SERVER, IID_IFileOpenDialog, (void**) (&ifd));
    if (result == S_OK) {
        ifd->SetOptions(FOS_PATHMUSTEXIST | FOS_FILEMUSTEXIST);
        ifd->SetTitle(L"Select Video file to open");
        COMDLG_FILTERSPEC filterSpec[] = { { L"All Files", L"*.*"} };
        ifd->SetFileTypes(1, filterSpec);
        if (ifd->Show(hwnd) == S_OK) {
            IShellItem* item = nullptr;
            if (ifd->GetResult(&item) == S_OK) {
                PWSTR file = nullptr;
                if (item->GetDisplayName(SIGDN_FILESYSPATH, &file) == S_OK) {
                    out = file;
                }
            }
        }
    }

    ifd->Release();
    return out;
}


//output file
winrt::hstring selectFileSave(HWND hwnd, bool alwaysOverwrite) {
    winrt::hstring out = L"";
    IFileDialog* ifd = nullptr;

    HRESULT result = CoCreateInstance(CLSID_FileSaveDialog, NULL, CLSCTX_INPROC_SERVER, IID_IFileSaveDialog, (void**) (&ifd));
    if (result == S_OK) {
        FILEOPENDIALOGOPTIONS op = alwaysOverwrite ? 0 : FOS_OVERWRITEPROMPT;
        ifd->SetOptions(op);
        ifd->SetDefaultExtension(L"*.mp4");
        ifd->SetTitle(L"Select Video file to save");
        COMDLG_FILTERSPEC filterSpec[] = { { L"Video Files", L"*.mp4;*.mkv" }, { L"All Files", L"*.*" } };
        ifd->SetFileTypes(2, filterSpec);
        if (ifd->Show(hwnd) == S_OK) {
            IShellItem* item = nullptr;
            if (ifd->GetResult(&item) == S_OK) {
                PWSTR file = nullptr;
                if (item->GetDisplayName(SIGDN_FILESYSPATH, &file) == S_OK) {
                    out = file;
                }
            }
        }
    }

    ifd->Release();
    return out;
}


//output folder to store image sequence
winrt::hstring selectFolder(HWND hwnd) {
    winrt::hstring out = L"";
    IFileDialog* ifd = nullptr;

    HRESULT result = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_INPROC_SERVER, IID_IFileOpenDialog, (void**) (&ifd));
    if (result == S_OK) {
        ifd->SetOptions(FOS_PICKFOLDERS);
        if (ifd->Show(hwnd) == S_OK) {
            IShellItem* item = nullptr;
            if (ifd->GetResult(&item) == S_OK) {
                PWSTR folder = nullptr;
                if (item->GetDisplayName(SIGDN_FILESYSPATH, &folder) == S_OK) {
                    out = folder;
                }
            }
        }
    }

    ifd->Release();
    return out;
}


//-------------------------------------------------------------------------
//-------------------- Progress Implementations ---------------------------
//-------------------------------------------------------------------------

using namespace winrt::cuvistaWinui::implementation;

//on background thread
void ProgressGui::init() {
    HRESULT hr = CoCreateInstance(CLSID_TaskbarList, NULL, CLSCTX_INPROC_SERVER, IID_ITaskbarList3, (void**) (&mTaskbarPtr));
    if (hr != S_OK) mTaskbarPtr = nullptr;
}


//on background thread
void ProgressGui::update(const ProgressInfo& progress, bool force) {
    //progress value
    double progressValue = progress.totalProgress * 10.0;
    mainWindow.DispatcherQueue().TryEnqueue([&, progressValue] { mainWindow.progressBar().Value(progressValue); });

    if (mTaskbarPtr && isDue(force)) {
        ULONGLONG total = (ULONGLONG) progressValue;
        mTaskbarPtr->SetProgressValue(hwnd, total, 1000);
    }

    //progress images and timestamps
    auto timePointNow = std::chrono::steady_clock::now();
    std::chrono::nanoseconds delta = timePointNow - mTimePoint;
    bool imageDue = delta.count() / 1'000'000 > 250;

    if (imageDue && progress.readIndex > 0) {
        mTimePoint = timePointNow;
        int64_t idx = progress.readIndex - 1;
        mExecutor.getInput(idx, mainWindow.mProgressInput);
        winrt::hstring hstr = winrt::to_hstring(mExecutor.mFrame.ptsForFrameAsString(idx));
        mainWindow.DispatcherQueue().TryEnqueue([&, hstr] { 
            mainWindow.textTimeInput().Text(hstr); 
            mainWindow.mProgressInput.invalidate();
        });
    }

    if (imageDue && progress.writeIndex > 0) {
        mTimePoint = timePointNow;
        int64_t idx = progress.writeIndex - 1;
        mExecutor.getWarped(idx, mainWindow.mProgressOutput);
        winrt::hstring hstr = winrt::to_hstring(mExecutor.mFrame.ptsForFrameAsString(idx));
        mainWindow.DispatcherQueue().TryEnqueue([&, hstr] { 
            mainWindow.textTimeOutput().Text(hstr); 
            mainWindow.mProgressOutput.invalidate();
        });
    }
}


//on background thread
void ProgressGui::updateStatus(const std::string& msg) {
    winrt::hstring hstr = winrt::to_hstring(msg);
    mainWindow.DispatcherQueue().TryEnqueue([&, hstr] { mainWindow.textProgressState().Text(hstr); });
}


//on background thread
void ProgressGui::terminate() {
    if (mTaskbarPtr) {
        mTaskbarPtr->SetProgressState(hwnd, TBPF_NOPROGRESS);
        mTaskbarPtr->Release();
    }
}


Controls::IContentDialog ProgressGui::dialog() {
    return mainWindow.progressDialog();
}


//-------------------------------------------------------------------------
//-------------------- Player Writer --------------------------------------
//-------------------------------------------------------------------------

using namespace winrt;
using namespace winrt::Windows::Media::Audio;


//on ui thread
void PlayerWriter::open(OutputOption outputOption) {
    //handling input streams
    for (StreamContext& sc : mReader.mInputStreams) {
        auto posc = std::make_shared<OutputStreamContext>();
        posc->inputStream = sc.inputStream;

        if (sc.inputStream->index == mReader.videoStream->index) {
            posc->handling = StreamHandling::STREAM_STABILIZE;

        } else if (sc.inputStream->index == mainWindow.mAudioStreamIndex) {
            posc->handling = StreamHandling::STREAM_DECODE;
            mAudioContext = posc;
            mPlayAudio = true;

        } else {
            posc->handling = StreamHandling::STREAM_IGNORE;
        }
        sc.outputStreams.push_back(posc);
    }

    loadImageScaled(mainWindow.imageRealtime(), L"ms-appx:///Assets/signs-01.png");
    mainWindow.imageRealtime().Visibility(winrt::Microsoft::UI::Xaml::Visibility::Collapsed);
    mainWindow.lblSpeaker().Symbol(mPlayAudio ? Controls::Symbol::Volume : Controls::Symbol::Mute);
    mainWindow.sliderVolume().IsEnabled(mPlayAudio);
    mainWindow.lblPlayerStatus().Text(L"Buffering...");
    mainWindow.mPlayerPaused = false;
    mainWindow.btnPlayerPause().IsChecked(false);
}


//on background thread
void PlayerWriter::start() {
    if (mAudioContext) {
        int sampleRate = mReader.openAudioDecoder(*mAudioContext);
        AudioGraphSettings setting = AudioGraphSettings(winrt::Windows::Media::Render::AudioRenderCategory::Media);
        CreateAudioGraphResult result = AudioGraph::CreateAsync(setting).get();
        if (result.Status() == AudioGraphCreationStatus::Success) {
            mAudioGraph = result.Graph();

            //open output node
            CreateAudioDeviceOutputNodeResult result = mAudioGraph.CreateDeviceOutputNodeAsync().get();
            if (result.Status() == AudioDeviceNodeCreationStatus::Success) {
                AudioDeviceOutputNode deviceOutputNode = result.DeviceOutputNode();

                //open input node
                Windows::Media::MediaProperties::AudioEncodingProperties props = mAudioGraph.EncodingProperties();
                props.ChannelCount(2);
                props.SampleRate(sampleRate);
                mAudioInputNode = mAudioGraph.CreateFrameInputNode(props);

                //connect
                mAudioInputNode.AddOutgoingConnection(deviceOutputNode);

                //start audio graph
                mAudioGraph.Start();
            }
        }
    }
}


//on background thread
void PlayerWriter::writeOutput(const FrameExecutor& executor) {
    executor.getOutputImage(frameIndex, mainWindow.mProgressOutput);

    //presentation time for next frame
    auto t1 = mReader.ptsForFrameAsMillis(frameIndex);
    auto t2 = mReader.ptsForFrameAsMillis(frameIndex + 1);
    int64_t delta = t1.has_value() && t2.has_value() ? (*t2 - *t1) : 0;

    using namespace winrt::Microsoft::UI::Xaml;

    //check time to play video frame
    auto tnow = std::chrono::steady_clock::now();
    Visibility vis = (tnow > mNextPts) ? Visibility::Visible : Visibility::Collapsed;
    mainWindow.DispatcherQueue().TryEnqueue([&, vis] { mainWindow.imageRealtime().Visibility(vis); });
    while (tnow < mNextPts || mainWindow.mPlayerPaused) {
        tnow = std::chrono::steady_clock::now();
    }
    mainWindow.DispatcherQueue().TryEnqueue([&] { mainWindow.mProgressOutput.invalidate(); });

    //play audio
    if (mAudioContext && mAudioGraph && mAudioInputNode) {
        std::unique_lock<std::mutex> lock(mAudioContext->mMutexSidePackets);
        double videoPts = t1.value_or(0.0) / 1000.0;
        for (auto it = mAudioContext->sidePackets.begin(); it != mAudioContext->sidePackets.end() && it->pts < videoPts + 0.25; ) {
            uint32_t siz = (uint32_t) it->audioData.size();
            Windows::Media::AudioFrame frame(siz);
            Windows::Media::AudioBuffer buffer = frame.LockBuffer(Windows::Media::AudioBufferAccessMode::Write);
            winrt::Windows::Foundation::IMemoryBufferReference ref = buffer.CreateReference();
            std::copy_n(it->audioData.data(), siz, ref.data());
            ref.Close();
            buffer.Close();
            it = mAudioContext->sidePackets.erase(it);

            mAudioInputNode.AddFrame(frame);
        }

        mAudioInputNode.OutgoingGain(mainWindow.mAudioGain);
    }

    //set next presentation time
    mNextPts = tnow + std::chrono::milliseconds(delta);
    frameIndex++;
}


//on background thread
bool PlayerWriter::flush() {
    //wait some time after the last frame is displayed before closing the player
    std::this_thread::sleep_for(std::chrono::milliseconds(750));
    return false;
}


//on background thread
void PlayerWriter::close() {
    if (mAudioGraph) {
        mAudioGraph.Stop();
        mAudioGraph.Close();
    }
}


//on background thread
void PlayerProgress::update(const ProgressInfo& progress, bool force) {
    int64_t idx = progress.writeIndex - 1;
    auto opstr = mExecutor.mFrame.mReader.ptsForFrameAsString(idx);

    //frame stats
    winrt::hstring frameString = L"";
    if (opstr.has_value()) {
        frameString = hformat("Frame: {} ({})", idx, opstr.value());
    }

    //player state
    winrt::hstring status;
    if (mainWindow.mPlayerPaused) status = L"Pausing...";
    else if (idx < 0) status = L"Buffering...";
    else if (progress.writeIndex == progress.readIndex) status = L"Ending...";
    else status = L"Playing...";

    mainWindow.DispatcherQueue().TryEnqueue([&, status, frameString] {
        mainWindow.lblPlayerStatus().Text(status);
        mainWindow.lblPlayerFrame().Text(frameString);
    });
}


//on background thread
void PlayerProgress::terminate() {

}

//on ui thread
Controls::IContentDialog PlayerProgress::dialog() {
    return mainWindow.playerDialog();
}


//-------------------------------------------------------------------------
//-------------------- Load Bitmap Synchronous ----------------------------
//-------------------------------------------------------------------------

void loadImageScaled(winrt::Microsoft::UI::Xaml::Controls::IImage image, winrt::hstring file) {
    winrt::Windows::Foundation::Uri uri(file);
    winrt::Microsoft::UI::Xaml::Media::Imaging::BitmapImage bitmap;
    bitmap.DecodePixelWidth((int32_t) image.as<Controls::Image>().Width());
    bitmap.UriSource(uri);
    image.Source(bitmap);
}
