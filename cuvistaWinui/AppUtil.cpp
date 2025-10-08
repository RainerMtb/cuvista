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

#include "pch.h"
#include "MainWindow.xaml.h"
#include <ShObjIdl.h>
#undef min
#undef max

#include "AppUtil.hpp"

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

    return out;
}

winrt::hstring selectFileSave(HWND hwnd, bool alwaysOverwrite) {
    winrt::hstring out = L"";
    IFileDialog* ifd = nullptr;

    HRESULT result = CoCreateInstance(CLSID_FileSaveDialog, NULL, CLSCTX_INPROC_SERVER, IID_IFileSaveDialog, (void**) (&ifd));
    if (result == S_OK) {
        FILEOPENDIALOGOPTIONS op = alwaysOverwrite ? 0 : FOS_OVERWRITEPROMPT;
        ifd->SetOptions(op);
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

    return out;
}

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
        mExecutor.getInput(idx, mainWindow.mProgressInputBgra);
        winrt::hstring hstr = winrt::to_hstring(mExecutor.mFrame.ptsForFrameAsString(idx));
        mainWindow.DispatcherQueue().TryEnqueue([&, hstr] { 
            mainWindow.textTimeInput().Text(hstr); 
            mainWindow.mProgressInputBitmap.Invalidate();
        });
    }

    if (imageDue && progress.writeIndex > 0) {
        mTimePoint = timePointNow;
        int64_t idx = progress.writeIndex - 1;
        mExecutor.getWarped(idx, mainWindow.mProgressOutputBgra);
        winrt::hstring hstr = winrt::to_hstring(mExecutor.mFrame.ptsForFrameAsString(idx));
        mainWindow.DispatcherQueue().TryEnqueue([&, hstr] { 
            mainWindow.textTimeOutput().Text(hstr); 
            mainWindow.mProgressOutputBitmap.Invalidate();
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
    }
}


//-------------------------------------------------------------------------
//-------------------- Load Bitmap and Scale to Fit Destination -----------
//-------------------------------------------------------------------------

using namespace winrt::Windows::Graphics::Imaging;
using namespace winrt::Windows::Storage::Streams;

winrt::Windows::Foundation::IAsyncOperation<SoftwareBitmap> loadScaledImage(winrt::hstring file, int w, int h) {
    int s = std::min(w, h);

    //load image scaled
    winrt::Windows::Foundation::Uri uri(file);
    IRandomAccessStream ras = co_await RandomAccessStreamReference::CreateFromUri(uri).OpenReadAsync();
    BitmapDecoder decoder = co_await BitmapDecoder::CreateAsync(ras);
    BitmapTransform transform;
    transform.ScaledWidth(s);
    transform.ScaledHeight(s);
    PixelDataProvider pixelProvider = co_await decoder.GetPixelDataAsync(
        BitmapPixelFormat::Bgra8, BitmapAlphaMode::Premultiplied, transform,
        ExifOrientationMode::IgnoreExifOrientation, ColorManagementMode::DoNotColorManage
    );

    //access pixel data
    winrt::com_array<uint8_t> data = pixelProvider.DetachPixelData();
    unsigned char* src = data.data();
    SoftwareBitmap sb(BitmapPixelFormat::Bgra8, w, h, BitmapAlphaMode::Straight);
    BitmapBuffer buffer = sb.LockBuffer(BitmapBufferAccessMode::Write);
    winrt::Windows::Foundation::IMemoryBufferReference ref = buffer.CreateReference();
    
    //set all pixels transparent first
    std::fill(ref.data(), ref.data() + ref.Capacity(), 0);

    //copy pixel data
    unsigned char* dest = ref.data() + (h - s) * w * 2 + (w - s) * 2;
    for (int i = 0; i < s; i++) {
        std::copy(src, src + 4 * s, dest);
        dest += 4 * w;
        src += 4 * s;
    }
    ref.Close();
    buffer.Close();

    co_return sb;
}


//-------------------------------------------------------------------------
//-------------------- Load Bitmap Synchronous ----------------------------
//-------------------------------------------------------------------------

ImageBGRA loadImage(winrt::hstring file) {
    winrt::Windows::Foundation::Uri uri(file);
    IRandomAccessStream ras = RandomAccessStreamReference::CreateFromUri(uri).OpenReadAsync().get();
    BitmapDecoder decoder = BitmapDecoder::CreateAsync(ras).get();
    PixelDataProvider pixelProvider = decoder.GetPixelDataAsync().get();
    winrt::com_array<uint8_t> pixelData = pixelProvider.DetachPixelData();

    ImageBGRA out(decoder.PixelHeight(), decoder.PixelWidth());
    std::copy(pixelData.data(), pixelData.data() + pixelData.size(), out.data());
    return out;
}