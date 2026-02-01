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

#pragma once

#include <chrono>
#include "ProgressDisplay.hpp"
#include "FrameExecutor.hpp"


//util functions
void debugPrint(winrt::hstring str);
void debugPrint(const std::string& str);

template <class... Args> winrt::hstring hformat(std::format_string<Args...> fmt, Args&&... args) {
    return winrt::to_hstring(std::format(fmt, std::forward<Args>(args)...));
}

winrt::hstring selectFileOpen(HWND hwnd);
winrt::hstring selectFileSave(HWND hwnd, bool alwaysOverwrite);
winrt::hstring selectFolder(HWND hwnd);

winrt::Windows::Foundation::IInspectable box_string(const std::string& str);
void loadImageScaled(winrt::Microsoft::UI::Xaml::Controls::IImage image, winrt::hstring file);


//progress dialog window
struct ITaskbarList3;
struct MainWindow;

class ProgressDialog : public ProgressDisplay {

public:
    ProgressDialog(int interval) :
        ProgressDisplay(interval)
    {}

    virtual winrt::Microsoft::UI::Xaml::Controls::IContentDialog dialog() = 0;
};


class ProgressGui : public ProgressDialog {

private:
    winrt::cuvistaWinui::implementation::MainWindow& mainWindow;
    ITaskbarList3* mTaskbarPtr = nullptr;
    HWND hwnd;
    FrameExecutor& mExecutor;
    std::chrono::steady_clock::time_point mTimePoint;

public:
    ProgressGui(winrt::cuvistaWinui::implementation::MainWindow& mainWindow, HWND hwnd, FrameExecutor& executor) :
        ProgressDialog(50),
        mainWindow { mainWindow },
        hwnd { hwnd },
        mExecutor { executor }
    {}

    void init(const ProgressInfo& progress) override;
    void update(const ProgressInfo& progress, bool force) override;
    void updateStatus(const std::string& msg) override;
    void terminate() override;

    winrt::Microsoft::UI::Xaml::Controls::IContentDialog dialog() override;
};


//player dialog window
class PlayerWriter : public NullWriter {

private:
    winrt::cuvistaWinui::implementation::MainWindow& mainWindow;
    FrameExecutor& mExecutor;
    std::chrono::time_point<std::chrono::steady_clock> mNextPts;
    std::shared_ptr<OutputStreamContext> mAudioContext;
    winrt::Windows::Media::Audio::IAudioGraph mAudioGraph;
    winrt::Windows::Media::Audio::IAudioFrameInputNode mAudioInputNode;
    bool mPlayAudio;

public:
    PlayerWriter(winrt::cuvistaWinui::implementation::MainWindow& mainWindow, FrameExecutor& executor, MainData& data, MovieReader& reader) :
        NullWriter(data, reader),
        mainWindow { mainWindow },
        mExecutor { executor },
        mPlayAudio { false }
    {}

    //writer
    void open(OutputOption outputOption) override;
    void start() override;
    void writeOutput(const FrameExecutor& executor) override;
    bool flush() override;
    void close() override;
};


class PlayerProgress : public ProgressDialog {

private:
    winrt::cuvistaWinui::implementation::MainWindow& mainWindow;
    FrameExecutor& mExecutor;

public:
    PlayerProgress(winrt::cuvistaWinui::implementation::MainWindow& mainWindow, FrameExecutor& executor) :
        ProgressDialog(0),
        mainWindow { mainWindow },
        mExecutor { executor }
    {}

    void update(const ProgressInfo& progress, bool force) override;
    void terminate() override;

    //dialog
    winrt::Microsoft::UI::Xaml::Controls::IContentDialog dialog() override;
};