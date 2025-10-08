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

#include <hstring.h>

void debugPrint(winrt::hstring str);
void debugPrint(const std::string& str);

winrt::Windows::Foundation::IInspectable box_string(const std::string& str);

winrt::hstring selectFileOpen(HWND hwnd);
winrt::hstring selectFileSave(HWND hwnd, bool alwaysOverwrite);
winrt::hstring selectFolder(HWND hwnd);


#include "ProgressDisplay.hpp"
#include "FrameExecutor.hpp"

struct ITaskbarList3;
struct winrt::cuvistaWinui::implementation::MainWindow;

class ProgressGui : public ProgressDisplay {

private:
    winrt::cuvistaWinui::implementation::MainWindow& mainWindow;
    ITaskbarList3* mTaskbarPtr = nullptr;
    HWND hwnd;
    FrameExecutor& mExecutor;
    std::chrono::steady_clock::time_point mTimePoint;

public:
    ProgressGui(winrt::cuvistaWinui::implementation::MainWindow& mainWindow, HWND hwnd, FrameExecutor& executor) :
        ProgressDisplay(50),
        mainWindow { mainWindow },
        hwnd { hwnd },
        mExecutor { executor }
    {}

    void init() override;
    void update(const ProgressInfo& progress, bool force) override;
    void updateStatus(const std::string& msg) override;
    void terminate() override;
};

winrt::Windows::Foundation::IAsyncOperation<winrt::Windows::Graphics::Imaging::SoftwareBitmap> loadScaledImage(winrt::hstring file, int w, int h);

ImageBGRA loadImage(winrt::hstring file);