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

#include "MainWindow.g.h"
#include "EncodingOptionXaml.g.h"
#undef min
#undef max

#include "MovieFrame.hpp"
#include "MovieReader.hpp"
#include "FrameResult.hpp"
#include "FrameExecutor.hpp"


void debugPrint(winrt::hstring str);
void debugPrint(const std::string& str);

namespace winrt::cuvistaWinui::implementation {
    
    using namespace winrt::Microsoft::UI::Xaml;

    struct EncodingOptionXaml : EncodingOptionXamlT<EncodingOptionXaml> {

        //constructor defined in IDL
        EncodingOptionXaml();
        //constructor used in code
        EncodingOptionXaml(EncodingOption option);

        //property getters called from XAML
        hstring Device() const;
        hstring Codec() const;

        //original property
        EncodingOption mOption;
    };

    struct OutputTypeSelector {
        hstring name;
        OutputType outputType;
    };

    struct MainWindow : MainWindowT<MainWindow>, util::MessagePrinter {

        MainWindow();

        winrt::fire_and_forget btnOpenClick(const IInspectable& sender, const RoutedEventArgs& args);
        void btnColorOk(const IInspectable& sender, const RoutedEventArgs& args);
        void btnColorCancel(const IInspectable& sender, const RoutedEventArgs& args);

        void lblColorClick(const IInspectable& sender, const Input::TappedRoutedEventArgs& args);
        void imageBackgroundClick(const IInspectable& sender, const Input::TappedRoutedEventArgs& args);

        void btnStartClick(const IInspectable& sender, const RoutedEventArgs& args);
        winrt::fire_and_forget btnInfoClick(const IInspectable& sender, const RoutedEventArgs& args);
        void btnResetClick(const IInspectable& sender, const RoutedEventArgs& args);

        void btnInfoCloseClick(const IInspectable& sender, const RoutedEventArgs& args);
        void btnInfoTestClick(const IInspectable& sender, const RoutedEventArgs& args);
        void infoDialogClosing(const Controls::ContentDialog& sender, const Controls::ContentDialogClosingEventArgs& args);

        winrt::fire_and_forget dropFile(const IInspectable& sender, const DragEventArgs& args);
        void dragFile(const IInspectable& sender, const DragEventArgs& args);
        void setInputFile(const IInspectable& sender, const Controls::SelectionChangedEventArgs& args);

        void comboDeviceChanged(const IInspectable& sender, const Controls::SelectionChangedEventArgs& args);
        void dynZoomClick(const IInspectable& sender, const RoutedEventArgs& args);

        void print(const std::string& str) override;
        void printNewLine() override;

    private:
        Windows::UI::Color mColor;
        Windows::Storage::ApplicationDataContainer mLocalSettings = Windows::Storage::ApplicationData::Current().LocalSettings();
        Windows::Storage::Pickers::PickerLocationId mInputDir;
        Windows::Storage::Pickers::PickerLocationId mOutputDir;
        Microsoft::UI::Dispatching::DispatcherQueue mDispatcher = Microsoft::UI::Dispatching::DispatcherQueue::GetForCurrentThread();

        MainData mData;
        hstring mInfoBoxString;
        std::future<void> mFuture = std::async([&] {});

        std::vector<OutputTypeSelector> mOutputImageTypes = {
            { L"BMP", OutputType::SEQUENCE_BMP },
            { L"JPG", OutputType::SEQUENCE_JPG }
        };

        void setBackgroundColor(Windows::UI::Color color);

        void addInputFile(hstring file);

        void appendText(std::string str);
    };
}

namespace winrt::cuvistaWinui::factory_implementation
{
    struct MainWindow : MainWindowT<MainWindow, implementation::MainWindow> {};

    struct EncodingOptionXaml : EncodingOptionXamlT<EncodingOptionXaml, implementation::EncodingOptionXaml> {};
}
