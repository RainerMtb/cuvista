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


void debugPrint(winrt::hstring str);
void debugPrint(const std::string& str);

namespace winrt::cuvistaWinui::implementation {
    
    using namespace winrt::Microsoft::UI::Xaml;
    using namespace Windows::Graphics::Imaging;

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

        void btnOpenClick(const IInspectable& sender, const RoutedEventArgs& args);
        void btnColorOk(const IInspectable& sender, const RoutedEventArgs& args);
        void btnColorCancel(const IInspectable& sender, const RoutedEventArgs& args);

        void lblColorClick(const IInspectable& sender, const Input::TappedRoutedEventArgs& args);
        void imageBackgroundClick(const IInspectable& sender, const Input::TappedRoutedEventArgs& args);

        winrt::fire_and_forget btnStartClick(const IInspectable& sender, const RoutedEventArgs& args);
        winrt::fire_and_forget btnInfoClick(const IInspectable& sender, const RoutedEventArgs& args);
        void btnResetClick(const IInspectable& sender, const RoutedEventArgs& args);

        void btnInfoCloseClick(const IInspectable& sender, const RoutedEventArgs& args);
        void btnInfoTestClick(const IInspectable& sender, const RoutedEventArgs& args);
        void infoDialogClosing(const Controls::ContentDialog& sender, const Controls::ContentDialogClosingEventArgs& args);

        winrt::fire_and_forget dropFile(const IInspectable& sender, const DragEventArgs& args);
        void dragFile(const IInspectable& sender, const DragEventArgs& args);
        void setInputFile(const IInspectable& sender, const Controls::SelectionChangedEventArgs& args);

        void comboDeviceChanged(const IInspectable& sender, const Controls::SelectionChangedEventArgs& args);

        void imageInputLoaded(const IInspectable& sender, const RoutedEventArgs& args);

        void print(const std::string& str) override;
        void printNewLine() override;

        void windowClosedEvent(const IInspectable& sender, const WindowEventArgs& args);

    private:
        Windows::UI::Color mBackgroundColor;
        Windows::Storage::ApplicationDataContainer mLocalSettings = Windows::Storage::ApplicationData::Current().LocalSettings();
        Microsoft::UI::Dispatching::DispatcherQueue mDispatcher = Microsoft::UI::Dispatching::DispatcherQueue::GetForCurrentThread();

        MainData mData;
        ImageYuv mInputYUV;
        ImageBGRA mInputBGRA;
        FFmpegReader mReader;

        std::shared_ptr<MovieWriter> mWriter;
        std::shared_ptr<MovieFrame> mFrame;
        std::shared_ptr<FrameExecutor> mExecutor;

        bool mInputReady = false;
        bool mOutputReady = false;
		hstring mInputFile = L"";

        //SoftwareBitmap mInputImageBitmapPlaceholder = SoftwareBitmap(BitmapPixelFormat::Bgra8, 100, 100, BitmapAlphaMode::Premultiplied);
        //SoftwareBitmap mInputImageBitmap = mInputImageBitmapPlaceholder;
        //Media::Imaging::SoftwareBitmapSource mInputImageSource;
        Media::Imaging::WriteableBitmap mInputImageBitmapPlaceholder = Media::Imaging::WriteableBitmap(100, 100);
        Media::Imaging::WriteableBitmap mInputImageBitmap = mInputImageBitmapPlaceholder;

        hstring mInfoBoxString;
        std::future<void> mFuture = std::async([&] {});
        std::future<LoopResult> mLoopFuture = std::async([&] { return LoopResult::LOOP_NONE; });

        std::vector<OutputTypeSelector> mOutputImageTypes = {
            { L"BMP", OutputType::SEQUENCE_BMP },
            { L"JPG", OutputType::SEQUENCE_JPG }
        };

        void setBackgroundColor(Windows::UI::Color color);
        void addInputFile(hstring file);

        void seek(double frac);
        void updateInputImage();

        void infoBoxAppendText(std::string str);

        LoopResult runLoop();

    public:
    };
}

namespace winrt::cuvistaWinui::factory_implementation
{
    struct MainWindow : MainWindowT<MainWindow, implementation::MainWindow> {};

    struct EncodingOptionXaml : EncodingOptionXamlT<EncodingOptionXaml, implementation::EncodingOptionXaml> {};
}
