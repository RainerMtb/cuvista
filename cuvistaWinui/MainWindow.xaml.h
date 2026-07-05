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
#include "CustomRuntimeXaml.g.h"
#undef min
#undef max

#include <any>
#include "MovieFrame.hpp"
#include "FrameExecutor.hpp"
#include "AppImage.hpp"
#include "ProgressDisplay.hpp"


namespace winrt::cuvistaWinui::implementation {
    
    using namespace winrt::Microsoft::UI::Xaml;

    class ProgressDialog : public ProgressDisplay {

    public:
        ProgressDialog(int interval) :
            ProgressDisplay(interval)
        {}

        virtual winrt::Microsoft::UI::Xaml::Controls::IContentDialog dialog() = 0;
    };

    struct LoopResultWrapper : winrt::implements<LoopResultWrapper, winrt::Windows::Foundation::IInspectable> {
        LoopResult result;

        LoopResultWrapper(LoopResult result) :
            result { result } 
        {}
    };

    struct CustomRuntimeXaml : CustomRuntimeXamlT<CustomRuntimeXaml> {

    private:
        //original property
        std::any object;

        //name to display
        hstring name;

    public:
        //constructor defined in IDL
        CustomRuntimeXaml();

        //constructor used in code
        CustomRuntimeXaml(std::string name, std::any object);

        //property getters called from XAML
        hstring displayName() const;

        //get user property
        template <class T> T get();
    };

    struct MainWindow : MainWindowT<MainWindow>, util::MessagePrinter, public UserInput {

        ImageXamlBGRA mProgressInput;
        ImageXamlBGRA mProgressOutput;
        bool mPlayerPaused = false;
        int mAudioStreamIndex = -1;
        double mAudioGain = 1.0;

        MainWindow();
        fire_and_forget OnLoading(const FrameworkElement& sender, const IInspectable& args);

        void btnColorOk(const IInspectable& sender, const RoutedEventArgs& args);
        void btnColorCancel(const IInspectable& sender, const RoutedEventArgs& args);
        void lblColorClick(const IInspectable& sender, const Input::TappedRoutedEventArgs& args);
        fire_and_forget imageInputClick(const IInspectable& sender, const Input::TappedRoutedEventArgs& args);

        void btnOpenClick(const IInspectable& sender, const RoutedEventArgs& args);
        fire_and_forget btnStartClick(const IInspectable& sender, const RoutedEventArgs& args);
        fire_and_forget btnInfoClick(const IInspectable& sender, const RoutedEventArgs& args);
        void btnResetClick(const IInspectable& sender, const RoutedEventArgs& args);
        void btnStopClick(const IInspectable& sender, const RoutedEventArgs& args);

        void btnPlayerPauseClick(const IInspectable& sender, const RoutedEventArgs& args);
        void btnPlayerStopClick(const IInspectable& sender, const RoutedEventArgs& args);

        fire_and_forget statusLinkClick(const IInspectable& sender, const RoutedEventArgs& args);

        void btnInfoCloseClick(const IInspectable& sender, const RoutedEventArgs& args);
        void btnInfoTestClick(const IInspectable& sender, const RoutedEventArgs& args);
        void infoDialogClosing(const Controls::ContentDialog& sender, const Controls::ContentDialogClosingEventArgs& args);

        fire_and_forget dropFile(const IInspectable& sender, const DragEventArgs& args);
        void dragFile(const IInspectable& sender, const DragEventArgs& args);
        void setInputFile(const IInspectable& sender, const Controls::SelectionChangedEventArgs& args);

        void comboDeviceChanged(const IInspectable& sender, const Controls::SelectionChangedEventArgs& args);
        void modeSelectionChanged(const IInspectable& sender, const Controls::SelectionChangedEventArgs& args);

        void windowClosedEvent(const IInspectable& sender, const WindowEventArgs& args);
        void windowSizeChanged(const IInspectable& sender, const WindowSizeChangedEventArgs& args);
        void dialogClosedEvent(const Controls::ContentDialog& sender, const Controls::ContentDialogClosedEventArgs& args); 
        
        void sliderVolumeChanged(const IInspectable& sender, const Controls::Primitives::RangeBaseValueChangedEventArgs& args);
        void imageGridResize(const IInspectable& sender, const SizeChangedEventArgs& args);

    private:
        Windows::UI::Color mBackgroundColor;
        Windows::Storage::ApplicationDataContainer mLocalSettings = Windows::Storage::ApplicationData::Current().LocalSettings();

        MainData mData;
        ImageYuv mInput;
        ImageXamlBGRA mInputBGRA;
        std::shared_ptr<MovieReader> mReader;
        std::shared_ptr<MovieWriter> mWriter;
        std::shared_ptr<MovieFrame> mFrame;
        std::shared_ptr<FrameExecutor> mExecutor;
        std::shared_ptr<ProgressDialog> mProgress;

        bool mInputReady = false;
        double inputVideoFraction = 0.0;
		hstring mInputFile = L"";
        hstring mOutputFile = L"";

        hstring mInfoBoxString;
        std::future<void> mFutureInfo = std::async([&] {});

        std::map<hstring, OutputOption> mOutputImageTypeMap = {
            { L"BMP", OutputOption::IMAGE_BMP },
            { L"JPG", OutputOption::IMAGE_JPG }
        };

        std::map<int, int> mAudioTrackMap;

        UserInputEnum mUserInput = UserInputEnum::CONTINUE;

        void setBackgroundColor(Windows::UI::Color color);
        void addInputFile(hstring file);

        fire_and_forget seekAsync(double frac);

        winrt::Windows::Foundation::IAsyncOperation<Controls::ContentDialogResult> showErrorDialogAsync(hstring title, hstring content);
        winrt::Windows::Foundation::IAsyncOperation<Controls::ContentDialogResult> showErrorDialogAsync(const std::string& title, const std::string& content);

        void infoBoxAppendText(std::string str);


        //--------------------------------------
        //-------- class overrides -------------
        //--------------------------------------
   
        //print message for device info
        void print(const std::string& str) override;
        void printNewLine() override;

        //user input to cancel loop
        UserInputEnum checkState() override;
    };
}

namespace winrt::cuvistaWinui::factory_implementation
{
    struct MainWindow : MainWindowT<MainWindow, implementation::MainWindow> {};

    struct CustomRuntimeXaml : CustomRuntimeXamlT<CustomRuntimeXaml, implementation::CustomRuntimeXaml> {};
}