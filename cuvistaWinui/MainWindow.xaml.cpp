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

#include <microsoft.ui.xaml.window.h>
#include <Shobjidl.h> //for IInitializeWithWindow for FilePicker

#if __has_include("MainWindow.g.cpp")
#include "MainWindow.g.cpp"
#endif
#if __has_include("EncodingOptionXaml.g.cpp")
#include "EncodingOptionXaml.g.cpp"
#endif

#include <filesystem>

using namespace winrt;
using namespace Windows::Graphics;
using namespace Windows::Storage;
using namespace Windows::UI::Xaml;

using namespace Microsoft::UI::Windowing;
using namespace Microsoft::UI::Dispatching;
//using namespace Windows::Foundation; --> error C2872: 'IUnknown': ambiguous symbol ???


//-------------------------------------------------------------------------
//-------------------- EncodingOptionXaml ---------------------------------
//-------------------------------------------------------------------------


template <class... Args> hstring hformat(std::format_string<Args...> fmt, Args&&... args) {
    return to_hstring(std::format(fmt, std::forward<Args>(args)...));
}

static Windows::Foundation::IInspectable box_string(const std::string& str) {
    return winrt::box_value(to_hstring(str));
}

void debugPrint(hstring str) {
    OutputDebugString(L"-------- ");
    OutputDebugString(str.c_str());
    OutputDebugString(L"\n");
}

void debugPrint(const std::string& str) {
    debugPrint(to_hstring(str));
}


namespace winrt::cuvistaWinui::implementation {

    //-------------------------------------------------------------------------
    //-------------------- EncodingOptionXaml ---------------------------------
    //-------------------------------------------------------------------------

    EncodingOptionXaml::EncodingOptionXaml() :
        EncodingOptionXaml({ EncodingDevice::AUTO, Codec::AUTO })
    {}

    EncodingOptionXaml::EncodingOptionXaml(EncodingOption option) {
        mOption = option;
    }

    hstring EncodingOptionXaml::Device() const {
        return to_hstring(mapDeviceToString[mOption.device]);
    }

    hstring EncodingOptionXaml::Codec() const {
        return to_hstring(mapCodecToString[mOption.codec]);
    }

    //-------------------------------------------------------------------------
    //-------------------- MainWindow Constructor -----------------------------
    //-------------------------------------------------------------------------

    MainWindow::MainWindow() {
        InitializeComponent(); //should not be used??

        mInputDir = Pickers::PickerLocationId::VideosLibrary;
        mOutputDir = Pickers::PickerLocationId::VideosLibrary;

        lblStatus().Text(hformat("Version {}", CUVISTA_VERSION));

        mData.console = &mData.nullStream;
        mData.printHeader = false;
        mData.printSummary = false;
        mData.probeCuda();
        mData.probeOpenCl();
        mData.collectDeviceInfo();

        //set modes list
        comboMode().Items().Append(box_string("Combined - Single Pass"));
        comboMode().Items().Append(box_string("Two Pass - Analyze then Write"));
        for (int i = 2; i <= 4; i++) {
            comboMode().Items().Append(box_string(std::format("Multi Pass - Analyze {}x", i)));
        }
        comboMode().SelectedIndex(0);

        //available devices
        int siz = (int) mData.deviceList.size();
        for (int i = 0; i < siz; i++) {
            comboDevice().Items().Append(box_string(mData.deviceList[i]->getName()));
        }

        //select highest device, set encoding options
        comboDevice().SelectedIndex(siz - 1);

        //set background color
        auto localValues = mLocalSettings.Values();
        uint8_t colorRed = localValues.Lookup(L"colorRed").try_as<uint8_t>().value_or(mData.backgroundColor.getChannel(0));
        uint8_t colorGreen = localValues.Lookup(L"colorGreen").try_as<uint8_t>().value_or(mData.backgroundColor.getChannel(1));
        uint8_t colorBlue = localValues.Lookup(L"colorBlue").try_as<uint8_t>().value_or(mData.backgroundColor.getChannel(2));
        Windows::UI::Color bg = Windows::UI::ColorHelper::FromArgb(255, colorRed, colorGreen, colorBlue);
        setBackgroundColor(bg);

        //image sequence format
        for (const OutputTypeSelector& ots : mOutputImageTypes) {
            IInspectable iis = box_value(ots.name);
            comboImageType().Items().Append(iis);
        }
        comboImageType().SelectedIndex(0);

        //NumberBox Formatters
        Windows::Globalization::NumberFormatting::DecimalFormatter decimalFormatter;
        decimalFormatter.FractionDigits(2);
        Windows::Globalization::NumberFormatting::IncrementNumberRounder inr;
        inr.Increment(0.01);
        decimalFormatter.NumberRounder(inr);
        spinRadius().NumberFormatter(decimalFormatter);

        Windows::Globalization::NumberFormatting::PercentFormatter percentFormatter;
        percentFormatter.FractionDigits(0);
        percentFormatter.NumberRounder(inr);
        spinZoomMin().NumberFormatter(percentFormatter);
        spinZoomMax().NumberFormatter(percentFormatter);

        //limits
        spinRadius().Minimum(defaults.radsecMin);
        spinRadius().Maximum(defaults.radsecMax);
        spinZoomMin().Minimum(defaults.imZoomMin - 1.0);
        spinZoomMin().Maximum(defaults.imZoomMax - 1.0);
        spinZoomMax().Minimum(defaults.imZoomMin - 1.0);
        spinZoomMax().Maximum(defaults.imZoomMax - 1.0);

        //load recent files list without opening files
        for (int idx = 0; idx < 6; idx++) {
            hstring id = L"input" + to_hstring(idx);
            hstring recentFile = localValues.Lookup(id).try_as<hstring>().value_or(L"");
            if (recentFile.empty() == false) comboInputFile().Items().Append(box_value(recentFile));
        }

        //load file when given as command line argument or when file was dropped on the app icon
        LPWSTR cmd = GetCommandLineW();
        int numArgs;
        LPWSTR* cmdArgs = CommandLineToArgvW(cmd, &numArgs);
        if (numArgs > 1) {
            std::filesystem::path fpath = cmdArgs[1];
            if (std::filesystem::exists(fpath)) {
                addInputFile(hstring(cmdArgs[1]));
            }
        }

        //stored settings
        int windowX = localValues.Lookup(L"windowX").try_as<int>().value_or(50);
        int windowY = localValues.Lookup(L"windowY").try_as<int>().value_or(50);
        int windowWidth = localValues.Lookup(L"windowWidth").try_as<int>().value_or(700);
        int windowHeight = localValues.Lookup(L"windowHeight").try_as<int>().value_or(800);
        AppWindow().MoveAndResize({ windowX, windowY, windowWidth, windowHeight });

        //min size
        OverlappedPresenter op = AppWindow().Presenter().as<OverlappedPresenter>();
        op.PreferredMinimumWidth(700);
        op.PreferredMinimumHeight(700);

        //more settings
        chkOverwrite().IsChecked(localValues.Lookup(L"overwrite").try_as<bool>().value_or(false));
        spinRadius().Value(localValues.Lookup(L"radius").try_as<double>().value_or(defaults.radsec));
        spinZoomMin().Value(localValues.Lookup(L"zoomMin").try_as<double>().value_or(defaults.zoomMin - 1.0));
        spinZoomMax().Value(localValues.Lookup(L"zoomMax").try_as<double>().value_or(defaults.zoomMax - 1.0));
        chkDynamicZoom().IsChecked(localValues.Lookup(L"zoomDynamic").try_as<bool>().value_or(true));
        chkFrameLimit().IsChecked(localValues.Lookup(L"limitEnabled").try_as<bool>().value_or(false));
        spinFrameLimit().Value(localValues.Lookup(L"limitValue").try_as<double>().value_or(defaults.frameLimit));
    }

    void MainWindow::imageInputLoaded(const IInspectable& sender, const RoutedEventArgs& args) {
        //debugPrint("image loaded");
        //mInputImageSource.SetBitmapAsync(mInputImageBitmapPlaceholder);
        //imageInput().Source(mInputImageSource);
        imageInput().Source(mInputImageBitmapPlaceholder);
    }

    void MainWindow::seek(double frac) {
        //debugPrint("seek " + std::to_string(frac));
        if (mInputReady && mReader.seek(frac) && mReader.read(mInputYUV)) {
            updateInputImage();
            inputPosition().Value(frac * 100.0);
        }
    }


    //-------------------------------------------------------------------------
    //-------------------- Event Handlers -------------------------------------
    //-------------------------------------------------------------------------

    void MainWindow::dragFile(const IInspectable& sender, const DragEventArgs& args) {
        args.AcceptedOperation(winrt::Windows::ApplicationModel::DataTransfer::DataPackageOperation::Link);
    }

    winrt::fire_and_forget MainWindow::dropFile(const IInspectable& sender, const DragEventArgs& args) {
        //debugPrint("drop");
        hstring str = winrt::Windows::ApplicationModel::DataTransfer::StandardDataFormats::StorageItems();
        if (args.DataView().Contains(str)) {
            auto items = co_await args.DataView().GetStorageItemsAsync();
            if (items.Size() > 0) {
                Windows::Storage::StorageFile storageFile = items.First().Current().try_as<Windows::Storage::StorageFile>();
                if (storageFile != nullptr) {
                    addInputFile(storageFile.Path());
                }
            }
        }
    }

    winrt::fire_and_forget MainWindow::btnOpenClick(const IInspectable& sender, const RoutedEventArgs& args) {
        //debugPrint("open");
        Pickers::FileOpenPicker openPicker;
        HWND hwnd = GetActiveWindow();
        openPicker.as<IInitializeWithWindow>()->Initialize(hwnd);
        openPicker.SuggestedStartLocation(mInputDir);
        openPicker.FileTypeFilter().ReplaceAll({ L"*", L".mp4", L".mkv" });
        Windows::Storage::StorageFile file = co_await openPicker.PickSingleFileAsync();
        if (file != nullptr) {
            addInputFile(file.Path());
        }
    }

    void MainWindow::btnColorOk(const IInspectable& sender, const RoutedEventArgs& args) {
        //debugPrint("ok");
        setBackgroundColor(colorPicker().Color());
        colorFlyout().Hide();
        radioColor().IsChecked(true);
    }

    void MainWindow::btnColorCancel(const IInspectable& sender, const RoutedEventArgs& args) {
        //debugPrint("cancel");
        colorFlyout().Hide();
    }

    void MainWindow::lblColorClick(const IInspectable& sender, const Input::TappedRoutedEventArgs& args) {
        //debugPrint("color picker");
        colorPicker().Color(mBackgroundColor);
        Controls::Primitives::FlyoutBase::ShowAttachedFlyout(sender.as<FrameworkElement>());
    }

    void MainWindow::imageBackgroundClick(const IInspectable& sender, const Input::TappedRoutedEventArgs& args) {
        winrt::Windows::Foundation::Point p = args.GetPosition(imageBackground());
        double fraction = 1.0 * p.X / imageBackground().ActualWidth();
        seek(fraction);
    }

    winrt::fire_and_forget MainWindow::btnInfoClick(const IInspectable& sender, const RoutedEventArgs& args) {
        //devices info text
        std::stringstream ss;
        mData.showDeviceInfo(ss);
        mInfoBoxString = to_hstring(ss.str());
        infoBox().Text(mInfoBoxString);

        //email link
        hstring strEmail = L"cuvista@a1.net";
        Windows::Foundation::Uri emailUri(L"mailto:" + strEmail);
        infoLinkEmail().NavigateUri(emailUri);
        Documents::Run emailRun;
        emailRun.Text(strEmail);
        infoLinkEmail().Inlines().Append(emailRun);

        //github link
        hstring strGit = L"https://github.com/RainerMtb/cuvista";
        Windows::Foundation::Uri gitUri(strGit);
        infoLinkGit().NavigateUri(gitUri);
        Documents::Run gitRun;
        gitRun.Text(strGit);
        infoLinkGit().Inlines().Append(gitRun);

        //header and footer
        infoRunHeader().Text(std::format(L"CUVISTA - Cuda Video Stabilizer, Version {}\n\u00A9 2025 Rainer Bitschi ", to_hstring(CUVISTA_VERSION)));
        infoRunFooter().Text(L"\nLicense GNU GPLv3+: GNU GPL version 3 or later");

        //button will be disabled when starting tests
        btnInfoTest().IsEnabled(true);
        infoDialog().XamlRoot(rootPanel().XamlRoot());
        Controls::ContentDialogResult result = co_await infoDialog().ShowAsync();
        //debugPrint("done");
    }

    void MainWindow::btnInfoTestClick(const IInspectable& sender, const RoutedEventArgs& args) {
        btnInfoTest().IsEnabled(false);
        mFuture = std::async(std::launch::async, runSelfTest, std::ref(*this), mData.deviceList);
    }

    void MainWindow::btnInfoCloseClick(const IInspectable& sender, const RoutedEventArgs& args) {
        infoDialog().Hide();
    }

    void MainWindow::infoDialogClosing(const Controls::ContentDialog& sender, const Controls::ContentDialogClosingEventArgs& args) {
        //debugPrint("closing");
        mFuture.wait();
    }

    void MainWindow::comboDeviceChanged(const IInspectable& sender, const Controls::SelectionChangedEventArgs& args) {
        comboEncoding().Items().Clear();
        int32_t index = comboDevice().SelectedIndex();
        std::span<EncodingOption> s = mData.deviceList[index]->encodingOptions;
        for (EncodingOption& e : s) {
            auto eox = winrt::make_self<cuvistaWinui::implementation::EncodingOptionXaml>(e);
            comboEncoding().Items().Append(eox.as<IInspectable>());
        }
        comboEncoding().SelectedIndex(0);
    }

    //-------------------------------------------------------------------------
    //-------------------- Open Input File ------------------------------------
    //-------------------------------------------------------------------------

    void MainWindow::setInputFile(const IInspectable& sender, const Controls::SelectionChangedEventArgs& args) {
        hstring inputPath = L"";
        if (args.AddedItems().Size() > 0) {
            inputPath = args.AddedItems().First().Current().as<hstring>();
        }
        if (inputPath.empty()) {
            return;
        }
        debugPrint(L"set input '" + inputPath + L"'");

        mInputReady = false;

        try {
            mReader.close();
            errorLogger().clearErrors();
            mReader.open(to_string(inputPath));
            mInputYUV = ImageYuv(mReader.h, mReader.w);

            mReader.read(mInputYUV); //read first image

            if (errorLogger().hasNoError()) {
                mReader.read(mInputYUV); //try to read again for second image
                mInputImageBitmap = Media::Imaging::WriteableBitmap(mReader.w, mReader.h);
                Windows::Storage::Streams::IBuffer pixelBuffer = mInputImageBitmap.PixelBuffer();
                mInputBGRA = ImageBGRA(mReader.h, mReader.w, pixelBuffer.Length() / mReader.h, pixelBuffer.data());
                imageInput().Source(mInputImageBitmap);
                updateInputImage();
            }

            if (errorLogger().hasError()) {
                throw AVException(errorLogger().getErrorMessage());

            } else {
                mInputReady = true;
            }

            //info about streams
            std::string str;
            for (StreamContext& sc : mReader.mInputStreams) {
                str += sc.inputStreamInfo().inputStreamSummary();
                if (sc.inputStream->index == mReader.videoStream->index) {
                    str += mReader.videoStreamSummary();
                }
            }
            size_t pos = str.size() - 1;
            while (pos > 0 && str[pos] == '\n') {
                str[pos] = '\0';
                pos--;
            }

            seek(0.1);
            lblStatus().Text(hformat("Version {}", CUVISTA_VERSION));
            texInput().Text(to_hstring(str));

        } catch (const AVException& ex) {
            imageInput().Source(imageError().Source());
            lblStatus().Text(to_hstring(ex.what()));
            texInput().Text(L"");
        }
    }

    void MainWindow::updateInputImage() {
        mInputYUV.toBaseRgb(mInputBGRA);
        mInputImageBitmap.Invalidate();
    }

    //-------------------------------------------------------------------------
    //-------------------- Start Stabililzing ---------------------------------
    //-------------------------------------------------------------------------

    void MainWindow::btnStartClick(const IInspectable& sender, const RoutedEventArgs& args) {
        debugPrint("start");

        IInspectable option = comboEncoding().SelectedValue();
        EncodingOption eo = option.as<cuvistaWinui::implementation::EncodingOptionXaml>()->mOption;
        //debugPrint(mapDeviceToString[eo.device]);
        //debugPrint(mapCodecToString[eo.codec]);

        ITaskbarList3* taskbarPtr = nullptr;
        HRESULT hr = CoCreateInstance(CLSID_TaskbarList, NULL, CLSCTX_INPROC_SERVER, IID_ITaskbarList3, (void**) (&taskbarPtr));
        HWND hwnd = GetActiveWindow();
        taskbarPtr->SetProgressValue(hwnd, 50, 100);
    }

    //-------------------------------------------------------------------------
    //-------------------- Private Methods ------------------------------------
    //-------------------------------------------------------------------------

    void MainWindow::addInputFile(hstring file) {
        if (file.size() > 0) {
            //debugPrint(L"add file '" + file + L"'");
            auto item = winrt::box_value(file);
            uint32_t idx;
            bool isFound = comboInputFile().Items().IndexOf(item, idx);
            comboInputFile().Items().InsertAt(0, item);
            comboInputFile().SelectedIndex(0);
            if (isFound) {
                comboInputFile().Items().RemoveAt(idx + 1);
            }
            if (comboInputFile().Items().Size() > 6) {
                comboInputFile().Items().RemoveAtEnd();
            }
        }
    }

    void MainWindow::setBackgroundColor(Windows::UI::Color color) {
        mBackgroundColor = color;
        Media::SolidColorBrush brush(mBackgroundColor);
        imageBackground().Background(brush);
        lblColor().Background(brush);
    }

    void MainWindow::print(const std::string& str) {
        infoBoxAppendText(str);
    }

    void MainWindow::printNewLine() {
        infoBoxAppendText("\n");
    }

    void MainWindow::infoBoxAppendText(std::string str) {
        mInfoBoxString = mInfoBoxString + to_hstring(str);
        mDispatcher.TryEnqueue([&, str = mInfoBoxString] {
            infoBox().Text(str);
            infoScroller().ScrollToVerticalOffset(infoScroller().ScrollableHeight());
        });
    }

    void MainWindow::btnResetClick(const IInspectable& sender, const RoutedEventArgs& args) {
        setBackgroundColor(Windows::UI::ColorHelper::FromArgb(255, defaults.bgColorRed, defaults.bgColorGreen, defaults.bgColorBlue));

        chkOverwrite().IsChecked(false);
        spinRadius().Value(defaults.radsec);
        spinZoomMin().Value(defaults.zoomMin - 1.0);
        spinZoomMax().Value(defaults.zoomMax - 1.0);
        chkDynamicZoom().IsChecked(true);
        chkFrameLimit().IsChecked(false);
        spinFrameLimit().Value(defaults.frameLimit);

        mReader.close();
        texInput().Text(L"");
        comboInputFile().Items().Clear();
        inputPosition().Value(0.0);
        imageInput().Source(mInputImageBitmapPlaceholder);
    }

    //-------------------------------------------------------------------------
    //-------------------- Destructor -----------------------------------------
    //-------------------------------------------------------------------------

    void MainWindow::windowClosedEvent(const IInspectable& sender, const WindowEventArgs& args) {
        //debugPrint("closed");
        auto localValues = mLocalSettings.Values();
        localValues.Insert(L"colorRed", box_value(mBackgroundColor.R));
        localValues.Insert(L"colorGreen", box_value(mBackgroundColor.G));
        localValues.Insert(L"colorBlue", box_value(mBackgroundColor.B));

        PointInt32 pos = AppWindow().Position();
        localValues.Insert(L"windowX", box_value(pos.X));
        localValues.Insert(L"windowY", box_value(pos.Y));
        SizeInt32 siz = AppWindow().Size();
        localValues.Insert(L"windowWidth", box_value(siz.Width));
        localValues.Insert(L"windowHeight", box_value(siz.Height));

        localValues.Insert(L"overwrite", box_value(chkOverwrite().IsChecked()));
        localValues.Insert(L"radius", box_value(spinRadius().Value()));
        localValues.Insert(L"zoomMin", box_value(spinZoomMin().Value()));
        localValues.Insert(L"zoomMax", box_value(spinZoomMax().Value()));
        localValues.Insert(L"zoomDynamic", box_value(chkDynamicZoom().IsChecked()));
        localValues.Insert(L"limitEnabled", box_value(chkFrameLimit().IsChecked()));
        localValues.Insert(L"limitValue", box_value(spinFrameLimit().Value()));

        Controls::ItemCollection recentFiles = comboInputFile().Items();
        uint32_t idx = 0;
        for (; idx < recentFiles.Size(); idx++) {
            hstring id = L"input" + to_hstring(idx);
            IInspectable file = recentFiles.GetAt(idx);
            localValues.Insert(id, box_value(file));
        }
        for (; idx < 6; idx++) {
            hstring id = L"input" + to_hstring(idx);
            localValues.Insert(id, box_value(L""));
        }
    }
}
