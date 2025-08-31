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

        //load file when given as command line argument or when file was ddropped on the app icon
        LPWSTR cmd = GetCommandLineW();
        int numArgs;
        LPWSTR* cmdArgs = CommandLineToArgvW(cmd, &numArgs);
        if (numArgs > 1) {
            std::filesystem::path fpath = cmdArgs[1];
            if (std::filesystem::exists(fpath)) {
                addInputFile(hstring(cmdArgs[1]));
            }
        }

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
        spinRadius().Minimum(mData.limits.radsecMin);
        spinRadius().Maximum(mData.limits.radsecMax);
        spinZoomMin().Minimum(mData.limits.imZoomMin - 1.0);
        spinZoomMin().Maximum(mData.limits.imZoomMax - 1.0);
        spinZoomMax().Minimum(mData.limits.imZoomMin - 1.0);
        spinZoomMax().Maximum(mData.limits.imZoomMax - 1.0);

        /*
        * ---------------------------
        */

        int w = 700;
        int h = 800;
        SizeInt32 rect = { w, h };
        AppWindow().Resize(rect);

        OverlappedPresenter op = AppWindow().Presenter().as<OverlappedPresenter>();
        op.PreferredMinimumWidth(w);
        op.PreferredMinimumHeight(h);

        texInput().Text(to_hstring("sfsdf \nsdfrgfd \ngdfgdf gdf \ngdfgd\n sfsdfs\nssfsd sdfsd sdf\n sdfqgerh \nerhdehd\ndffdgd\ndfgfdg\ndfgfdg"));

        int posx = localValues.Lookup(L"posx").try_as<int>().value_or(10);
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
    }

    void MainWindow::btnColorCancel(const IInspectable& sender, const RoutedEventArgs& args) {
        //debugPrint("cancel");
        colorFlyout().Hide();
    }

    void MainWindow::lblColorClick(const IInspectable& sender, const Input::TappedRoutedEventArgs& args) {
        //debugPrint("color picker");
        colorPicker().Color(mColor);
        Controls::Primitives::FlyoutBase::ShowAttachedFlyout(sender.as<FrameworkElement>());
    }

    void MainWindow::imageBackgroundClick(const IInspectable& sender, const Input::TappedRoutedEventArgs& args) {
        winrt::Windows::Foundation::Point p = args.GetPosition(imageBackground());
        double fraction = 100.0 * p.X / imageBackground().ActualWidth();
        debugPrint("seek " + std::to_string(fraction));
        inputPosition().Value(fraction);
    }

    void MainWindow::dynZoomClick(const IInspectable& sender, const RoutedEventArgs& args) {
        //debugPrint("dynamic zoom");
        spinZoomMax().IsEnabled(chkDynamicZoom().IsChecked().GetBoolean());
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

    void MainWindow::btnResetClick(const IInspectable& sender, const RoutedEventArgs& args) {

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
        debugPrint("set input");
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
            debugPrint(L"open file '" + file + L"'");
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
        mColor = color;
        Media::SolidColorBrush brush(mColor);
        imageBackground().Background(brush);
        lblColor().Background(brush);
    }

    void MainWindow::print(const std::string& str) {
        appendText(str);
    }

    void MainWindow::printNewLine() {
        appendText("\n");
    }

    void MainWindow::appendText(std::string str) {
        mInfoBoxString = mInfoBoxString + to_hstring(str);
        mDispatcher.TryEnqueue([&, str = mInfoBoxString] {
            infoBox().Text(str);
            infoScroller().ScrollToVerticalOffset(infoScroller().ScrollableHeight());
        });
    }
}
