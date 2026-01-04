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

#if __has_include("MainWindow.g.cpp")
#include "MainWindow.g.cpp"
#endif
#if __has_include("CustomRuntimeXaml.g.cpp")
#include "CustomRuntimeXaml.g.cpp"
#endif

#include <filesystem>
#include "FrameResult.hpp"
#include "FrameExecutor.hpp"
#include "CudaWriter.hpp"
#include "AppUtil.hpp"


using namespace winrt;
using namespace Windows::Graphics;
using namespace Windows::Storage;
using namespace Windows::UI::Xaml;

using namespace Microsoft::UI::Windowing;
//using namespace Windows::Foundation; --> error C2872: 'IUnknown': ambiguous symbol ???


namespace winrt::cuvistaWinui::implementation {

    //-------------------------------------------------------------------------
    //-------------------- EncodingOptionXaml ---------------------------------
    //-------------------------------------------------------------------------

    CustomRuntimeXaml::CustomRuntimeXaml() :
        CustomRuntimeXaml({}, {})
    {}

    CustomRuntimeXaml::CustomRuntimeXaml(std::string name, std::any object) :
        name { to_hstring(name) },
        object { object }
    {}

    hstring CustomRuntimeXaml::displayName() const {
        return name;
    }

    template <class T> T CustomRuntimeXaml::get() {
        return std::any_cast<T>(object);
    }


    //-------------------------------------------------------------------------
    //-------------------- MainWindow Constructor -----------------------------
    //-------------------------------------------------------------------------

    MainWindow::MainWindow() {
        InitializeComponent(); //should not be used??
        lblStatus().Text(hformat("Version {}", CUVISTA_VERSION));

        mData.console = &mData.nullStream;
        mData.printHeader = false;
        mData.printSummary = false;
        mData.deviceInfoCuda = mData.probeCuda();
        mData.deviceInfoOpenCl = mData.probeOpenCl();
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
        for (const auto& oit : mOutputImageTypeMap) {
            IInspectable iis = box_value(oit.first);
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
        spinRadius().Minimum(defaultParam.radsecMin);
        spinRadius().Maximum(defaultParam.radsecMax);
        spinZoomMin().Minimum(defaultParam.imZoomMin - 1.0);
        spinZoomMin().Maximum(defaultParam.imZoomMax - 1.0);
        spinZoomMax().Minimum(defaultParam.imZoomMin - 1.0);
        spinZoomMax().Maximum(defaultParam.imZoomMax - 1.0);

        //load recent files list without opening files
        for (int idx = 0; idx < 6; idx++) {
            hstring id = L"input" + to_hstring(idx);
            hstring recentFile = localValues.Lookup(id).try_as<hstring>().value_or(L"");
            if (recentFile.empty() == false) comboInputFile().Items().Append(box_value(recentFile));
        }

        //stored settings
        int minW = 700;
        int minH = 750;
        int windowX = localValues.Lookup(L"windowX").try_as<int>().value_or(50);
        int windowY = localValues.Lookup(L"windowY").try_as<int>().value_or(50);
        int windowWidth = localValues.Lookup(L"windowWidth").try_as<int>().value_or(0);
        int windowHeight = localValues.Lookup(L"windowHeight").try_as<int>().value_or(0);
        windowWidth = std::max(windowWidth, minW);
        windowHeight = std::max(windowHeight, minH);
        AppWindow().MoveAndResize({ windowX, windowY, windowWidth, windowHeight });

        //min size
        OverlappedPresenter op = AppWindow().Presenter().as<OverlappedPresenter>();
        op.PreferredMinimumWidth(minW);
        op.PreferredMinimumHeight(minH);

        //more settings
        chkOverwrite().IsChecked(localValues.Lookup(L"overwrite").try_as<bool>().value_or(false));
        spinRadius().Value(localValues.Lookup(L"radius").try_as<double>().value_or(defaultParam.radsec));
        spinZoomMin().Value(localValues.Lookup(L"zoomMin").try_as<double>().value_or(defaultParam.zoomMin - 1.0));
        spinZoomMax().Value(localValues.Lookup(L"zoomMax").try_as<double>().value_or(defaultParam.zoomMax - 1.0));
        chkDynamicZoom().IsChecked(localValues.Lookup(L"zoomDynamic").try_as<bool>().value_or(true));
        spinFrameLimit().Value(localValues.Lookup(L"limitValue").try_as<double>().value_or(defaultParam.frameLimit));
        //chkFrameLimit().IsChecked(localValues.Lookup(L"limitEnabled").try_as<bool>().value_or(false));

        sliderQuality().Value(localValues.Lookup(L"encodingQuality").try_as<double>().value_or(defaultParam.encodingQuality));
        sliderLevels().Value(3.0);

        //load file when given as command line argument or when file was dropped on the app icon
        LPWSTR cmd = GetCommandLineW();
        int numArgs;
        LPWSTR* cmdArgs = CommandLineToArgvW(cmd, &numArgs);
        if (numArgs > 1) {
            std::filesystem::path p(cmdArgs[1]);
            if (std::filesystem::exists(p)) {
                addInputFile(cmdArgs[1]);
            }
        }
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
        //debugPrint(L"set input '" + inputPath + L"'");

        mInputReady = false;
        mInputFile = inputPath;
        comboAudioTrack().Items().Clear();
        comboAudioTrack().Items().Append(box_string("No Audio"));
        mAudioTrackMap.clear();

        try {
            mReader.close();
            errorLogger().clear();
            mReader.open(to_string(inputPath));
            mInputYUV = ImageYuv(mReader.h, mReader.w);

            mReader.read(mInputYUV); //read first image

            if (errorLogger().hasNoError()) {
                mReader.read(mInputYUV); //try to read again for second image
                mInputBGRA = ImageXamlBGRA::create(imageInput(), mReader.h, mReader.w);
                mInputYUV.toBaseRgb(mInputBGRA);
            }

            if (errorLogger().hasError()) {
                throw AVException(errorLogger().getErrorMessage());

            } else {
                mInputReady = true;
            }

            //info about streams
            std::string str;
            for (StreamContext& sc : mReader.mInputStreams) {
                StreamInfo info = sc.inputStreamInfo();
                str += info.inputStreamSummary();
                if (sc.inputStream->index == mReader.videoStream->index) {
                    str += mReader.videoStreamSummary();
                }
                if (info.mediaType == AVMEDIA_TYPE_AUDIO) {
                    hstring hstr = hformat("Track {}: {}", sc.inputStream->index, info.codec);
                    mAudioTrackMap.insert({ comboAudioTrack().Items().Size(), sc.inputStream->index });
                    comboAudioTrack().Items().Append(box_value(hstr));
                }
            }
            //trim string
            size_t pos = str.size() - 1;
            while (pos > 0 && str[pos] == '\n') {
                str[pos] = '\0';
                pos--;
            }

            seekAsync(0.1);
            comboAudioTrack().SelectedIndex(comboAudioTrack().Items().Size() > 1);
            lblStatus().Text(hformat("Version {}", CUVISTA_VERSION));
            texInput().Text(to_hstring(str));
            lblStatusLink().Inlines().Clear();

            spinStackLeft().Maximum(mReader.w * 40.0 / 100.0);
            spinStackRight().Maximum(mReader.w * 40.0 / 100.0);

        } catch (const AVException& ex) {
            imageInput().Source(xamlImageError().Source());
            lblStatus().Text(to_hstring(ex.what()));
            texInput().Text(L"");
            lblStatusLink().Inlines().Clear();
        }
    }


    //-------------------------------------------------------------------------
    //-------------------- Start Stabililzing ---------------------------------
    //-------------------------------------------------------------------------

    fire_and_forget MainWindow::btnStartClick(const IInspectable& sender, const RoutedEventArgs& args) {
        //debugPrint("start");
        if (mInputFile.empty() || mInputReady == false) {
            co_return;
        }

        lblStatus().Text(L"");
        lblStatusLink().Inlines().Clear();

        //with radio buttons the IsChecked() method returns a IReference which ALWAYS will be true -> need to check the Value()
        if (chkEncode().IsChecked().Value() || chkStack().IsChecked().Value()) {
            //get output video file
            mOutputFile = selectFileSave(GetActiveWindow(), chkOverwrite().IsChecked().Value());
            if (mOutputFile.empty()) co_return;
        }

        //get output image sequence folder
        if (chkSequence().IsChecked().Value()) {
            mOutputFile = L"";
            hstring folder = selectFolder(GetActiveWindow());
            if (folder.size() > 0) {
                //check for overwrite
                std::string fileExt = to_string(comboImageType().SelectedValue().as<hstring>());
                std::transform(fileExt.begin(), fileExt.end(), fileExt.begin(), [] (char c) { return std::tolower(c); });
                std::string firstFile = ImageWriter::makeFilename(to_string(folder), 0, fileExt);

                std::filesystem::path p(firstFile);
                if (std::filesystem::exists(p) && chkOverwrite().IsChecked().Value() == false) {
                    //ask for overwrite
                    LPCWSTR msgTitle = L"Confirm File Overwrite";
                    std::wstring firstFileW = std::wstring(firstFile.begin(), firstFile.end());
                    std::wstring msgTextW = L"File " + firstFileW + L" exists,\noverwrite this and subsequent files?";
                    int selection = MessageBoxW(NULL, msgTextW.c_str(), msgTitle, MB_ICONWARNING | MB_YESNO | MB_DEFBUTTON2);
                    if (selection == IDYES) {
                        mOutputFile = folder;
                    }

                } else {
                    //new images
                    mOutputFile = folder;
                }
            }
            if (mOutputFile.empty()) co_return;
        }
        //debugPrint(outputFile);

        //check if input and output point to same file
        std::filesystem::path p1 = mInputFile.c_str();
        std::filesystem::path p2 = mOutputFile.c_str();
        //use error code, otherwise a non existent output file will throw an exception
        std::error_code ec;
        if (std::filesystem::equivalent(p1, p2, ec)) {
            showErrorDialogAsync(L"Invalid File Selection", L"Please select different files\nfor input and output");
            co_return;
        }

        //set parameters
        mData.fileOut = winrt::to_string(mOutputFile);
        mData.deviceRequested = true;
        mData.deviceSelected = comboDevice().SelectedIndex();

        IInspectable option = comboEncoding().SelectedValue();
        mData.outputOption = option.as<CustomRuntimeXaml>()->get<OutputOption>();
        mData.requestedCrf = mData.outputOption.percentToCrf(sliderQuality().Value());
        mData.radsec = spinRadius().Value();
        mData.zoomMin = 1.0 + spinZoomMin().Value();
        mData.zoomMax = chkDynamicZoom().IsChecked().Value() ? 1.0 + spinZoomMax().Value() : mData.zoomMin;
        mData.bgmode = radioBlend().IsChecked().Value() ? BackgroundMode::BLEND : BackgroundMode::COLOR;
        mData.maxFrames = chkFrameLimit().IsChecked().Value() ? (int64_t) spinFrameLimit().Value() : std::numeric_limits<int64_t>::max();
        mData.pyramidLevelsRequested = (int) sliderLevels().Value();

        mData.backgroundColor = Color::rgb(mBackgroundColor.R, mBackgroundColor.G, mBackgroundColor.B);
        mData.backgroundColor.toYUVfloat(&mData.bgcolorYuv.y, &mData.bgcolorYuv.u, &mData.bgcolorYuv.v);

        std::shared_ptr<ProgressDialog> progress;
        std::shared_ptr<MovieFrame> frame;
        std::shared_ptr<FrameExecutor> executor;
        std::shared_ptr<MovieWriter> writer;
        try {
            //rewind reader to beginning of input
            mReader.rewind();
            //check input parameters
            mData.validate(mReader);
            //reset input handler
            mUserInput = UserInputEnum::CONTINUE;
            //crop setting for stack
            mData.stackCrop = { (int) spinStackLeft().Value(), (int) spinStackRight().Value() };
            //audio track to play
            mAudioStreamIndex = -1;
            if (comboAudioTrack().SelectedIndex() > 0) {
                mAudioStreamIndex = mAudioTrackMap.at(comboAudioTrack().SelectedIndex());
            }

            //select writer
            OutputOption imageType = mOutputImageTypeMap.at(comboImageType().SelectedValue().as<hstring>());
            if (chkStack().IsChecked().Value())
                writer = std::make_shared<StackedWriter>(mData, mReader);
            else if (chkSequence().IsChecked().Value() && imageType == OutputOption::IMAGE_BMP)
                writer = std::make_shared<BmpImageWriter>(mData, mReader);
            else if (chkSequence().IsChecked().Value() && imageType == OutputOption::IMAGE_BMP)
                writer = std::make_shared<JpegImageWriter>(mData, mReader);
            else if (chkEncode().IsChecked().Value() && mData.outputOption.group == OutputGroup::VIDEO_NVENC)
                writer = std::make_shared<CudaFFmpegWriter>(mData, mReader);
            else if (chkEncode().IsChecked().Value() && mData.outputOption.group == OutputGroup::VIDEO_FFMPEG)
                writer = std::make_shared<FFmpegWriter>(mData, mReader);
            else if (chkPlayer().IsChecked().Value())
                writer = std::make_shared<PlayerWriter>(*this, *executor, mData, mReader);
            else
                co_return;

            //open writer
            writer->open(mData.outputOption);

            //select frame handler
            mData.mode = comboMode().SelectedIndex();
            if (mData.mode == 0) {
                frame = std::make_shared<MovieFrameCombined>(mData, mReader, *writer);
            } else {
                frame = std::make_shared<MovieFrameConsecutive>(mData, mReader, *writer);
            }

            //select frame executor class
            executor = mData.deviceList[mData.deviceSelected]->create(mData, *frame);
            executor->init();

            //check error logger
            if (errorLogger().hasError()) {
                throw AVException(errorLogger().getErrorMessage());
            }

        } catch (AVException e) {
            hstring msg = to_hstring(e.what());
            showErrorDialogAsync(L"Error", msg);
            lblStatus().Text(msg);
            errorLogger().clear();
            co_return;
        }

        //setup dialog
        if (chkPlayer().IsChecked().Value()) {
            //start gui progress class
            progress = std::make_shared<PlayerProgress>(*this, *executor);

            //player output video image
            mProgressOutput = ImageXamlBGRA::create(imageVideoPlayer(), mReader.h, mReader.w);
            mProgressOutput.loadImageScaledToFit(L"ms-appx:///Assets/signs-02.png");

        } else {
            //start gui progress class
            progress = std::make_shared<ProgressGui>(*this, GetActiveWindow(), *executor);

            //progress dialog
            progressBar().Value(0.0);

            Media::SolidColorBrush brush(mBackgroundColor);
            progressInputGrid().Background(brush);
            progressOutputGrid().Background(brush);

            //progress dialog images
            mProgressInput = ImageXamlBGRA::create(imageProgressInput(), mReader.h, mReader.w);
            mProgressInput.loadImageScaledToFit(L"ms-appx:///Assets/signs-02.png");
            mProgressOutput = ImageXamlBGRA::create(imageProgressOutput(), mReader.h, mReader.w);
            mProgressOutput.loadImageScaledToFit(L"ms-appx:///Assets/signs-02.png");
        }
        lblStatus().Text(L"stabilizing...");

        //start timing
        mData.timeStart();

        //start process loop ----------------------------------------------------
        auto loopFunction = [&] {
            LoopResult result = frame->runLoop(*progress, *this, executor);
            DispatcherQueue().TryEnqueue([&] { progress->dialog().Hide(); });
            return result;
        };
        mFutureLoop = std::async(std::launch::async, loopFunction);

        //show dialog -----------------------------------------------------------
        co_await progress->dialog().ShowAsync();

        //dialog hidden again ----------------------------------------------------
        //debugPrint("loop done");
        LoopResult result = mFutureLoop.get();
        double secs = mData.timeElapsedSeconds();
        double fps = writer->frameIndex / secs;
        std::string fileSize = util::byteSizeToString(writer->encodedBytesTotal);
        writer.reset();
        executor.reset();
        frame.reset();
        progress.reset();

        if (result == LoopResult::LOOP_ERROR) {
            hstring msg = to_hstring(errorLogger().getErrorMessage());
            showErrorDialogAsync(L"Error", msg);
            lblStatus().Text(msg);
            errorLogger().clear();

        } else if (result == LoopResult::LOOP_CANCELLED) {
            lblStatus().Text(L"Operation was cancelled");

        } else if (result == LoopResult::LOOP_SUCCESS) {
            Documents::Run linkRun;
            linkRun.Text(mOutputFile);
            lblStatusLink().Inlines().ReplaceAll({ linkRun });
            lblStatus().Text(hformat("({}) written in {:.1f} min at {:.1f} fps", fileSize, secs / 60.0, fps));
        }
        //debugPrint("done");
    }


    //-------------------------------------------------------------------------
    //-------------------- Event Handlers -------------------------------------
    //-------------------------------------------------------------------------

    fire_and_forget MainWindow::seekAsync(double frac) {
        //debugPrint("seek " + std::to_string(frac));
        if (mInputReady && mReader.seek(frac) && mReader.read(mInputYUV)) {
            inputVideoFraction = frac;
            mInputYUV.toBaseRgb(mInputBGRA);
            DispatcherQueue().TryEnqueue([&, frac] {
                mInputBGRA.invalidate();
                double w = imageBackground().ActualWidth() * frac;
                inputPosition().Width(w);
            });
        }
        co_return;
    }

    void MainWindow::imageGridResize(const IInspectable& sender, const SizeChangedEventArgs& args) {
        inputPosition().Width(imageBackground().ActualWidth() * inputVideoFraction);
    }

    void MainWindow::comboDeviceChanged(const IInspectable& sender, const Controls::SelectionChangedEventArgs& args) {
        comboEncoding().Items().Clear();
        int32_t index = comboDevice().SelectedIndex();
        for (OutputOption& op : mData.deviceList[index]->videoEncodingOptions) {
            IInspectable obj = winrt::make<CustomRuntimeXaml>(op.displayName(), std::make_any<OutputOption>(op));
            comboEncoding().Items().Append(obj);
        }
        comboEncoding().SelectedIndex(0);
    }

    void MainWindow::dragFile(const IInspectable& sender, const DragEventArgs& args) {
        args.AcceptedOperation(winrt::Windows::ApplicationModel::DataTransfer::DataPackageOperation::Link);
    }

    fire_and_forget MainWindow::dropFile(const IInspectable& sender, const DragEventArgs& args) {
        //debugPrint("drop");
        hstring str = winrt::Windows::ApplicationModel::DataTransfer::StandardDataFormats::StorageItems();
        if (args.DataView().Contains(str)) {
            Windows::Foundation::Collections::IVectorView items = co_await args.DataView().GetStorageItemsAsync();
            if (items.Size() > 0) {
                StorageFile storageFile = items.First().Current().try_as<StorageFile>();
                if (storageFile != nullptr) {
                    addInputFile(storageFile.Path());
                }
            }
        }
    }

    //select file
    void MainWindow::btnOpenClick(const IInspectable& sender, const RoutedEventArgs& args) {
        //debugPrint("open");
        hstring file = selectFileOpen(GetActiveWindow());
        if (file.size() > 0) {
            addInputFile(file);
        }
    }

    //commit color selection
    void MainWindow::btnColorOk(const IInspectable& sender, const RoutedEventArgs& args) {
        //debugPrint("ok");
        setBackgroundColor(colorPicker().Color());
        colorFlyout().Hide();
        radioColor().IsChecked(true);
    }

    //cancel color selection
    void MainWindow::btnColorCancel(const IInspectable& sender, const RoutedEventArgs& args) {
        //debugPrint("cancel");
        colorFlyout().Hide();
    }

    //select color
    void MainWindow::lblColorClick(const IInspectable& sender, const Input::TappedRoutedEventArgs& args) {
        //debugPrint("color picker");
        colorPicker().Color(mBackgroundColor);
        Controls::Primitives::FlyoutBase::ShowAttachedFlyout(sender.as<FrameworkElement>());
    }

    //seek to file position
    fire_and_forget MainWindow::imageInputClick(const IInspectable& sender, const Input::TappedRoutedEventArgs& args) {
        winrt::Windows::Foundation::Point p = args.GetPosition(imageBackground());
        double fraction = 1.0 * p.X / imageBackground().ActualWidth();
        co_await winrt::resume_background();
        seekAsync(fraction);
    }

    //cancel stabilization
    void MainWindow::btnStopClick(const IInspectable& sender, const RoutedEventArgs& args) {
        //set the cancel signal, send to frame loop via callback
        mUserInput = UserInputEnum::QUIT;
    }

    //open video file in registered windows app
    fire_and_forget MainWindow::statusLinkClick(const IInspectable& sender, const RoutedEventArgs& args) {
        Windows::Storage::StorageFile file = co_await Windows::Storage::StorageFile::GetFileFromPathAsync(mOutputFile);
        if (file) {
            bool result = co_await Windows::System::Launcher::LaunchFileAsync(file);
        }
    }


    //---------------------------------------------------------------
    //-------------------- Player Events ----------------------------
    //---------------------------------------------------------------

    void MainWindow::btnPlayerPauseClick(const IInspectable& sender, const RoutedEventArgs& args) {
        bool p = btnPlayerPause().IsChecked().Value();
        lblPlayerStatus().Text(p ? L"Pausing..." : L"Playing...");
        mPlayerPaused = p;
    }

    void MainWindow::btnPlayerStopClick(const IInspectable& sender, const RoutedEventArgs& args) {
        mUserInput = UserInputEnum::QUIT;
        mPlayerPaused = false;
        btnPlayerPause().IsChecked(false);
    }

    void MainWindow::sliderVolumeChanged(const IInspectable& sender, const Controls::Primitives::RangeBaseValueChangedEventArgs& args) {
        double volume = args.NewValue();
        mAudioGain = volume / 100.0;
    }

    //---------------------------------------------------------------
    //-------------------- Info Box  --------------------------------
    //---------------------------------------------------------------

    fire_and_forget MainWindow::btnInfoClick(const IInspectable& sender, const RoutedEventArgs& args) {
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
        infoLinkEmail().Inlines().ReplaceAll({ emailRun });

        //github link
        hstring strGit = L"https://github.com/RainerMtb/cuvista";
        Windows::Foundation::Uri gitUri(strGit);
        infoLinkGit().NavigateUri(gitUri);
        Documents::Run gitRun;
        gitRun.Text(strGit);
        infoLinkGit().Inlines().ReplaceAll({ gitRun });

        //header and footer
        infoRunHeader().Text(std::format(L"CUVISTA - Cuda Video Stabilizer, Version {}\n\u00A9 2025 Rainer Bitschi ",
            to_hstring(CUVISTA_VERSION)
        ));
        using namespace Microsoft::Windows::ApplicationModel::WindowsAppRuntime;
        infoRunFooter().Text(std::format(L"\nWindows App SDK {}, Windows App Runtime {}\nLicense GNU GPLv3+: GNU GPL version 3 or later",
            ReleaseInfo::AsString(),
            RuntimeInfo::AsString()
        ));

        //button will be disabled when starting tests
        btnInfoTest().IsEnabled(true);
        infoDialog().XamlRoot(rootPanel().XamlRoot());
        Controls::ContentDialogResult result = co_await infoDialog().ShowAsync();
        //debugPrint("done");
    }

    void MainWindow::btnInfoTestClick(const IInspectable& sender, const RoutedEventArgs& args) {
        btnInfoTest().IsEnabled(false);
        mFutureInfo = std::async(std::launch::async, runSelfTest, std::ref(*this), mData.deviceList);
    }

    void MainWindow::btnInfoCloseClick(const IInspectable& sender, const RoutedEventArgs& args) {
        infoDialog().Hide();
    }

    void MainWindow::infoDialogClosing(const Controls::ContentDialog& sender, const Controls::ContentDialogClosingEventArgs& args) {
        //debugPrint("closing");
        mFutureInfo.wait();
    }


    //-------------------------------------------------------------------------
    //-------------------- Private Methods ------------------------------------
    //-------------------------------------------------------------------------

    fire_and_forget MainWindow::showErrorDialogAsync(hstring title, hstring content) {
        Controls::ContentDialog dialog;
        dialog.Title(box_value(title));
        dialog.Content(box_value(content));
        dialog.CloseButtonText(L"OK");
        dialog.XamlRoot(rootPanel().XamlRoot());
        co_await dialog.ShowAsync();
        co_return;
    }

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

    void MainWindow::infoBoxAppendText(std::string str) {
        mInfoBoxString = mInfoBoxString + to_hstring(str);
        DispatcherQueue().TryEnqueue([&, str = mInfoBoxString] {
            infoBox().Text(str);
            infoScroller().ScrollToVerticalOffset(infoScroller().ScrollableHeight());
        });
    }

    void MainWindow::btnResetClick(const IInspectable& sender, const RoutedEventArgs& args) {
        setBackgroundColor(Windows::UI::ColorHelper::FromArgb(255, defaultParam.bgColorRed, defaultParam.bgColorGreen, defaultParam.bgColorBlue));

        chkOverwrite().IsChecked(false);
        spinRadius().Value(defaultParam.radsec);
        spinZoomMin().Value(defaultParam.zoomMin - 1.0);
        spinZoomMax().Value(defaultParam.zoomMax - 1.0);
        chkDynamicZoom().IsChecked(true);
        chkFrameLimit().IsChecked(false);
        spinFrameLimit().Value(defaultParam.frameLimit);

        sliderLevels().Value(defaultParam.levels);
        sliderQuality().Value(defaultParam.encodingQuality);

        mReader.close();
        mInputFile = {};

        texInput().Text(L"");
        comboInputFile().Items().Clear();
        inputPosition().Width(1.0);
        mInputBGRA = ImageXamlBGRA::create(imageInput(), 100, 100);
    }

    void MainWindow::modeSelectionChanged(const IInspectable& sender, const Controls::SelectionChangedEventArgs& args) {
        if (comboMode().SelectedIndex() == 0) {
            chkPlayer().IsEnabled(true);

        } else {
            chkPlayer().IsEnabled(false);
            chkPlayer().IsChecked(false);
        }
    }


    //-------------------------------------------------------------------------
    //-------------------- subclass methods -----------------------------------
    //-------------------------------------------------------------------------

    //called on background thread
    UserInputEnum MainWindow::checkState() {
        UserInputEnum e = mUserInput;
        mUserInput = UserInputEnum::NONE;
        return e;
    }

    //called on background thread
    void MainWindow::print(const std::string& str) {
        infoBoxAppendText(str);
    }

    //called on background thread
    void MainWindow::printNewLine() {
        infoBoxAppendText("\n");
    }


    //-------------------------------------------------------------------------
    //-------------------- Destructor -----------------------------------------
    //-------------------------------------------------------------------------

   void MainWindow::windowClosedEvent(const IInspectable& sender, const WindowEventArgs& args) {
        //debugPrint("closed");
        //args.Handled(true); //cancel the event
       if (mFutureLoop.valid()) {
           mUserInput = UserInputEnum::QUIT;
           mFutureLoop.wait();
       }

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
        localValues.Insert(L"encodingQuality", box_value(sliderQuality().Value()));

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
