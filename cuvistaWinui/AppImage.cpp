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
#include "AppImage.hpp"

#include <winrt/Windows.Graphics.Imaging.h>
#include <winrt/Windows.Storage.Streams.h>
#include <winrt/Microsoft.UI.Xaml.Media.Imaging.h>


ImageXamlBGRA::ImageXamlBGRA() :
    bitmap(1, 1)
{}

ImageXamlBGRA::ImageXamlBGRA(winrt::Microsoft::UI::Xaml::Media::Imaging::WriteableBitmap bitmap, int h, int w, unsigned char* data) :
    ImageBGRA(h, w, w * 4, data),
    bitmap { bitmap }
{}

//create WriteableBitmap, create ImageBGRA from that bitmap, set Xaml Image Source to that bitmap
ImageXamlBGRA ImageXamlBGRA::create(winrt::Microsoft::UI::Xaml::Controls::IImage xamlImage, int h, int w) {
    winrt::Microsoft::UI::Xaml::Media::Imaging::WriteableBitmap bitmap(w, h);
    xamlImage.Source(bitmap);
    return ImageXamlBGRA(bitmap, h, w, bitmap.PixelBuffer().data());
}


using namespace winrt::Windows::Graphics::Imaging;
using namespace winrt::Windows::Storage::Streams;

winrt::fire_and_forget ImageXamlBGRA::loadImageScaledToFit(winrt::hstring file) {
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
    winrt::com_array<uint8_t> pixelData = pixelProvider.DetachPixelData();
    unsigned char* src = pixelData.data();

    //first set all destination pixels transparent
    std::fill(data(), data() + sizeInBytes(), 0);

    //copy pixel data to WriteableBitmap
    unsigned char* dest = data() + (h - s) * w * 2 + (w - s) * 2;
    for (int i = 0; i < s; i++) {
        std::copy_n(src, 4 * s, dest);
        dest += 4 * w;
        src += 4 * s;
    }

    invalidate();
    co_return;
}

void ImageXamlBGRA::invalidate() {
    bitmap.Invalidate();
}

