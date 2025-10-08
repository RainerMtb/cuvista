
/*
* COMMDLG Variant
* 
OPENFILENAME fileStruct = {};
fileStruct.lStructSize = sizeof(fileStruct);
fileStruct.hwndOwner = GetActiveWindow();
wchar_t szFile[260] = {};
fileStruct.lpstrFile = szFile;
fileStruct.lpstrFile[0] = '\0';
fileStruct.nMaxFile = sizeof(szFile);
bool retval = GetSaveFileName(&fileStruct);
if (retval) {
    LPWSTR file = fileStruct.lpstrFile;
    debugPrint(file);
}
*/

/*
* WinUi Picker Variant
* 
Pickers::FileSavePicker savePicker;
savePicker.as<IInitializeWithWindow>()->Initialize(GetActiveWindow());
savePicker.SuggestedStartLocation(Pickers::PickerLocationId::VideosLibrary);
std::vector<hstring> videos = { L".mp4", L".mkv" };
savePicker.FileTypeChoices().Insert(L"Video Files", winrt::single_threaded_vector<hstring>(std::move(videos)));

Windows::Storage::StorageFile file = co_await savePicker.PickSaveFileAsync();
if (file != nullptr) {
    outFile = file.Path();
} else {
    co_return;
}
*/
