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

#include "ofxMain.hpp"
#include "CoreData.hpp"

using namespace ofx;

std::string MainContext::getString(OfxPropertySetHandle handle, const char* id, int index) {
	char* cstr;
	propertySuite->propGetString(handle, id, index, &cstr);
	return cstr;
}

double MainContext::getDouble(OfxPropertySetHandle handle, const char* id, int index) {
	double d;
	propertySuite->propGetDouble(handle, id, index, &d);
	return d;
}

int MainContext::getInt(OfxPropertySetHandle handle, const char* id, int index) {
	int i;
	propertySuite->propGetInt(handle, id, index, &i);
	return i;
}


#if defined(_WIN64)

#include <Windows.h>

HMODULE guiLib;
typedef void (*LoadFunc)(OfxGuiContext& guiContext);

bool MainContext::guiLoadLibrary(OfxGuiContext& guiContext) {
	std::filesystem::current_path(guiContext.pluginPath);

	guiLib = LoadLibraryA("cuvistaOfxGui.dll");
	if (!guiLib) return false;
	FARPROC f = GetProcAddress(guiLib, "loadGui");
	if (!f) return false;
	LoadFunc loadFunc = (LoadFunc) f;
	loadFunc(guiContext);
	return true;
}

void MainContext::guiFreeLibrary(OfxGuiContext& guiContext) {
	guiContext.gui.reset();
	FreeLibrary(guiLib);
	debugLogger().log("gui unloaded");
}

#endif


//set up effect parameters
void MainContext::setupParameters(OfxParamSetHandle paramSet) {
	OfxPropertySetHandle paramProps;

	main.parameterSuite->paramDefine(paramSet, kOfxParamTypePushButton, "stabilize", &paramProps);
	main.propertySuite->propSetString(paramProps, kOfxPropLabel, 0, "   Stabilize   ");

	main.parameterSuite->paramDefine(paramSet, kOfxParamTypePushButton, "info", &paramProps);
	main.propertySuite->propSetString(paramProps, kOfxPropLabel, 0, "   About   ");

	main.parameterSuite->paramDefine(paramSet, kOfxParamTypeDouble, "radius", &paramProps);
	main.propertySuite->propSetString(paramProps, kOfxPropLabel, 0, "Temporal Radius");
	main.propertySuite->propSetDouble(paramProps, kOfxParamPropDefault, 0, defaultParam.radsec);
	main.propertySuite->propSetDouble(paramProps, kOfxParamPropMin, 0, defaultParam.radsecMin);
	main.propertySuite->propSetDouble(paramProps, kOfxParamPropMax, 0, defaultParam.radsecMax);
	main.propertySuite->propSetInt(paramProps, kOfxParamPropAnimates, 0, false);

	main.parameterSuite->paramDefine(paramSet, kOfxParamTypeDouble, "zoom", &paramProps);
	main.propertySuite->propSetString(paramProps, kOfxPropLabel, 0, "Image Zoom");
	main.propertySuite->propSetString(paramProps, kOfxParamPropDoubleType, 0, kOfxParamDoubleTypeScale);
	main.propertySuite->propSetDouble(paramProps, kOfxParamPropDefault, 0, defaultParam.zoomMin);
	main.propertySuite->propSetDouble(paramProps, kOfxParamPropMin, 0, defaultParam.zoomMinRange);
	main.propertySuite->propSetDouble(paramProps, kOfxParamPropMax, 0, defaultParam.zoomMaxRange);

	//main.parameterSuite->paramDefine(paramSet, kOfxParamTypeDouble, "zoomMin", &paramProps);
	//main.propertySuite->propSetString(paramProps, kOfxPropLabel, 0, "Image Zoom Min");
	//main.propertySuite->propSetString(paramProps, kOfxParamPropDoubleType, 0, kOfxParamDoubleTypeScale);
	//main.propertySuite->propSetDouble(paramProps, kOfxParamPropDefault, 0, defaultParam.zoomMin);
	//main.propertySuite->propSetDouble(paramProps, kOfxParamPropMin, 0, defaultParam.zoomMinRange);
	//main.propertySuite->propSetInt(paramProps, kOfxParamPropAnimates, 0, false);

	//main.parameterSuite->paramDefine(paramSet, kOfxParamTypeBoolean, "zoomDynamic", &paramProps);
	//main.propertySuite->propSetString(paramProps, kOfxPropLabel, 0, "Dynamic Zoom up to");
	//main.propertySuite->propSetInt(paramProps, kOfxParamPropAnimates, 0, false);
	//main.propertySuite->propSetInt(paramProps, kOfxParamPropDefault, 0, true);

	//main.parameterSuite->paramDefine(paramSet, kOfxParamTypeDouble, "zoomMax", &paramProps);
	//main.propertySuite->propSetString(paramProps, kOfxPropLabel, 0, "Image Zoom Max");
	//main.propertySuite->propSetString(paramProps, kOfxParamPropDoubleType, 0, kOfxParamDoubleTypeScale);
	//main.propertySuite->propSetDouble(paramProps, kOfxParamPropDefault, 0, defaultParam.zoomMax);
	//main.propertySuite->propSetDouble(paramProps, kOfxParamPropMaxn, 0, defaultParam.zoomMaxRange);
	//main.propertySuite->propSetInt(paramProps, kOfxParamPropAnimates, 0, false);
}
