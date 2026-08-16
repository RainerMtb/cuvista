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

#include <QApplication>
#include <QDebug>
#include <QLayout>
#include <QProgressBar>
#include "util.hpp"
#include "ofxGui.hpp"


#if defined(_WIN64)
#define LIBRARY_EXPORT extern "C" __declspec(dllexport)
#else
#define LIBRARY_EXPORT extern "C"
#endif

using namespace ofx;

LIBRARY_EXPORT void loadGui(OfxGuiContext& guiContext) {
	guiContext.gui = std::make_shared<OfxGuiQt>(guiContext);
	guiContext.debugLogger->format("gui loaded");
}

GuiApplication::GuiApplication(int argc, char** argv) :
	QApplication(argc, argv)
{
	window = new QWidget(nullptr, Qt::Dialog | Qt::WindowStaysOnTopHint);
	window->setMinimumSize(200, 150);
	window->setWindowTitle("Stabilizing...");

	QVBoxLayout* layout = new QVBoxLayout(window);
	QProgressBar* progress = new QProgressBar(window);
	progress->setMinimum(0);
	progress->setMaximum(1000);
	layout->addWidget(progress);
	window->setLayout(layout);

	connect(this, &GuiApplication::sigClose, window, &QWidget::close, Qt::QueuedConnection);
	connect(this, &GuiApplication::sigClose, this, &QApplication::quit, Qt::QueuedConnection);
	connect(this, &GuiApplication::sigShow, window, &QWidget::show);
	connect(this, &GuiApplication::sigHide, window, &QWidget::hide);
	connect(this, &GuiApplication::sigUpdate, progress, &QProgressBar::setValue);
}

void GuiApplication::shutdown() { sigClose(); }
void GuiApplication::showProgress() { sigShow(); }
void GuiApplication::updateProgress(int value) { sigUpdate(value); }
void GuiApplication::hideProgress() { sigHide(); }


//-------------------------------------------------------------------------------------

OfxGuiQt::OfxGuiQt(OfxGuiContext& guiContext) :
	guiContext { guiContext }
{}

void OfxGuiQt::init() {
	guiContext.debugLogger->format("gui init");
	
	bool isStarted = false;
	auto func = [&] {
		guiContext.debugLogger->format("gui thread started");
		int argc = 1;
		char ch = '\0';
		char* argv = &ch;
		QApplication::setStyle("Fusion");
		QString qs = QString::fromStdString(guiContext.pluginPath.string());
		QApplication::setLibraryPaths(QStringList(qs));
		app = std::make_shared<GuiApplication>(argc, &argv);
		QIcon icon(":/res/cuvista.png");
		app->setWindowIcon(icon);
		app->setQuitOnLastWindowClosed(false);
		isStarted = true;
		guiContext.debugLogger->format("gui thread starting loop");
		app->exec();
		guiContext.debugLogger->format("gui thread ending loop");
		app.reset();
	};
	guiThread = std::thread(func);
	while (isStarted == false) {}
	guiContext.debugLogger->log("gui running");
}

void OfxGuiQt::shutdown() {
	guiContext.debugLogger->format("gui shutdown, thread {}", threadId());
	if (app) {
		app->shutdown();
		guiThread.join();
	}
	guiContext.debugLogger->format("gui shutdown complete, thread {}", threadId());
}

OfxGuiQt::~OfxGuiQt() {
	if (app) {
		app->shutdown();
		guiThread.join();
	}
	guiContext.debugLogger->format("gui destruct, thread {}", threadId());
}

void OfxGuiQt::showProgress() {
	app->showProgress();
}

void OfxGuiQt::updateProgress(double progress) {
	app->updateProgress((int) (progress * 1000.0));
}

void OfxGuiQt::hideProgress() {
	app->hideProgress();
}
