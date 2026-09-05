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
#include <QPushbutton>
#include <QCloseEvent>

#include "version.hpp"
#include "util.hpp"
#include "ofxGui.hpp"
#include "ImageClasses.hpp"


#if defined(_WIN64)
#define LIBRARY_EXPORT extern "C" __declspec(dllexport)
#else
#define LIBRARY_EXPORT extern "C"
#endif

template <class... Args> QString qformat(std::format_string<Args...> fmt, Args&&... args) {
	return QString::fromStdString(std::format(fmt, std::forward<Args>(args)...));
}

using namespace ofx;

LIBRARY_EXPORT void loadGui(OfxGuiContext& guiContext) {
	guiContext.gui = std::make_shared<OfxGuiQt>(guiContext);
	guiContext.debugLogger->format("gui loaded");
}


//----------------------------------------------------------------------------------------

ImageLabel::ImageLabel(QWidget* parent) :
	QLabel(parent)
{}

void ImageLabel::resizePixmap() {
	setPixmap(pixmap.scaled(width(), height(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void ImageLabel::setImage(QImage image) {
	this->pixmap = QPixmap::fromImage(image);
	resizePixmap();
}

void ImageLabel::resizeEvent(QResizeEvent* event) {
	resizePixmap();
}


//----------------------------------------------------------------------------------------

GuiWindow::GuiWindow(QWidget* parent, Qt::WindowFlags f) :
	QWidget(parent, f)
{}

GuiWindow::~GuiWindow() {}

void GuiWindow::closeEvent(QCloseEvent* event) {
	// clicking close will set cancelRequest
	// allow closing only when really done
	if (isDone) {
		event->accept();

	} else {
		cancelRequest = true;
		event->ignore();
	}
}


//-------------------------------------------------------------------------------------

OfxGuiQt::OfxGuiQt(OfxGuiContext& guiContext) :
	guiContext { guiContext }
{}

//must be called on the application thread
void OfxGuiQt::init() {
	guiContext.debugLogger->format("gui init");

	QApplication::setStyle("Fusion");
	QString qs = QString::fromStdString(guiContext.pluginPath.string());
	QApplication::setLibraryPaths(QStringList(qs));
	app = new QApplication(argc, &argv);
	QIcon icon(":/res/cuvista.png");
	app->setWindowIcon(icon);
}

bool OfxGuiQt::checkNewWindow() {
	if (window) {
		window->activateWindow();
		return false;

	} else {
		return true;
	}
}

void OfxGuiQt::cancel() {
	window->cancelRequest = true;
}

//must be called on the application thread
void OfxGuiQt::openProgress() {
	window = new GuiWindow(nullptr, Qt::Dialog | Qt::WindowStaysOnTopHint);
	window->setMinimumSize(300, 200);
	window->setWindowTitle("Stabilizing...");

	QVBoxLayout* layout = new QVBoxLayout(window);
	ImageLabel* imageLabel = new ImageLabel(window);
	imageLabel->setMinimumSize(160, 80);
	imageLabel->setAlignment(Qt::AlignCenter);
	layout->addWidget(imageLabel);
	QProgressBar* progress = new QProgressBar(window);
	progress->setMinimum(0);
	progress->setMaximum(1000);
	layout->addWidget(progress);
	QPushButton* btnCancel = new QPushButton("Cancel", window);
	layout->addWidget(btnCancel);
	window->setLayout(layout);

	QObject::connect(this, &OfxGuiQt::sigClose, window, &QWidget::close, Qt::QueuedConnection);
	QObject::connect(this, &OfxGuiQt::sigUpdateProgress, progress, &QProgressBar::setValue, Qt::QueuedConnection);
	QObject::connect(this, &OfxGuiQt::sigUpdateImage, imageLabel, &ImageLabel::setImage);
	QObject::connect(btnCancel, &QPushButton::clicked, this, &OfxGuiQt::cancel);

	//guiContext.debugLogger->format("gui open progress");
	window->show();
	//guiContext.debugLogger->format("gui loop starting");
	app->exec();
	//guiContext.debugLogger->format("gui loop ending");
	delete window;
	window = nullptr;
}

void OfxGuiQt::updateProgress(double progress, const Image8& image) {
	sigUpdateProgress((int) (progress * 1000.0));
	inputImage = QImage(image.data(), image.w(), image.h(), image.strideInBytes(), QImage::Format_RGBX8888);
	sigUpdateImage(inputImage);
}

//must be called on the application thread
void OfxGuiQt::openInfo(const std::string& infoString, const std::string& hostName, const std::string& hostVersion) {
	int boxHeight = 250;
	int boxWidth = 450;

	window = new GuiWindow(nullptr, Qt::Dialog | Qt::WindowStaysOnTopHint);
	window->setWindowTitle("Cuvista Info");
	std::string strEmail = "cuvista@a1.net";
	std::string strGitHub = "https://github.com/RainerMtb/cuvista";
	QString headerText = qformat(
		"CUVISTA - Cuda Video Stabilizer, Version {}<br>"
		"Cuvista OpenFX Plugin, Host: {}, Api Version: {}<br>"
		"Copyright (c) 2026 Rainer Bitschi <a href='mailto:{}'>{}</a> <a href='{}'>{}</a><br>"
		"License GNU GPLv3+: GNU GPL version 3 or later<br>"
		"Gui compiled with Qt version {}, running on version {}",
		CUVISTA_VERSION, hostName, hostVersion, strEmail, strEmail, strGitHub, strGitHub, QT_VERSION_STR, qVersion());

	QLabel* header = new QLabel(window);
	header->setText(headerText);
	header->setTextFormat(Qt::RichText);

	QPlainTextEdit* textBox = new QPlainTextEdit(window);
	QString qstr = QString::fromStdString(infoString);
	textBox->setPlainText(qstr);
	textBox->setMinimumHeight(boxHeight);
	textBox->setReadOnly(true);
	textBox->setFont(QFont("Consolas"));
	textBox->setLineWrapMode(QPlainTextEdit::LineWrapMode::NoWrap);

	QPushButton* btnClose = new QPushButton("Close", window);
	btnClose->setFixedWidth(70);
	btnClose->setFixedHeight(28);

	//create widgets
	QVBoxLayout* layout = new QVBoxLayout(window);
	layout->addWidget(header);
	layout->addWidget(textBox);
	QHBoxLayout* box = new QHBoxLayout(window);
	box->addStretch();
	box->addWidget(btnClose);
	layout->addLayout(box);

	//calculate text width
	QFontMetrics fm = textBox->fontMetrics();
	for (auto& s : qstr.split('\n')) {
		int w = fm.horizontalAdvance(s);
		if (w > boxWidth) boxWidth = w;
	}
	textBox->setMinimumWidth(boxWidth + 40);

	window->isDone = true;
	connect(btnClose, &QPushButton::clicked, window, &QWidget::close);

	auto updateFcn = [&] (const std::string& infoString) {
		QString qstr = textBox->toPlainText() + QString::fromStdString(infoString);
		textBox->setPlainText(qstr);
		textBox->moveCursor(QTextCursor::End);
	};
	connect(this, &OfxGuiQt::sigUpdateInfo, this, updateFcn, Qt::QueuedConnection);

	window->show();
	//guiContext.debugLogger->format("gui loop starting");
	app->exec();
	//guiContext.debugLogger->format("gui loop ending");
	delete window;
	window = nullptr;
}

void OfxGuiQt::updateInfo(const std::string& infoString) {
	sigUpdateInfo(infoString);
}

void OfxGuiQt::close() {
	window->isDone = true;
	sigClose();
}

//must be called on the application thread
void OfxGuiQt::shutdown() {
	guiContext.debugLogger->format("gui shutdown");
	delete app;
}

bool OfxGuiQt::isCancelled() {
	return window->cancelRequest;
}

OfxGuiQt::~OfxGuiQt() {
	guiContext.debugLogger->format("gui destruct on thread {}", threadId());
}
