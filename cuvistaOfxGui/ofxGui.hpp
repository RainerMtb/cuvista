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

#include <QApplication>
#include <QWidget>
#include <QPlainTextEdit>
#include <QImage>
#include <QPixmap>
#include <QLabel>

#include <thread>
#include <filesystem>
#include "ofxGuiInterface.hpp"

namespace ofx {

	using namespace im;

	class ImageLabel : public QLabel {
		Q_OBJECT

	private:
		QPixmap pixmap;

		void resizePixmap();

	public:
		ImageLabel(QWidget* parent);
		
		void resizeEvent(QResizeEvent* event) override;

	public slots:
		void setImage(QImage image);
	};


	class GuiWindow : public QWidget {
		Q_OBJECT

	public:
		bool isDone = false;
		bool cancelRequest = false;

		GuiWindow(QWidget* parent, Qt::WindowFlags f);
		~GuiWindow();

		void closeEvent(QCloseEvent* event) override;
	};


	class OfxGuiQt : public QObject, public OfxGui {
		Q_OBJECT

	signals:
		void sigClose();
		void sigUpdateProgress(int value);
		void sigUpdateImage(QImage image);
		void sigUpdateInfo(const std::string& infoString);

	private slots:
		void cancel();

	private:
		int argc = 1;
		char ch = '\0';
		char* argv = &ch;
		GuiWindow* window = nullptr;
		QImage inputImage;

		OfxGuiContext& guiContext;
		QApplication* app = nullptr;

	public:
		OfxGuiQt(OfxGuiContext& guiContext);
		~OfxGuiQt();

		bool isCancelled() override;

		void init() override;
		bool checkNewWindow() override;
		void shutdown() override;

		void openInfo(const std::string& infoString, const std::string& hostName, const std::string& hostVersion) override;
		void updateInfo(const std::string& infoString) override;

		void openProgress() override;
		void updateProgress(double progress, const Image8& image) override;
		void close() override;
	};
}
