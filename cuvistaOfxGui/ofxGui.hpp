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
#include <thread>
#include <filesystem>
#include "ofxGuiInterface.hpp"

namespace ofx {

	class GuiApplication : public QApplication {
		Q_OBJECT

	signals:
		void sigClose();
		void sigShow();
		void sigHide();
		void sigUpdate(int value);

	public:
		QWidget* window;

		GuiApplication(int argc, char** argv);

		void shutdown();
		void showProgress();
		void updateProgress(int value);
		void hideProgress();
	};


	class OfxGuiQt : public OfxGui {

	private:
		OfxGuiContext& guiContext;
		std::shared_ptr<GuiApplication> app = {};
		std::thread guiThread;

	public:
		OfxGuiQt(OfxGuiContext& guiContext);
		~OfxGuiQt();

		void init() override;
		void shutdown() override;
		void showProgress() override;
		void updateProgress(double progress) override;
		void hideProgress() override;
	};
}
