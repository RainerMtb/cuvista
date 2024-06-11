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

#include "progress.h"
#include "MovieFrame.hpp"

ProgressWindow::ProgressWindow(QWidget* parent) :
	QMainWindow(parent)
{
	ui.setupUi(this);
	resize(minimumSize());
	setWindowModality(Qt::ApplicationModal);
	connect(ui.btnStop, &QPushButton::clicked, this, &ProgressWindow::close); //connect button to close signal
}

void ProgressWindow::changeEvent(QEvent* event) {
	if (event->type() == QEvent::WindowStateChange && isMinimized()) {
		parentWidget()->showMinimized();
	}
}

void ProgressWindow::showEvent(QShowEvent* event) {
	ui.btnStop->setEnabled(true);
	ui.progressBar->reset();
}

void ProgressWindow::closeEvent(QCloseEvent* event) {
	ui.btnStop->setEnabled(false);
	event->ignore(); //hide only after output is terminated in main window
	cancel();
}

void ProgressWindow::progress(bool isFinite, double value) {
	if (isFinite) {
		ui.progressBar->setMaximum(1000);
		ui.progressBar->setValue((int) (value * 10.0));

	} else {
		ui.progressBar->setMaximum(0);
	}
}

void ProgressWindow::updateInput(QImage im, QString time) {
	ui.imageInput->setImage(im);
	ui.lblTimeInput->setText(time);
}

void ProgressWindow::updateOutput(QImage im, QString time) {
	ui.imageOutput->setImage(im);
	ui.lblTimeOutput->setText(time);
}

void ProgressGui::update(bool force) {
	if (isDue(force)) {
		progressWindow->sigProgress(isFinite(), progressPercent());
	}

	auto timePointNow = std::chrono::steady_clock::now();
	std::chrono::nanoseconds delta = timePointNow - timePoint;
	bool imageDue = delta.count() / 1'000'000 > 250;

	if (imageDue && frame.mReader.frameIndex > 0 && progressWindow->isVisible()) {
		timePoint = timePointNow;
		uint64_t idx = frame.mReader.frameIndex - 1;
		frame.getInput(idx, input);
		QImage im(input.data(), input.w, input.h, input.stride, QImage::Format_RGBX8888);
		progressWindow->sigUpdateInput(im, QString::fromStdString(frame.getTimeForFrame(idx)));
	}
	if (imageDue && frame.mWriter.frameIndex > 0 && progressWindow->isVisible()) {
		timePoint = timePointNow;
		uint64_t idx = frame.mWriter.frameIndex - 1;
		frame.getWarped(idx, output);
		QImage im(output.data(), output.w, output.h, output.stride, QImage::Format_RGBX8888);
		progressWindow->sigUpdateOutput(im, QString::fromStdString(frame.getTimeForFrame(idx)));
	}
}