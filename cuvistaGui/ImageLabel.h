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

#include <QLabel>
#include <QMouseEvent>
#include "Image2.hpp"

class ImageLabel : public QLabel {
	Q_OBJECT

signals:
	void mouseClicked(double pos);

private:
	QPixmap pm;

public:
	ImageLabel(QWidget* parent) : QLabel(parent) {
		pm = QPixmap(100, 100);
		pm.fill(Qt::transparent);
	}

	void resizePixmap() {
		QLabel::setPixmap(pm.scaled(width(), height(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	}

	void setImage(const ImagePPM& ppm) {
		pm.loadFromData(ppm.data(), ppm.size(), "PPM");
		resizePixmap();
	}

	void setImage(const QPixmap& pm) {
		this->pm = pm;
		resizePixmap();
	}

	void setImage(QImage im) {
		setImage(QPixmap::fromImage(im));
	}

	void setImage(const ImageYuv& im) {
		setImage(QImage(im.toBGR().data(), im.w, im.h, im.w * 3ull, QImage::Format_BGR888));
	}

	void resizeEvent(QResizeEvent* event) override {
		resizePixmap();
	}

	void mousePressEvent(QMouseEvent* event) override {
		double frac = 1.0 * event->localPos().x() / width();
		mouseClicked(frac);
	}
};