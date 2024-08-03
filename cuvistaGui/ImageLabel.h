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
	ImageLabel(QWidget* parent = nullptr);

	void resizePixmap();

	void setImage(const ImagePPM& ppm);

	void setImage(const QPixmap& pm);

	void setImage(QImage im);

	void setImage(const ImageYuv& im);

	void resizeEvent(QResizeEvent* event) override;

	void mousePressEvent(QMouseEvent* event) override;
};