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

#include "ImageLabel.h"

ImageLabel::ImageLabel(QWidget* parent) : QLabel(parent) {
    pm = QPixmap(100, 100);
    pm.fill(Qt::transparent);
}

void ImageLabel::resizePixmap() {
    QLabel::setPixmap(pm.scaled(width(), height(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void ImageLabel::setImage(const ImagePPM& ppm) {
    pm.loadFromData(ppm.data(), ppm.size(), "PPM");
    resizePixmap();
}

void ImageLabel::setImage(const QPixmap& pm) {
    this->pm = pm;
    resizePixmap();
}

void ImageLabel::setImage(QImage im) {
    setImage(QPixmap::fromImage(im));
}

void ImageLabel::setImage(const ImageYuv& im) {
    setImage(QImage(im.toBGR().data(), im.w, im.h, im.w * 3ull, QImage::Format_BGR888));
}

void ImageLabel::resizeEvent(QResizeEvent* event) {
    resizePixmap();
}

void ImageLabel::mousePressEvent(QMouseEvent* event) {
    double frac = 1.0 * event->position().x() / width();
    mouseClicked(frac);
}