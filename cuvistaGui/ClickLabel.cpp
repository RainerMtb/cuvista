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

#include "ClickLabel.h"
#include <QKeyEvent>

ClickLabel::ClickLabel(QWidget* parent) :
    QLabel(parent) 
{}

//forward mouse click to click signal
void ClickLabel::mousePressEvent(QMouseEvent* event) {
    clicked();
}

//forward keys to click signal
void ClickLabel::keyPressEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key::Key_Space || event->key() == Qt::Key::Key_Enter || event->key() == Qt::Key::Key_Return) {
        clicked();
    }
}