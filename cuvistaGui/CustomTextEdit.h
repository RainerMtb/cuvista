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

#include "Util.hpp"
#include <QPlainTextEdit>

class MessagePrinterGui : public QObject, public util::MessagePrinter {
    Q_OBJECT

signals:
    void appendText(const QString& str);

public:
    void print(const std::string& str) override;

    void printNewLine() override;
};


class ScrollingTextEdit : public QPlainTextEdit {
    Q_OBJECT

public slots:
    void appendText(const QString& str);

public:
    ScrollingTextEdit(QWidget* parent = nullptr);
};


class DropTextEdit : public QPlainTextEdit {
    Q_OBJECT

public:
    DropTextEdit(QWidget* parent = nullptr);

    void dragEnterEvent(QDragEnterEvent* event) override;
    void dragMoveEvent(QDragMoveEvent* event) override;
    void dropEvent(QDropEvent* event) override;

signals:
    void fileDropped(QString inputPath);
};