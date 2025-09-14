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

#include "CustomTextEdit.h"
#include <QDebug>
#include <QDrag>
#include <QMimeData>
#include <QMouseEvent>

void MessagePrinterGui::print(const std::string& str) {
    appendText(QString::fromStdString(str));
}

void MessagePrinterGui::printNewLine() {
    appendText("\n");
}

//------------------------------------------------------

ScrollingTextEdit::ScrollingTextEdit(QWidget* parent) :
    QPlainTextEdit(parent) 
{}

void ScrollingTextEdit::appendText(const QString& str) {
    //qDebug() << str;
    QString qstr = toPlainText() + str;
    setPlainText(qstr);
    moveCursor(QTextCursor::End);
}

//------------------------------------------------------

DropTextEdit::DropTextEdit(QWidget* parent) :
    QPlainTextEdit(parent)
{}

void DropTextEdit::dragEnterEvent(QDragEnterEvent* event) {
    //qDebug() << "drag enter";
    if (event->mimeData()->hasUrls()) {
        event->setDropAction(Qt::DropAction::LinkAction);
        event->accept();
    }
}

// In order to drop a file onto a TextEdit dragMoveEvent needs to be overridden
// docs only tell about implementing dragEnterEvent and dropEvent
// THIS IS EXACTLY WHY I HATE THIS FU**ING PIECE OF SHIT QT SO MUCH, INSANE

void DropTextEdit::dragMoveEvent(QDragMoveEvent* event) {
    //qDebug() << "drag move";
}

void DropTextEdit::dropEvent(QDropEvent* event) {
    //qDebug() << "drop";
    if (event->mimeData()->hasUrls()) {
        QUrl url = event->mimeData()->urls().first();
        QString str = url.toLocalFile();
        //qDebug() << "drop " << str;
        fileDropped(str);
    }
}