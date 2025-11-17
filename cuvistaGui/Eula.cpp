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

#include <QPushButton>
#include <QVBoxLayout>
#include <QApplication>
#include <QSettings>
#include <QFile>

#include "Eula.h"
#include "MainData.hpp"

Eula::Eula() : 
    QMainWindow(nullptr) 
{
    ui.setupUi(this);

    //when declining eula we quit
    connect(ui.btnDecline, &QPushButton::clicked, this, &QApplication::exit);

    //when accepting eula we switch over to the main window
    auto fcn = [&] () {
        //save current version to registry
        QSettings settings("RainerMtb", "cuvista");
        settings.setValue("version", QString::fromStdString(CUVISTA_VERSION));

        //switch to main window
        hide();
        showMainWindow();
    };
    connect(ui.btnAccept, &QPushButton::clicked, this, fcn);

    //load markdown text data
    QFile file(":cuvistaGui/res/License.md");
    bool retval = file.open(QIODeviceBase::ReadOnly);
    QString md = QString::fromLocal8Bit(file.readAll());
    ui.textEdit->setMarkdown(md);
    file.close();
}

bool Eula::needToShowEula() {
    QSettings settings("RainerMtb", "cuvista");
    QString settingsVersion = settings.value("version", "").toString();
    return settingsVersion != QString::fromStdString(CUVISTA_VERSION);
}