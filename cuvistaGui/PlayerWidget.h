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

#include <QOpenGLWidget>
#include <QOpenGLFunctions_4_5_Core>
#include "Image2.hpp"

//player widget
class PlayerWidget : public QOpenGLWidget, protected QOpenGLFunctions_4_5_Core {
    Q_OBJECT

public:
    PlayerWidget(QWidget* parent);
    ~PlayerWidget();

    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;

public slots:
    void open(int h, int w, int stride, QImage imageWorking);
    void upload(int64_t frameIndex, ImageRGBA image);
    void playNextFrame(int64_t frameIndex);

private:
    GLuint fbo = 0;
    GLuint textures[2] = {0, 0};
    int texHeight = 32;
    int texWidth = 32;

    int64_t currentFrameIndex = 0;
};
