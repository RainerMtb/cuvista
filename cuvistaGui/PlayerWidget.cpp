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

#include <QDebug>
#include <cassert>
#include "PlayerWidget.h"

/*
* use QOpenGLContext::makeCurrent() before using native gl... functions
* otherwise graphics artifacts occur in the main window which does not even have a OpenGLWidget
* but makeCurrent() can be omitted in initializeGL, paintGL, resizeGL
*/

static void errorCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
	qDebug() << "OpenGL error: " << message;
}

PlayerWidget::PlayerWidget(QWidget* parent) :
	QOpenGLWidget(parent) {}

void PlayerWidget::initializeGL() {
	//qDebug() << "-- init";
	initializeOpenGLFunctions();
	glDebugMessageCallback(&errorCallback, 0);
	
	//create framebuffer
	glGenFramebuffers(1, &fbo);
	//create textures
	glGenTextures(2, textures);
	//set textures
	for (int i = 0; i < 2; i++) {
		glBindTexture(GL_TEXTURE_2D, textures[i]);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texWidth, texHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	}
}

//prepare texture size for this video
void PlayerWidget::open(int h, int w, int stride, QImage imageWorking) {
	makeCurrent();
	for (int i = 0; i < 2; i++) {
		glBindTexture(GL_TEXTURE_2D, textures[i]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_BGRA, GL_UNSIGNED_BYTE, imageWorking.bits());
	}

	//store video size
	texHeight = h;
	texWidth = w;
}

//upload texture for upcoming video frame
void PlayerWidget::upload(int64_t frameIndex, ImageRGBA image) {
	assert(image.w == texWidth && image.h == texHeight && "invalid image dimensions");
	makeCurrent();
	int idx = frameIndex % 2;
	glBindTexture(GL_TEXTURE_2D, textures[idx]);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texWidth, texHeight, GL_RGBA, GL_UNSIGNED_BYTE, image.data());
}

//show the next video frame on the widget
void PlayerWidget::playNextFrame(int64_t frameIndex) {
	currentFrameIndex = frameIndex;
	update(); //will trigger paintGL() on main thread
}

void PlayerWidget::paintGL() {
	//qDebug() << "-- paint " << currentFrameIndex;

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	//switch to texture holding next video frame to display
	int idx = currentFrameIndex % 2;
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textures[idx], 0);

	//switch framebuffers
	glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, defaultFramebufferObject());

	//centering image in opengl widget
	double x, y, w, h;
	if (width() * texHeight > texWidth * height()) {
		h = height();
		y = 0.0;
		w = h * texWidth / texHeight;
		x = (width() - w) / 2.0;

	} else {
		w = width();
		x = 0.0;
		h = w * texHeight / texWidth;
		y = (height() - h) / 2.0;
	}

	//blit the texture
	glBlitFramebuffer(0, 0, texWidth, texHeight, x, y + h, x + w, y, GL_COLOR_BUFFER_BIT, GL_LINEAR);
}

void PlayerWidget::resizeGL(int w, int h) {}

PlayerWidget::~PlayerWidget() {
	//delete opengl objects if had been initialized
	makeCurrent();
	if (textures[0] && textures[1]) glDeleteTextures(2, textures);
	if (fbo) glDeleteFramebuffers(1, &fbo);
}