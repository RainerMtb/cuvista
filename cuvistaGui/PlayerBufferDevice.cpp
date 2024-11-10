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
#include "PlayerBufferDevice.h"

PlayerBufferDevice::PlayerBufferDevice(size_t bufferSize) :
    bufferData(bufferSize) {}

qint64 PlayerBufferDevice::readData(char* data, qint64 maxSize) {
    qDebug() << "-- buffer read" << maxSize;
    std::memcpy(data, bufferData.data() + bufferPos, maxSize);
    bufferPos += maxSize;
    return maxSize;
}

qint64 PlayerBufferDevice::writeData(const char* data, qint64 maxSize) {
    return 0;
}

bool PlayerBufferDevice::isSequential() const {
    return true;
}

qint64 PlayerBufferDevice::bytesAvailable() const {
    int avail = bufferData.size() - bufferPos + QIODevice::bytesAvailable();
    qDebug() << "-- buffer avail" << avail;
    return avail;
}

qint64 PlayerBufferDevice::size() const {
    qDebug() << "-- buffer size" << bufferData.size();
    return bufferData.size();
}