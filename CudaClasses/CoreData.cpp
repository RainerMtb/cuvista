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

#include "CoreData.hpp"

bool PointResult::isValid() const {
	return result > PointResultType::RUNNING;
}

int PointResult::resultValue() const {
	return static_cast<int>(result) - static_cast<int>(PointResultType::RUNNING);
}

bool PointResult::equal(double a, double b, double tol) const {
	return (isnan(a) && isnan(b)) || std::fabs(a - b) <= tol;
}

bool PointResult::equals(const PointResult& other, double tol) const {
	bool checkType = result == other.result; //type of result
	bool checkIndex = idx == other.idx && ix0 == other.ix0 && iy0 == other.iy0 && px == other.px && py == other.py && x == other.x && y == other.y;
	bool checkU = equal(u, other.u, tol); //displacement in X
	bool checkV = equal(v, other.v, tol); //displacement in Y
	return checkType && checkIndex && checkU && checkV;
}

bool PointResult::operator == (const PointResult& other) const {
	return equals(other, 0.0);
}

bool PointResult::operator != (const PointResult& other) const {
	return !(*this == other);
}

std::ostream& operator << (std::ostream& out, const PointResult& res) {
	out << "ix0=" << res.ix0 << ", iy0=" << res.iy0 << ", u=" << res.u << ", v=" << res.v;
	return out;
}