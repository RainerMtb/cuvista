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

#include "AffineTransform.hpp"

const AffineTransform& AffineTransform::computeAffineDirect(const PointResult& p1, const PointResult& p2, const PointResult& p3) {
	//set up A and b, members of AffineTransform
	double* ap = A.data();
	double* bp = b.data();
	for (const PointResult& pr : { p1, p2, p3}) {
		ap[0] = pr.x;	ap[1] = pr.y;	ap[2] = 1.0;	ap[3] = 0.0;    ap[4] = 0.0;   ap[5] = 0.0;
		ap[6] = 0.0;    ap[7] = 0.0;    ap[8] = 0.0;    ap[9] = pr.x;  ap[10] = pr.y; ap[11] = 1.0;
		ap += 12; //move two rows of A

		bp[0] = pr.x + pr.u;
		bp[1] = pr.y + pr.v;
		bp += 2; //move two rows of b
	}

	//solve and write the 6 result values directly to this AffineTransform
	LUDecompositor<double> ludec(A);
	ludec.solveAffine(b.data(), array);
	setParam(array[0], array[1], array[2], array[3], array[4], array[5]);
	return *this;
}

const AffineTransform& AffineTransform::computeSimilarDirect(const PointResult& p1, const PointResult& p2) {
	Matd As = A.share(4, 4);
	As[0][0] = p1.x;   As[0][1] =  p1.y;   As[0][2] = 1.0;   As[0][3] = 0.0;
	As[1][0] = p1.y;   As[1][1] = -p1.x;   As[1][2] = 0.0;   As[1][3] = 1.0;
	As[2][0] = p2.x;   As[2][1] =  p2.y;   As[2][2] = 1.0;   As[2][3] = 0.0;
	As[3][0] = p2.y;   As[3][1] = -p2.x;   As[3][2] = 0.0;   As[3][3] = 1.0;

	double* bp = b.data();
	bp[0] = p1.x + p1.u;
	bp[1] = p1.y + p1.v;
	bp[2] = p2.x + p2.u;
	bp[3] = p2.y + p2.v;

	LUDecompositor<double> ludec(As);
	ludec.solveAffine(bp, array);
	setParam(array[0], array[1], array[2], array[3]);
	return *this;
}

bool AffineTransform::computeAffine(std::vector<PointResult>& points) {
	size_t count = points.size();
	assert(count >= 3 && "transform needs at least three points");
	Matd A = Matd::zeros(count * 2, 6);
	Matd b = Matd::zeros(count * 2, 1);
	const PointResult* p = points.data();
	for (size_t i = 0, k = 0; i < count; i++) {
		double xx = p->x;
		double yy = p->y;

		A[k][0] = xx;
		A[k][1] = yy;
		A[k][2] = 1.0;
		b[k][0] = xx + p->u;
		k++;

		A[k][3] = xx;
		A[k][4] = yy;
		A[k][5] = 1.0;
		b[k][0] = yy + p->v;
		k++;

		p++;
	}
	auto res = A.solveInPlace(b);
	setParam(res->at(0, 0), res->at(1, 0), res->at(2, 0), res->at(3, 0), res->at(4, 0), res->at(5, 0));
	return res.has_value();
}

const AffineTransform& AffineSolver::computeSimilar(std::span<PointResult> points) {
	std::vector<PointBase> baseVector(points.begin(), points.end());
	return computeSimilar(baseVector);
}

const AffineTransform& AffineSolver::computeSimilar(std::span<PointContext> points) {
	std::vector<PointBase> baseVector(points.begin(), points.end());
	return computeSimilar(baseVector);
}

const AffineTransform& AffineSolverSimple::computeSimilar(std::span<PointBase> points) {
	assert(points.size() >= 2 && "similar transform needs at least two points");
	size_t m = points.size() * 2;
	Matd A = Adata.share(m, 4);
	A.setValues(0.0);
	Matd b = bdata.share(m, 1);
	b.setValues(0.0);
	auto it = points.begin();
	for (size_t i = 0; i < points.size(); i++) {
		double xx = it->x;
		double yy = it->y;
		size_t k = i * 2;

		A[k][0] = xx;
		A[k][1] = yy;
		A[k][2] = 1.0;
		A[k + 1][0] = yy;
		A[k + 1][1] = -xx;
		A[k + 1][3] = 1.0;

		b[k][0] = xx + it->u;
		b[k + 1][0] = yy + it->v;

		it++;
	}
	//savePointResults(begin, count);
	QRDecompositor<double> qrdec(A);
	auto res = qrdec.solve(b);
	setParam(res->at(0, 0), res->at(1, 0), res->at(2, 0), res->at(3, 0));
	return *this;
}

const AffineTransform& AffineSolverFast::computeSimilar(std::span<PointBase> points) {
	assert(points.size() >= 2 && "similar transform needs at least two points");
	size_t m = points.size() * 2;
	Matd A = Adata.share(6, m);  //A is transposed when compared to loop version

	int dy = points[0].y;         //represents a2, needs to be smallest value
	int dx = points[1].x;         //represents a3
	long long int nn = 0;      //int could overflow
	long long int s0 = 0;      //in docs s1
	long long int s1 = 0;      //in docs s2

	double* p = A.data() + m * 4; //5th row holds adjusted x and y values
	double* q = A.data() + m * 5; //6th row holds adjusted b values

	//accumulate s0, s1, nn and adjust coords
	auto it = points.begin();
	for (size_t idx = 0; idx < m; ) {
		int x = it->x - dx;
		p[idx] = x;
		s0 += x;
		nn += x * x;
		q[idx] = it->x + it->u - dx;
		idx++;

		int y = it->y - dy;
		p[idx] = y;
		s1 += y;
		nn += y * y;
		q[idx] = it->y + it->v - dy;
		idx++;

		it++;
	}

	//compute parameters
	double sign0 = p[0] < 0 ? -1.0 : 1.0;
	double n = std::sqrt(nn) * sign0;
	double b = p[0] + n;
	double sign2 = sign0 * n * b < p[3] * s1 ? -1.0 : 1.0;
	double t = sign2 * std::sqrt(points.size() * nn - s0 * s0 - s1 * s1);
	double z = b * (n + t) - p[3] * s1;
	double e = n / t;
	double f = s1 / (b * t);
	
	double sn = s0 + n;
	double g = sn / (b * t);
	double h = (p[3] / b * (sn * sn + s1 * s1) - s1 * (n + t)) / (t * z);
	double j = sn * (t + n) / (t * z);
	double k = p[3] * n * sn / (t * z);
	
	double rd[] = { n, -n, t / n, t / n };

	//first row of matrix A
	threadPool.add([&] {
		double* a = A.data();
		a[0] = b / n;
		a[1] = 0.0;
		a[2] = 0.0;
		a[3] = p[3] / n;
		for (size_t idx = 4; idx < m; ) {
			a[idx] = p[idx] / n;
			idx++;
			a[idx] = p[idx] / n;
			idx++;
		}
	});
	//second row
	threadPool.add([&] {
		double* a = A.data() + m;
		a[0] = 0.0;
		a[1] = 1.0 + p[0] / n;
		a[2] = -p[3] / n;
		a[3] = 0.0;
		for (size_t idx = 4; idx < m; ) {
			a[idx] = -p[idx + 1] / n;
			idx++;
			a[idx] = p[idx - 1] / n;
			idx++;
		}
	});
	//third row
	threadPool.add([&] {
		double* a = A.data() + m * 2;
		a[0] = -s0 / n;
		a[1] = s1 / n;
		a[2] = 1.0 + e - p[3] * f;
		a[3] = -p[3] * g;
		for (size_t idx = 4; idx < m; ) {
			a[idx] = e - p[idx + 1] * f - p[idx] * g;
			idx++;
			a[idx] = p[idx - 1] * f - p[idx] * g;
			idx++;
		}
	});
	//fourth row
	threadPool.add([&] {
		double* a = A.data() + m * 3;
		a[0] = -s1 / n;
		a[1] = -s0 / n;
		a[2] = 0.0;
		a[3] = 1.0 + e + p[3] * h;
		for (size_t idx = 4; idx < m; ) {
			a[idx] = p[idx + 1] * j + p[idx] * h - k;
			idx++;
			a[idx] = e + p[idx] * h - p[idx - 1] * j;
			idx++;
		}
	});
	threadPool.wait();
	//A.trans().toConsole();

	//back substitution step 1
	for (size_t k = 0; k < 4; k++) {
		double* row = A.data() + m * k;
		double s = std::inner_product(row + k, row + m, q + k, 0.0);
		s /= -A[k][k];
		for (size_t i = k; i < m; i++) {
			q[i] += s * A[k][i];
		}
	}

	//back substitution step 2, only need first four values
	for (int k = 3; k >= 0; k--) {
		q[k] /= -rd[k];
		for (int i = 0; i < k; i++) {
			q[i] -= q[k] * A[k][i];
		}
	}

	//readjust transform parameter values back to given points
	setParam(q[0], q[1], -dx * q[0] - dy * q[1] + q[2] + dx, dx * q[1] - dy * q[0] + q[3] + dy);
	return *this;
}

std::array<double, 6> AffineTransform::toArray() const {
	return { array[0], array[1], array[2], array[3], array[4], array[5] };
}

std::ostream& operator << (std::ostream& os, const AffineTransform& trf) {
	os << "trf: s=" << trf.scale() << ", r=" << trf.rot() << ", dx=" << trf.dX() << ", dy=" << trf.dY();
	return os;
}