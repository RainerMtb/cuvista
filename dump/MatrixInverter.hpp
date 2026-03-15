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

#include "MatInverter.hpp"
#include "Mat.hpp"

class PseudoInverter : public MatInverter<double> {
private:
	size_t s;
	Matd I;
	Matd Xk, Xk1, Yk;

public:
	PseudoInverter(Matd& A, size_t s);

	std::optional<Matd> inv();
};

template <class T> class MatPseudoInverter : public MatInverter<T> {
protected:
	T eps = 0.05f;
	T e = 1.0f;
	size_t s;
	int maxIter = 20;
	Mat<T> I;

public:
	MatPseudoInverter(Mat<T>& A, size_t s) :
		MatInverter<T>(A),
		s { s },
		I { Mat<T>::eye(s) } {}
};

//3.19
template <class T> class IterativePseudoInverse1 : public MatPseudoInverter<T> {
	using MatPseudoInverter<T>::s;
	using MatPseudoInverter<T>::I;
	using MatInverter<T>::A;

public:
	IterativePseudoInverse1(Mat<T>& A, size_t s) :
		MatPseudoInverter<T>(A, s) {}

	std::optional<Mat<T>> inv() {
		Mat<T> Xk = Mat<T>::eye(s).timesEach(1 / A.normF());
		Mat<T> Bk, Sk, Xk1;

		int i = 0;
		for (; i < this->maxIter && this->e > this->eps; i++) {
			Bk = A.times(Xk);
			Sk = Bk.times(Bk.minus(I));
			Xk1 = Xk.times(I.timesEach(2).minus(Bk)).times(I.timesEach(3).minus(Bk.timesEach(2)).plus(Sk)).times(I.plus(Sk));
			Xk = Xk1;
			this->e = A.times(Xk).minus(I).normF2();
		}
		return { Xk };
	}
};

//3.18 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
template <class T> class IterativePseudoInverse2 : public MatPseudoInverter<T> {
	using MatPseudoInverter<T>::s;
	using MatPseudoInverter<T>::I;
	using MatInverter<T>::A;

public:
	IterativePseudoInverse2(Mat<T>& A, size_t s) :
		MatPseudoInverter<T>(A, s) {}

	std::optional<Mat<T>> inv() {
		Mat<T> Xk = Mat<T>::eye(s).timesEach(1 / A.normF());
		Mat<T> Yk, Xk1;

		int i = 0;
		for (; i < this->maxIter && this->e > this->eps; i++) {
			Yk = I.minus(A.times(Xk));
			Xk1 = Xk.times(I.plus(Yk.times(I.plus(Yk.times(I.plus(Yk))))));
			Xk = Xk1;
			//this->e = A.times(Xk).minus(I).normF2();
			this->e = Yk.normF2();
		}
		return { Xk };
	}
};

//3.20
template <class T> class IterativePseudoInverse3 : public MatPseudoInverter<T> {
	using MatPseudoInverter<T>::s;
	using MatPseudoInverter<T>::I;
	using MatInverter<T>::A;

public:
	IterativePseudoInverse3(Mat<T>& A, size_t s) :
		MatPseudoInverter<T>(A, s) {}

	std::optional<Mat<T>> inv() {
		Mat<T> Xk = Mat<T>::eye(s).timesEach(1 / A.normF());
		Mat<T> Bk, Xk1;

		int i = 0;
		for (; i < this->maxIter && this->e > this->eps; i++) {
			Bk = A.times(Xk);
			Xk1 = Xk.timesEach(0.5).times(I.timesEach(9).minus(Bk.times(I.timesEach(16).minus(Bk.times(I.timesEach(14).minus(Bk.times(I.timesEach(6).minus(Bk))))))));
			Xk = Xk1;
			this->e = A.times(Xk).minus(I).normF2();
		}
		return { Xk };
	}
};

//1.7
template <class T> class IterativePseudoInverse4 : public MatPseudoInverter<T> {
	using MatPseudoInverter<T>::s;
	using MatPseudoInverter<T>::I;
	using MatInverter<T>::A;

public:
	IterativePseudoInverse4(Mat<T>& A, size_t s) :
		MatPseudoInverter<T>(A, s) {}

	std::optional<Mat<T>> inv() {
		Mat<T> Xk = Mat<T>::eye(s).timesEach(1 / A.normF());
		Mat<T> Bk, Xk1;

		int i = 0;
		for (; i < this->maxIter && this->e > this->eps; i++) {
			Bk = A.times(Xk);
			Xk1 = Xk.minus(Xk.timesEach(0.5).times(Bk.times(Bk).minus(I)));
			Xk = Xk1;
			this->e = A.times(Xk).minus(I).normF2();
		}
		return { Xk };
	}
};
