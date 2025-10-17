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

#include <iostream>
#include <iomanip>
#include <fstream>
#include <functional>
#include <vector>
#include <random>
#include <sstream>
#include <format>
#include <numbers>
#include <optional>
#include <span>

#include "LUDecompositor.hpp"
#include "QRDecompositor.hpp"
#include "QRDecompositorUD.hpp"
#include "SVDecompositor.hpp"

#include "CoreMat.hpp"
#include "ThreadPoolBase.h"
#include "MatIterator.hpp"
#include "OutputLine.hpp"
#include "Util.hpp"
#include "Image.hpp"

//---------------------------------------------------------
//------------------------ CLASS MATROW FOR [][] INDEXING
//---------------------------------------------------------

template <class T> class MatRow {

private:
	T* rowptr;
	size_t cols;

	std::ostream& print(std::ostream& out) const;

public:
	MatRow(T* rowptr, size_t cols) : 
		rowptr { rowptr }, 
		cols { cols } 
	{}

	//operator from row to value&
	T& operator [] (size_t col) {
		assert(col < cols && "column index out of bounds");
		return rowptr[col];
	}

	//print values of this row
	friend std::ostream& operator << (std::ostream& out, const MatRow<T>& mr) {
		return mr.print(out);
	}
};


//---------------------------------------------------------
//------------------------ MAIN MATRIX CLASS
//---------------------------------------------------------


template <class T> class Mat : public CoreMat<T> {

	friend class Affine2D;
	friend class MatRow<T>;

public:
	using CoreMat<T>::array;
	using CoreMat<T>::rows;
	using CoreMat<T>::cols;
	using CoreMat<T>::at;
	using CoreMat<T>::addr;

	enum class Direction {
		VERTICAL = 0,
		HORIZONTAL = 1,
	};

protected:
	//pretty print mat values
	inline static int printDigits = 5;

	//maximum width and height
	static constexpr size_t MAX_DIM = 1 << 30;
	static constexpr int MAX_DIGITS = 25;

	//noop thread pool
	inline static ThreadPoolBase defaultPool;

	//create mat using existing data array, sharing memory
	Mat(T* array, size_t rows, size_t cols, bool ownData) : 
		CoreMat<T>(array, rows, cols, ownData) {}

	//create new mat, allocate data array
	Mat(size_t rows, size_t cols) : 
		Mat<T>(new T[rows * cols], rows, cols, true) {}

	//apply unary op
	Mat<T> unaryOpSafe(const Mat<T>& other, std::function <T(size_t, size_t)> op) const {
		assert(rows() == other.rows() && cols() == other.cols() && "dimensions do not agree");
		return generate(rows(), cols(), op);
	}

	size_t clampUnsigned(size_t valuePositive, size_t valueNegative, size_t lo, size_t hi) const {
		size_t out = 0;
		if (valuePositive < lo + valueNegative) out = lo;
		else if (valuePositive > hi + valueNegative) out = hi;
		else out = valuePositive - valueNegative;
		return out;
	}

	static Mat<T> concatFunc(Direction catDim, std::span<const Mat<T>*> span) {
		if (span.size() == 0) return Mat<T>();

		//compute dimension of resulting mat
		size_t cat = static_cast<size_t>(catDim);
		std::vector<size_t> dims = { span[0]->rows(), span[0]->cols() };

		for (size_t i = 1; i < span.size(); i++) {
			std::vector<size_t> d = { span[i]->rows(), span[i]->cols() };
			dims[cat] += d[cat];
			assert(dims[1 - cat] == d[1 - cat] && "dimensions do not agree for concatenation");
		}

		//allocate space for combined mat
		Mat<T> out = allocate(dims[0], dims[1]);
		dims = { 0, 0 };

		//fill in mats
		for (const Mat<T>* ptr : span) {
			out.setArea(dims[0], dims[1], *ptr);
			dims[cat] += ptr->dim(cat);
		}
		return out;
	}

private:
	constexpr static T eps() { return 0; }

	//compute array index from interator index
	virtual std::function<size_t(size_t)> indexFunc(Direction dir) const {
		if (dir == Direction::HORIZONTAL) return [&] (size_t i) { return i; }; //iterator for row major order
		else return [&] (size_t i) { return i % rows() * cols() + i / rows(); }; //iterate in column major order
	}

	//create output lines
	std::vector<OutputLine> rowStrings(const std::string& title, int digits) const {
		if (digits <= 0 || digits >= 25) digits = printDigits;
		std::vector<OutputLine> lines(rows());
		std::string prefix = title.empty() ? "[" : title + " = [";
		//number of spaces between numbers
		const int spaces = 3;

		//title and opening [ bracket
		lines[0].add(prefix);
		for (size_t r = 0; r < rows(); r++) {
			lines[r].pad(prefix.length());
			lines[r].add('[');
		}

		size_t posDecimalMax;
		size_t maxlen = prefix.length() + 1 + spaces;
		//iterate over columns
		for (size_t c = 0; c < cols(); c++) {
			posDecimalMax = 0;
			//iterate over rows
			for (size_t r = 0; r < rows(); r++) {
				//analyse a value
				double d = (double) at(r, c);
				double absd = std::abs(d);
				//format value
				std::string str = "";
				if (absd < 1e5 && (d - rint(d)) == 0) str = std::format("{:.1f}", d);
				else if (absd < 1e-5)                 str = std::format("{:.{}e}", d, digits);
				else if (absd < 1e5)                  str = std::format("{:.{}f}", d, digits);
				else                                  str = std::format("{:.{}e}", d, digits);
				//store value
				lines[r].setNumStr(str, &posDecimalMax);
			}

			//insert values
			for (size_t r = 0; r < rows(); r++) lines[r].pad(maxlen);
			for (size_t r = 0; r < rows(); r++) lines[r].addValue(posDecimalMax, &maxlen);
			maxlen += spaces;
		}

		//closing bracket
		for (size_t r = 0; r < rows(); r++) {
			lines[r].pad(maxlen);
			lines[r].add(']');
		}
		//last row one more closing bracket
		lines[rows() - 1ull].add(']');
		return lines;
	}

	//format to output stream
	std::ostream& print(std::ostream& out, const std::string& title = "", int digits = -1) const {
		for (OutputLine line : rowStrings(title, digits)) {
			out << line.mStr << std::endl;
		}
		return out;
	}

	//format to output stream
	std::wostream& print(std::wostream& out, const std::string& title = "", int digits = -1) const {
		for (OutputLine line : rowStrings(title, digits)) {
			out << std::wstring(line.mStr.cbegin(), line.mStr.cend()) << std::endl;
		}
		return out;
	}

public:
	//default constructor produces invalid mat
	Mat() : 
		Mat<T>(nullptr, 0, 0, true) {}

	//implicit constructor for scalar Matrix
	Mat(T val) : 
		Mat<T>(new T[1] {val}, 1, 1, true) {}

	//copy constructor from CoreMat
	Mat(const CoreMat<T>& other) :
		CoreMat<T>(other) {}

	//conversion from other Mat
	template <class R> Mat(const CoreMat<R>& other) :
		Mat<T>(other.rows(), other.cols())
	{
		setValues([&] (size_t r, size_t c) { return (T) (other.at(r, c)); });
	}

	//move constructor from CoreMat
	Mat(CoreMat<T>&& other) noexcept :
		CoreMat<T>(other) {}

	//print precision
	static int precision(int digits) {
		if (digits > 0 && digits < MAX_DIGITS) {
			printDigits = digits;
		}
		return printDigits;
	}

	//print precision
	static int precision() {
		return printDigits;
	}

	//iterators
	auto begin(Direction dir = Direction::HORIZONTAL)           { return MatIterator<T>(this->data(), 0, indexFunc(dir)); }
	auto end(Direction dir = Direction::HORIZONTAL)	            { return MatIterator<T>(this->data(), rows() * cols(), indexFunc(dir)); }
	auto cbegin(Direction dir = Direction::HORIZONTAL) const    { return MatIterator<const T>(this->data(), 0, indexFunc(dir)); }
	auto cend(Direction dir = Direction::HORIZONTAL) const      { return MatIterator<const T>(this->data(), rows() * cols(), indexFunc(dir)); }

	//---------------------------------------------------------
	//------------------------ STATIC MAKERS
	//---------------------------------------------------------

	static Mat<T> allocate(size_t rows, size_t cols) {
		assert(rows > 0 && cols > 0 && rows < MAX_DIM && cols < MAX_DIM && "invalid dimensions for allocation");
		return Mat<T>(rows, cols);
	}

	static Mat<T> generate(size_t rows, size_t cols, std::function<T(size_t, size_t)> supplier, ThreadPoolBase& pool = defaultPool) {
		Mat<T> out = allocate(rows, cols);
		out.setArea(0, 0, rows, cols, supplier, pool);
		return out;
	}

	static Mat<T> fromArray(size_t rows, size_t cols, T* data, bool ownData = true) {
		return Mat<T>(data, rows, cols, ownData);
	}

	static Mat<T> fromRowData(size_t rows, size_t cols, size_t stride, const T* data) {
		assert(stride >= cols && "invalid value for stride");
		return generate(rows, cols, [&] (size_t r, size_t c) { return data[r * stride + c]; });
	}

	static Mat<T> fromRowData(size_t rows, size_t cols, const T* data) {
		return fromRowData(rows, cols, cols, data);
	}

	static Mat<T> fromRowData(const std::vector<std::vector<T>>& data) {
		assert(data.size() > 0 && data[0].size() > 0 && "invalid data size");
		for (size_t r = 1; r < data.size(); r++) assert(data[r].size() == data[0].size() && "invalid data size");
		return generate(data.size(), data[0].size(), [&] (size_t r, size_t c) { return data[r][c]; });
	}

	static Mat<T> fromRowData(size_t rows, size_t cols, const std::initializer_list<T>& data) {
		assert(rows * cols == data.size() && "invalid number of initializer values");
		Mat<T> out = allocate(rows, cols);
		std::copy(data.begin(), data.end(), out.begin());
		return out;
	}

	static Mat<T> fromRowData(size_t rows, size_t cols, const std::vector<T>& data) {
		assert(rows * cols == data.size() && "invalid data size");
		Mat<T> out = allocate(rows, cols);
		std::copy(data.begin(), data.end(), out.begin());
		return out;
	}

	static Mat<T> fromRowData(size_t rows, size_t cols, std::span<T> data) {
		assert(rows * cols == data.size() && "invalid data size");
		Mat<T> out = allocate(rows, cols);
		std::copy(data.begin(), data.end(), out.begin());
		return out;
	}

	static Mat<T> fromRow(const std::initializer_list<T>& dataList) {
		return fromRowData(1, dataList.size(), dataList);
	}

	static Mat<T> fromRow(auto... args) {
		size_t count = sizeof...(args);
		T values[] { args... };
		return generate(1, count, [&] (size_t r, size_t c) { return values[c]; });
	}

	static Mat<T> fromBinaryFile(const std::string& filename) {
		std::ifstream file(filename, std::ios::binary);
		size_t rows = 0;
		size_t cols = 0;
		size_t sizT = 0;
		file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
		file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
		file.read(reinterpret_cast<char*>(&sizT), sizeof(size_t));

		Mat<T> mat = Mat<T>::allocate(rows, cols);
		if (file.good() && sizT == sizeof(T)) {
			file.read(reinterpret_cast<char*>(mat.addr(0, 0)), rows * cols * sizeof(T));
		}
		file.close();
		return file.good() ? mat : Mat<T>();
	}

	//read text file
	//values must be separated by whitespaces
	static Mat<T> fromTextFile(const std::string& filename) {
		std::ifstream file(filename);
		std::vector<T> numbers;
		std::vector<size_t> dims;
		for (std::string line; std::getline(file, line);) {
			dims.push_back(0);
			std::istringstream stream(line);
			T value = 0;
			while (stream.eof() == false && stream.fail() == false) {
				stream >> value;
				numbers.push_back(value);
				dims[dims.size() - 1]++;
			}
		}

		Mat<T> out;
		if (dims.size() > 0 && dims[0] > 0 && std::all_of(dims.begin(), dims.end(), [&] (size_t i) { return i == dims[0]; })) {
			out = Mat<T>::fromRowData(dims.size(), dims[0], numbers);
		}

		return out;
	}

	static Mat<T> eye(size_t size) {
		return generate(size, size, [] (size_t r, size_t c) { return r == c ? (T) 1 : (T) 0; });
	}

	static Mat<T> zeros(size_t rows, size_t cols) {
		return generate(rows, cols, [] (size_t r, size_t c) { return (T) 0; });
	}

	static Mat<T> values(size_t rows, size_t cols, const T value) {
		return generate(rows, cols, [value] (size_t r, size_t c) { return value; });
	}

	static Mat<T> rand(size_t rows, size_t cols, T lowerBound = 0, T upperBound = 1) {
		static std::random_device generator;
		std::uniform_real_distribution dist(lowerBound, upperBound);
		return generate(rows, cols, [&] (size_t r, size_t c) { return dist(generator); });
	}

	static Mat<T> rand(size_t rows, size_t cols, T lowerBound, T upperBound, T maxCondition) {
		Mat<T> out;
		T cond = maxCondition;
		while (cond >= maxCondition) {
			out = rand(rows, cols, lowerBound, upperBound);
			cond = out.cond();
		}
		return out;
	}

	static Mat<T> hilb(size_t n) {
		return generate(n, n, [] (size_t r, size_t c) {return 1 / (1 + (T) r + (T) c); });
	}

	static Mat<T> concatVert(std::initializer_list<Mat<T>> matsList) {
		return concat(Direction::VERTICAL, matsList);
	}

	static Mat<T> concatHorz(std::initializer_list<Mat<T>> matsList) {
		return concat(Direction::HORIZONTAL, matsList);
	}

	static Mat<T> concat(Direction catDim, std::initializer_list<Mat<T>> matsList) {
		std::vector<const Mat<T>*> vec(matsList.size());
		std::transform(matsList.begin(), matsList.end(), vec.begin(), [&] (const Mat<T>& m) { return &m; });
		return concatFunc(catDim, vec);
	}

	static Mat<T> concatVert(auto... mats) {
		std::vector<const Mat<T>*> vec;
		([&] { vec.push_back(&mats); } (), ...);
		return concatFunc(Direction::VERTICAL, vec);
	}

	static Mat<T> concatHorz(auto... mats) {
		std::vector<const Mat<T>*> vec;
		([&] { vec.push_back(&mats); } (), ...);
		return concatFunc(Direction::HORIZONTAL, vec);
	}

	static Mat<T> concat(Direction catDim, auto... mats) {
		std::vector<const Mat<T>*> vec;
		([&] { vec.push_back(&mats); } (), ...);
		return concatFunc(catDim, vec);
	}

	//---------------------------------------------------------
	//------------------------ QUERIES FROM MAT
	//---------------------------------------------------------


	//check if data array is shared between mats
	bool isShared(const Mat<T>& other) const { return array == other.array; }

	//value from mat or default value when out of bounds
	const T& atOrElse(int row, int col, const T& defaultValue) const {
		if (row >= 0 && col >= 0 && this->isValidIndex(row, col)) return at(row, col);
		else return defaultValue;
	}

	Mat<T> abs() const { return unaryOp([] (T f) { return std::abs(f); }); }

	T sum() const { return std::accumulate(cbegin(), cend(), T(0)); }

	T max() const { return *std::max_element(cbegin(), cend()); }

	T min() const { return *std::min_element(cbegin(), cend()); }

	template <class R> int compare(const Mat<R>& other, T tolerance = eps()) const {
		if (rows() > other.rows()) return 1;
		if (rows() < other.rows()) return -1;

		if (cols() > other.cols()) return 1;
		if (cols() < other.cols()) return -1;

		for (size_t r = 0; r < rows(); r++) {
			for (size_t c = 0; c < cols(); c++) {
				const T& a = at(r, c);
				const T& b = other.at(r, c);
				if (a < b - tolerance) return -1;
				else if (a > b + tolerance) return 1;
				else if (a >= b - tolerance && a <= b + tolerance) continue;
				else return -2;
			}
		}
		return 0;
	}

	template <class R> bool equals(const Mat<R>& other, T tolerance = eps()) const {
		return compare(other, tolerance) == 0;
	}

	template <class R> bool equalsExact(const Mat<R>& other) const {
		return compare(other, 0) == 0;
	}

	bool equalsIdentity(T tolerance = eps()) const {
		return rows() == cols() && compare(Mat<T>::eye(rows()), tolerance) == 0;
	}

	bool isSymmetric(T tolerance = eps()) const {
		return compare(trans(), tolerance) == 0;
	}

	bool isFinite() const {
		return std::all_of(cbegin(), cend(), [] (const T& a) { return std::isfinite(a); });
	}

	//compute 1-norm of Mat
	T norm1() const {
		T maxSum = std::abs(at(0, 0));
		for (size_t c = 0; c < cols(); c++) {
			T colSum = std::abs(at(0, c));
			for (size_t r = 1; r < rows(); r++) {
				colSum += std::abs(at(r, c));
			}
			if (std::isnan(colSum) || colSum > maxSum) maxSum = colSum;
		}
		return maxSum;
	}

	//compute Frobenius-norm of Mat
	T normF2() const {
		return std::accumulate(cbegin(), cend(), T(0), [] (const T& a, const T& b) { return a + b * b; });
	}

	//compute square of Frobenius-norm of Mat
	T normF() const {
		return std::sqrt(normF2());
	}

	//compute inf-norm of Mat
	T normInf() const {
		return trans().norm1();
	}

	Mat<T> diag() const {
		size_t s = std::min(rows(), cols());
		return Mat<double>::generate(s, 1, [&] (size_t r, size_t c) { return at(r, r); });
	}

	uint64_t crc() const {
		util::CRC64 crc64;
		for (size_t r = 0; r < rows(); r++) {
			for (size_t c = 0; c < cols(); c++) {
				crc64.addDirect(at(r, c));
			}
		}
		return crc64.result();
	}

	//---------------------------------------------------------
	//------------------------ OUTPUT
	//---------------------------------------------------------

	virtual Mat<T>& toConsole(const std::string& title = "", int digits = -1) {
		print(std::cout, title, digits);
		return *this;
	}

	virtual Mat<T> toConsole(const std::string& title = "", int digits = -1) const {
		print(std::cout, title, digits);
		return *this;
	}

	virtual std::string toString(const std::string& title = "", int digits = -1) const {
		std::stringstream buf;
		print(buf, title, digits);
		return buf.str();
	}

	virtual std::wstring toWString(const std::string& title = "", int digits = -1) const {
		std::wstringstream buf;
		print(buf, title, digits);
		return buf.str();
	}

	std::vector<T> toVector(Direction dir = Direction::HORIZONTAL) const {
		return std::vector<T>(cbegin(dir), cend(dir));
	}

	void saveAsText(const std::string& filename, bool append = false, const std::string& delimiter = ";") const {
		std::ios::openmode mode = append ? std::ios::app : std::ios::out;
		std::ofstream file(filename, mode);
		file << rows() << delimiter << cols() << std::endl;
		file.precision(12);
		for (size_t r = 0; r < rows(); r++) {
			size_t c = 0;
			for (; c < cols() - 1; c++) file << at(r, c) << delimiter;
			file << at(r, c) << std::endl;
		}
	}

	void saveAsBinary(const std::string& filename) const {
		std::ofstream file(filename, std::ios::binary);
		size_t rr = rows();
		size_t cc = cols();
		file.write((char*) &rr, sizeof(size_t));
		file.write((char*) &cc, sizeof(size_t));
		size_t sizT = sizeof(T);
		file.write((char*) &sizT, sizeof(size_t));
		for (size_t r = 0; r < rows(); r++) {
			for (size_t c = 0; c < cols(); c++) {
				T val = at(r, c);
				file.write((char*) &val, sizeof(T));
			}
		}
	}

	void saveAsBMP(const std::string& filename, T scale = 1) const {
		int h = int(this->h);
		int w = int(this->w);
		im::ImageMatShared(h, w, w, this->array).saveAsBMP(filename, scale);
	}

	//---------------------------------------------------------
	//------------------------ OPERATIONS ON MAT
	//---------------------------------------------------------
 
	//set values to part of mat from source mat
	Mat<T>& setArea(size_t r0, size_t c0, const Mat<T>& input) {
		return setArea(r0, c0, input.rows(), input.cols(), [&] (size_t r, size_t c) { return input.at(r, c); });
	}

	//set values to part of mat from source function
	Mat<T>& setArea(size_t r0, size_t c0, size_t rows, size_t cols, std::function<T(size_t, size_t)> supplier, ThreadPoolBase& pool = defaultPool) {
		auto func = [&] (size_t r) {
			for (size_t c = 0; c < cols; c++) {
				set(r + r0, c + c0, supplier(r, c));
			}
		};
		pool.addAndWait(func, 0, rows);
		return *this;
	}

	//set values to part of mat from source function
	Mat<T>& setArea(const std::function<T(size_t, size_t)>& supplier, ThreadPoolBase& pool = defaultPool) {
		return setArea(0, 0, rows(), cols(), supplier, pool);
	}
	
	//set all values copied from source mat
	Mat<T>& setValues(const Mat<T>& input) {
		return setArea(0, 0, input);
	}

	//fill mat with consant value
	Mat<T>& setValues(T value) {
		return setValues([value] (size_t r, size_t c) { return value; });
	}

	//set each value according to function
	Mat<T>& setValues(std::function<T(size_t, size_t)> supplier, ThreadPoolBase& pool = defaultPool) {
		return setArea(0, 0, rows(), cols(), supplier, pool);
	}

	//set diagonal values to given value, otherwise 0
	Mat<T>& setDiag(T value) {
		return setValues([value] (size_t r, size_t c) { return r == c ? value : T(0); });
	}

	//copy data directly from input array
	Mat<T>& setData(const Mat<T>& input) {
		assert(this->numel() == input.numel() && "data size mismatch");
		std::copy_n(input.data(), input.numel(), this->data());
		return *this;
	}

	Mat<T> setData(T value) {
		std::fill(this->data(), this->data() + this->numel(), value);
		return *this;
	}

	Mat<T>& setValuesByRow(const std::initializer_list<T> dataList) {
		return setValuesByRow(0, 0, rows(), cols(), dataList);
	}

	Mat<T>& setValuesByRow(size_t r0, size_t c0, size_t rows, size_t cols, const std::initializer_list<T> dataList) {
		assert(rows * cols == dataList.size() && "invalid number of items");
		return setArea(r0, c0, rows, cols, [&] (size_t r, size_t c) { return *(dataList.begin() + r * cols + c); });
	}

	//set constant value to one entry of mat
	void set(size_t row, size_t col, T value) {
		assert(this->isValidIndex(row, col) && "index out of bounds");
		at(row, col) = value;
	}

	//copy and resize mat, either discard values or pad with value
	Mat<T> resize(size_t rows, size_t cols, T value = 0) {
		Mat<T> out = Mat<T>::values(rows, cols, value);
		out.setArea(std::min(this->rows(), rows), std::min(this->cols(), cols), this);
		return out;
	}

	//reuse memory for a new Mat
	Mat<T> share(size_t rows, size_t cols) {
		assert(rows * cols <= this->numel() && "new mat does not fit inside memory");
		return Mat<T>(array, rows, cols, false);
	}

	//---------------------------------------------------------
	//------------------------ OPERATOR OVERLOADING
	//---------------------------------------------------------

	//stream output
	friend std::ostream& operator << (std::ostream& out, const Mat<T>& m) {
		return m.print(out);
	}

	//spaceship operator
	template <class R> auto operator <=> (const Mat<R>& other) const { return compare(other); }

	//equals operator is not derived from <=> ??
	template <class R> bool operator == (const Mat<R>& other) const { return compare(other) == 0; }

	template <class R> bool operator != (const Mat<R>& other) const { return compare(other) != 0; }

	explicit operator bool() const { return this->numel() > 0; }

	Mat<T> operator + (const Mat<T>& b) { return plus(b); }

	Mat<T> operator + (T b) { return plus(b); }

	friend Mat<T> operator + (T b, const Mat<T>& a) { return a.plus(b); }

	Mat<T> operator - (const Mat<T>& b) { return minus(b); }

	Mat<T> operator - (T b) { return minus(b); }

	friend Mat<T> operator - (T b, const Mat<T>& a) { return a.unaryOp([b] (T f) { return b - f; }); }

	Mat<T> operator * (T b) { return timesEach(b); }

	friend Mat<T> operator * (T b, const Mat<T>& a) { return a.timesEach(b); }

	Mat<T> operator / (T b) { return divideEach(b); }

	friend Mat<T> operator / (T b, const Mat<T>& a) { return a.unaryOp([b] (T f) {return b / f; }); }

	Mat<T>& operator += (const Mat<T>& b) { return setValues([&] (size_t row, size_t col) { return at(row, col) + b.at(row, col); }); }

	Mat<T>& operator += (T b) { return setValues([&] (size_t row, size_t col) { return at(row, col) + b; }); }

	Mat<T>& operator -= (const Mat<T>& b) { return setValues([&] (size_t row, size_t col) { return at(row, col) - b.at(row, col); }); }

	Mat<T>& operator -= (T b) { return setValues([&] (size_t row, size_t col) { return at(row, col) - b; }); }

	Mat<T>& operator /= (const Mat<T>& b) { return setValues([&] (size_t row, size_t col) { return at(row, col) / b.at(row, col); }); }

	Mat<T>& operator /= (T b) { return setValues([&] (size_t row, size_t col) { return at(row, col) / b; }); }

	Mat<T>& operator *= (const Mat<T>& b) { return setValues([&] (size_t row, size_t col) { return at(row, col) * b.at(row, col); }); }

	Mat<T>& operator *= (T b) { return setValues([&] (size_t row, size_t col) { return at(row, col) * b; }); }

	virtual MatRow<T> operator [] (size_t row) {
		assert(row < rows() && "row index out of bounds");
		return MatRow<T>(array + row * cols(), cols());
	}

	virtual MatRow<const T> operator [] (size_t row) const {
		assert(row < rows() && "row index out of bounds");
		return MatRow<const T>(array + row * cols(), cols());
	}

	//math operations

	Mat<T> trans() const { return generate(cols(), rows(), [&] (size_t r, size_t c) { return at(c, r); }); }

	Mat<T> plus(const Mat<T>& other) const { return unaryOpSafe(other, [&] (size_t r, size_t c) { return at(r, c) + other.at(r, c); }); }

	Mat<T> minus(const Mat<T>& other) const { return unaryOpSafe(other, [&] (size_t r, size_t c) { return at(r, c) - other.at(r, c); }); }

	Mat<T> timesEach(const Mat<T>& other) const { return unaryOpSafe(other, [&] (size_t r, size_t c) { return at(r, c) * other.at(r, c); }); }

	Mat<T> divideEach(const Mat<T>& other) const { return unaryOpSafe(other, [&] (size_t r, size_t c) { return at(r, c) / other.at(r, c); }); }

	Mat<T> plus(T value) const { return unaryOp([value] (T f) { return f + value; }); }

	Mat<T> minus(T value) const { return unaryOp([value] (T f) { return f - value; }); }

	Mat<T> timesEach(T value) const { return unaryOp([value] (T f) { return f * value; }); }

	Mat<T> divideEach(T value) const { return unaryOp([value] (T f) { return f / value; }); }

	Mat<T> unaryOp(const std::function<T(T)>& func) const {
		return generate(rows(), cols(), [&] (size_t r, size_t c) { return func(at(r, c)); });
	}

	Mat<T> filter(const Mat<T>& f, ThreadPoolBase& pool = defaultPool) const {
		auto func = [&] (size_t r, size_t c) {
			T sum = 0;
			size_t dx = f.cols() / 2;
			size_t dy = f.rows() / 2;
			for (size_t x = 0; x < f.cols(); x++) {
				size_t ix = clampUnsigned(x + c, dx, 0ull, cols() - 1);
				for (size_t y = 0; y < f.rows(); y++) {
					size_t iy = clampUnsigned(y + r, dy, 0ull, rows() - 1);
					sum += at(iy, ix) * f.at(y, x);
				}
			}
			return sum;
		};
		return generate(rows(), cols(), func, pool);
	}

	//filter mat in one dimension
	Mat<T>& filter1D(const T* kernel, size_t siz, Direction dir, Mat<T>& dest, ThreadPoolBase& pool = defaultPool) {
		//check matrix dimensions
		assert(cols() == dest.cols() && rows() == dest.rows() && "dimension mismatch");

		//delta values
		size_t dy = 1 - (size_t) dir;
		size_t dx = 1 - dy;

		//function to filter for one point
		auto func = [&] (size_t r, size_t c) {
			T sum = 0;
			for (int i = 0; i < siz; i++) {
				size_t x = clampUnsigned(c + dx * i, dx * siz / 2, 0ull, cols() - 1);
				size_t y = clampUnsigned(r + dy * i, dy * siz / 2, 0ull, rows() - 1);
				sum += at(y, x) * kernel[i];
			}
			return sum;
		};

		//apply function
		return dest.setArea(func, pool);
	}

	//evaluate this mat at points given by x and y mats respective, put results into dest
	template <class R> Mat<T>& resample(const Mat<R>& x, const Mat<R>& y, Mat<T>& dest) {
		assert(x.rows() == y.rows() && x.cols() == y.cols() && "dimensions of x and y must be equal");
		assert(x.rows() >= dest.rows() && x.cols() >= dest.cols() && "not enough points to resample");
		for (size_t r = 0; r < dest.rows(); r++) {
			for (size_t c = 0; c < dest.cols(); c++) {
				dest.at(r, c) = interp2(x.at(r, c), y.at(r, c)).value_or(0);
			}
		}
		return dest;
	}

	//matrix multiplication A * B
	Mat<T> times(const Mat<T>& other) const {
		assert(cols() == other.rows() && "dimensions do not agree");
		return generate(rows(), other.cols(), [&] (size_t r, size_t c) {
			T sum = (T) 0;
			for (size_t i = 0; i < cols(); i++) sum += at(r, i) * other.at(i, c);
			return sum;
			});
	}

	//matrix multiplication A * B where both matrices must be symmetric
	//only one value symmetric value is computed and copied
	Mat<T> timesSymmetric(const Mat<T>& other) const {
		assert(cols() == other.rows() && rows() == other.cols() && "dimensions do not agree");
		Mat<T> out = Mat<T>::allocate(rows(), cols());
		for (size_t r = 0; r < rows(); r++) {
			for (size_t c = r; c < cols(); c++) {
				T sum = (T) 0;
				for (size_t i = 0; i < rows(); i++) {
					sum += at(r, i) * other.at(i, c);
				}
				out.at(r, c) = out.at(c, r) = sum;
			}
		}
		return out;
	}

	//make matrix symmetric by taking average values
	Mat<T> averageSymmetric() const {
		assert(cols() == rows() && "dimensions do not agree");
		Mat<T> out = Mat<T>::allocate(rows(), cols());
		for (size_t r = 0; r < rows(); r++) {
			for (size_t c = r; c < cols(); c++) {
				out.at(r, c) = out.at(c, r) = (at(r, c) + at(c, r)) / 2.0;
			}
		}
		return out;
	}

	//multiplication A * A'
	Mat<T> timesTransposed() const {
		Mat<T> out = Mat<T>::zeros(rows(), rows());
		for (size_t r = 0; r < rows(); r++) {
			const T* a = addr(r, 0);
			for (size_t c = r; c < rows(); c++) {
				const T* b = addr(c, 0);
				T res = 0;
				for (size_t i = 0; i < cols(); i++) {
					res += a[i] * b[i];
				}
				out.at(c, r) = out.at(r, c) = res;
			}
		}
		return out;
	}

	//reshape all values into one column
	Mat<T> flatToCol() const {
		return reshape(this->numel(), 1);
	}

	Mat<T> reshape(size_t rows, size_t cols) const {
		assert(rows * cols == this->numel() && "new matrix must have same number of values");
		return generate(rows, cols, [this, rows] (size_t r, size_t c) {
			size_t idx = c * rows + r;
			return at(idx % this->rows(), idx / this->rows());
			});
	}

	Mat<T> repeat(size_t copiesVert, size_t copiesHorz) const {
		return generate(rows() * copiesVert, cols() * copiesHorz, [this] (size_t r, size_t c) { return at(r % rows(), c % cols()); });
	}

	//take part of this mat as a copy for a new mat
	Mat<T> subMat(size_t r0, size_t c0, size_t h, size_t w) const {
		return generate(h, w, [&] (size_t r, size_t c) {return at(r0 + r, c0 + c); });
	}
	
	//solve A * x = b using this mat for decomposition
	//using different decompositions based on format of mat
	std::optional<Mat<T>> solveInPlace(const Mat<T>& b) {
		if (rows() != b.rows()) return std::nullopt;
		if (rows() == cols()) return LUDecompositor<T>(*this).solve(b);
		else if (rows() > cols()) return QRDecompositor<T>(*this).solve(b);
		else return QRDecompositorUD<T>(*this).solve(b);
	}

	//solve A * x = b
	//using different decompositions based on format of mat
	std::optional<Mat<T>> solve(const Mat<T>& b) const {
		Mat A = *this;
		return A.solveInPlace(b);
	}

	//condition of matrix
	T cond() const {
		Mat A = *this;
		return SVDecompositor<T>(A).cond();
	}

	//calculate inverse
	std::optional<Mat<T>> inv() const {
		Mat<T> copy = *this;
		if (rows() == cols()) return LUDecompositor<T>(copy).inv();
		else return std::nullopt;
	}

	//calculate pseudo inverse
	Mat<T> pinv() const {
		Mat<T> copy = *this;
		return SVDecompositor<T>(copy).inv().value();
	}
};


template <class T> std::ostream& MatRow<T>::print(std::ostream& out) const {
	return Mat<T>::fromArray(1, cols, rowptr, false).print(out);
}


using Matd = Mat<double>;
using Matf = Mat<float>;
using Matc = Mat<unsigned char>;


template<> constexpr double Matd::eps() { return 1e-12; }
template<> constexpr float Matf::eps() { return 1e-5f; }
