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

#include <random>

//superclass for random generators
class RNGbase {

public:
    virtual ~RNGbase() = default;
    typedef uint32_t result_type;             //type has to be hardcoded in superclass?
    virtual result_type operator() () = 0;    //will be overridden in subclass
    inline static result_type (*min) ();      //function pointer will be set in subclass constructor
    inline static result_type (*max) ();      //function pointer will be set in subclass constructor
};

//wrapper for random generators
template <class R> class RNG : public RNGbase {

public:
    using result_type = typename R::result_type;

    R r;

    RNG() {
        RNGbase::min = R::min;
        RNGbase::max = R::max;
    }

    result_type operator() () override {
        return r();
    }
};


//simple random number generator
class RandomSource {

public:
	typedef uint32_t result_type;

	static result_type min() {
		return 0;
	}

	static result_type max() {
		return 65'535;
	}

	result_type operator () ();

	RandomSource& operator = (const RandomSource& other) = delete;

private:
	int idx1 = 0;
	int idx2 = 0;

    const uint16_t rngData[256] = {
        117,    62,   144,   166,     8,   204,    55,   201,
        93,     10,   157,   253,   187,   225,    82,    59,
        168,   134,    68,   181,   241,   100,    44,   152,
        238,   129,    72,   153,   249,   227,    81,   147,
        26,     63,    66,   196,   150,    73,    20,   219,
        34,    109,    56,    83,    40,   136,    37,   126,
        16,      5,    64,    36,   149,   203,   118,   108,
        255,    95,   223,   156,    50,   250,   186,    12,
        169,    48,   123,    22,    57,   182,    88,    79,
        28,     99,   226,   252,   234,   195,   183,   216,
        146,   170,   236,   122,   191,    42,   155,   239,
        86,    212,   101,   139,   192,   173,    13,    69,
        164,    24,   128,   158,   138,   171,    96,   185,
        85,    127,    45,   224,   247,   102,   194,    18,
        140,   180,    27,   145,    98,    47,    38,    89,
        113,   143,     1,    11,    87,    23,   124,   200,
        184,   218,    80,   233,   198,    15,     9,   161,
        106,    39,   176,   217,   111,    19,   237,   148,
        121,   178,   231,   125,   246,   251,   197,    33,
        235,   210,   206,   214,     2,    46,    67,   162,
        159,    71,   190,   188,   175,   179,   119,     7,
        29,    205,   215,    35,    61,   103,   193,     3,
        0,     229,    97,   221,    78,    31,   167,    60,
        115,    54,    32,   114,   133,    58,    30,    14,
        222,   211,   232,    41,   202,    25,   110,    51,
        154,   213,    94,   189,    84,    43,    76,   116,
        230,   107,   151,   105,    91,   228,    65,   245,
        49,    208,    52,   174,   163,   240,   209,     4,
        142,   248,    70,   244,   242,   172,    17,     6,
        90,    165,   199,    92,   132,    21,   135,   220,
        160,   243,   137,   207,    77,   177,   120,   254,
        130,   141,    75,    74,    53,   104,   131,   112,
    };

};