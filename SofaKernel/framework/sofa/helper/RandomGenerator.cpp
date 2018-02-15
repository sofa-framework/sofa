/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*      Initialization and __dorand48 functions has been taken from OpenBSD, srand48 & _rand48 */
/*
 * Copyright (c) 1993 Martin Birgmeier
 * All rights reserved.
 *
 * You may redistribute unmodified or modified versions of this source
 * code provided that the above copyright notice and this and the
 * following conditions are retained.
 *
 * This software is provided ``as is'', and comes with no warranties
 * of any kind. I shall in no event be liable for anything that happens
 * to anyone/anything when using this software.
 */


#include "RandomGenerator.h"

#include <cmath>


namespace sofa
{

namespace helper
{

const unsigned long RandomGenerator::RANDOM_BASE_MAX = 4294967295U;

RandomGenerator::RandomGenerator()
{
    __rand48_seed[0] = RAND48_SEED_0;
    __rand48_seed[1] = RAND48_SEED_1;
    __rand48_seed[2] = RAND48_SEED_2;

    __rand48_mult[0] = RAND48_MULT_0;
    __rand48_mult[1] = RAND48_MULT_1;
    __rand48_mult[2] = RAND48_MULT_2;

    __rand48_add = RAND48_ADD;
}

RandomGenerator::RandomGenerator(long seed)
    : seed(seed)
{
    initSeed(seed);
}

RandomGenerator::~RandomGenerator()
{
}

void RandomGenerator::initSeed(long seed)
{
    this->seed = seed;

    __rand48_seed[0] = RAND48_SEED_0;
    __rand48_seed[1] = (unsigned short) seed;
    __rand48_seed[2] = (unsigned short) (seed >> 16);
    __rand48_mult[0] = RAND48_MULT_0;
    __rand48_mult[1] = RAND48_MULT_1;
    __rand48_mult[2] = RAND48_MULT_2;
    __rand48_add = RAND48_ADD;
}

void RandomGenerator::__dorand48(unsigned short xseed[3])
{
    unsigned long accu;
    unsigned short temp[2];

    accu = (unsigned long) __rand48_mult[0] * (unsigned long) xseed[0] +
            (unsigned long) __rand48_add;
    temp[0] = (unsigned short) accu;        /* lower 16 bits */
    accu >>= sizeof(unsigned short) * 8;
    accu += (unsigned long) __rand48_mult[0] * (unsigned long) xseed[1] +
            (unsigned long) __rand48_mult[1] * (unsigned long) xseed[0];
    temp[1] = (unsigned short) accu;        /* middle 16 bits */
    accu >>= sizeof(unsigned short) * 8;
    accu += __rand48_mult[0] * xseed[2] + __rand48_mult[1] * xseed[1] + __rand48_mult[2] * xseed[0];
    xseed[0] = temp[0];
    xseed[1] = temp[1];
    xseed[2] = (unsigned short) accu;
}

unsigned long RandomGenerator::randomBase()
{
    __dorand48(__rand48_seed);
    return ((unsigned long) __rand48_seed[2] << 16) + (unsigned long) __rand48_seed[1];
}



}

}
