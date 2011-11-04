/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*
 * RandomGenerator.h
 *
 *  Created on: 25 mai 2009
 *      Author: froy
 */

#ifndef RANDOMGENERATOR_H_
#define RANDOMGENERATOR_H_

#include <cstdlib>
#include <sofa/helper/helper.h>

#define RAND48_SEED_0   (0x330e)
#define RAND48_SEED_1   (0xabcd)
#define RAND48_SEED_2   (0x1234)
#define RAND48_MULT_0   (0xe66d)
#define RAND48_MULT_1   (0xdeec)
#define RAND48_MULT_2   (0x0005)
#define RAND48_ADD      (0x000b)

namespace sofa
{

namespace helper
{


//Generate Random number
//based on random functions from OpenBSD

class SOFA_HELPER_API RandomGenerator
{
private:
    unsigned short __rand48_seed[3];
    unsigned short __rand48_mult[3];
    unsigned short __rand48_add;

    long seed;

    void __dorand48(unsigned short xseed[3]);

public:
    RandomGenerator();
    RandomGenerator(long seed);
    virtual ~RandomGenerator();

    void initSeed(long seed);

    //integer between [0, 2^32)
    unsigned long int random();
    //integer between [min, max)
    long int randomInteger(long min, long max);
    //float number between [min, max)
    double randomDouble(double min, double max);
};

}

}

#endif /* RANDOMGENERATOR_H_ */
