/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/testing/LinearCongruentialRandomGenerator.h>

namespace sofa::testing
{

LinearCongruentialRandomGenerator::LinearCongruentialRandomGenerator(const unsigned int initialSeed)
: m_seed(initialSeed)
{}

unsigned LinearCongruentialRandomGenerator::generateRandom()
{
    // Parameters for the LCG formula (adjust as needed)
    constexpr unsigned int a = 1664525;
    constexpr unsigned int c = 1013904223;

    m_seed = a * m_seed + c; // LCG formula
    return m_seed;
}

double LinearCongruentialRandomGenerator::generateInRange(const double rmin, const double rmax)
{
    return rmin + generateInUnitRange<double>() * (rmax - rmin);
}

float LinearCongruentialRandomGenerator::generateInRange(const float rmin, const float rmax)
{
    return rmin + generateInUnitRange<float>() * (rmax - rmin);
}

}
