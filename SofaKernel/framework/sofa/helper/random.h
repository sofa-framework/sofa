/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_HELPER_RANDOM_H
#define SOFA_HELPER_RANDOM_H

#include <cstdlib>

namespace sofa
{

namespace helper
{

/** Define random methods srand and rand for double and int used only for the tests now.
*/

// Init seed
inline void srand(unsigned seed)
{
    std::srand(seed);
}

// Random integer value between 0 and RAND_MAX
inline int irand()
{
   return std::rand();
}

// Random double between 0 and 1 [0.0,1.0]
inline double drand()
{
   return double(std::rand()) / RAND_MAX;
}

// Random double between 0 and max [0.0,max]
// max must be positive
inline double drandpos(double max)
{
   assert( max >= 0 );
   return drand() * max;
}

// Random double between -max and max [-max,max]
// max must be positive
inline double drand(double max)
{
   return drandpos(max) * 2.0 - max;
}

// Random double in [-max,-min] U [min,max]
// min and max must be positve
inline double drand(double min, double max)
{
   assert( max >= 0 );
   assert( min >= 0 );
   assert( min <= max );

   double tmp = drand(max-min); // [ -(max-min), max-min ]
   return tmp < 0 ? tmp-min : tmp+min;
}


} // namespace helper

} // namespace sofa

#endif // SOFA_HELPER_RANDOM_H
