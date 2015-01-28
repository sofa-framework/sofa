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
#ifndef SOFA_HELPER_RANDOM_H
#define SOFA_HELPER_RANDOM_H

#include <stdlib.h>

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
   return (std::rand());
}

// Random double between 0 and 1 [0.0,1.0]
inline double drand()
{
   return (std::rand()* 1.0/RAND_MAX);
}

// Random double between -max and max [-max,max]
inline double drand(double max)
{
   return ( ((std::rand()*1.0/RAND_MAX) * 2 * max) - max );
}

} // namespace helper

} // namespace sofa

#endif // SOFA_HELPER_RANDOM_H
