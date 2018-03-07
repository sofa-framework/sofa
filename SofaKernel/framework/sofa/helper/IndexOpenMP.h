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
#ifndef SOFA_HELPER_INDEX_TYPE_H
#define SOFA_HELPER_INDEX_TYPE_H

namespace sofa
{

namespace helper
{

/// From any given index type, this struct gives a OpenMP valid index
/// Versions of OpenMP anterior to 3.0 are only able to manage signed index in a parallel for
/// Actual versions of visual studio only implement OpenMP 2.5 as old gcc.
template<class T>
struct IndexOpenMP
{
#if defined(_OPENMP) && _OPENMP < 200805 /*yearmonth of version 3.0*/
		typedef typename std::make_signed<T>::type type;
#else
		typedef T type;
#endif
}; // struct IndexOpenMP

} // helper

} // sofa

#endif // SOFA_HELPER_INDEX_TYPE_H
