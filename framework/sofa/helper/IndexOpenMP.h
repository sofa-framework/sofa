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
#ifndef SOFA_HELPER_INDEX_TYPE_H
#define SOFA_HELPER_INDEX_TYPE_H

#include <cassert>
#include <iostream>
#include <type_traits>
#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

template<class T>
struct SOFA_HELPER_API IndexOpenMP
{
#if defined(USING_OMP_PRAGMAS) && defined(WIN32)
		typedef typename std::make_signed<T>::type type;
#else
		typedef T type;
#endif
};// struct IndexOpenMP

} // helper

} //
#endif SOFA_HELPER_INDEX_TYPE_H