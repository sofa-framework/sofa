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
#pragma once

// execution policies only supported by MSVC >=2019 and GCC >=10
#ifdef _MSC_VER
    #define HAS_SUPPORT_STL_PARALLELISM (_MSC_VER > 1921)
#elif defined(__GNUC__)
    #define HAS_SUPPORT_STL_PARALLELISM  (__GNUC__ > 9)
#elif defined(__clang_major__)
    #define HAS_SUPPORT_STL_PARALLELISM  (__clang_major__ > 10)
#else
    #define HAS_SUPPORT_STL_PARALLELISM  false
#endif

#if HAS_SUPPORT_STL_PARALLELISM
#include <execution>

#define SOFA_STD_EXECUTION_SEQ       std::execution::seq,
#define SOFA_STD_EXECUTION_PAR       std::execution::par,
#define SOFA_STD_EXECUTION_PAR_UNSEQ std::execution::par_unseq,
#define SOFA_STD_EXECUTION_UNSEQ     std::execution::unseq,

#else

#define SOFA_STD_EXECUTION_SEQ
#define SOFA_STD_EXECUTION_PAR
#define SOFA_STD_EXECUTION_PAR_UNSEQ
#define SOFA_STD_EXECUTION_UNSEQ

#endif
