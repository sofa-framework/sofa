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
#ifndef INITLEAPMOTION_H
#define INITLEAPMOTION_H


#include <sofa/helper/system/config.h>


#ifndef WIN32
        #define SOFA_EXPORT_DYNAMIC_LIBRARY
        #define SOFA_IMPORT_DYNAMIC_LIBRARY
        #define SOFA_LEAPMOTION_API
#else
        #ifdef SOFA_BUILD_LEAPMOTION
                #define SOFA_EXPORT_DYNAMIC_LIBRARY __declspec( dllexport )
                #define SOFA_LEAPMOTION_API SOFA_EXPORT_DYNAMIC_LIBRARY
        #else
                #define SOFA_IMPORT_DYNAMIC_LIBRARY __declspec( dllimport )
                #define SOFA_LEAPMOTION_API SOFA_IMPORT_DYNAMIC_LIBRARY
        #endif
#endif // not WIN32

#endif // INITLEAPMOTION_H
