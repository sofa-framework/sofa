/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef INITSOFAPYTHON_H
#define INITSOFAPYTHON_H

#include <sofa/SofaPython.h>


namespace sofa
{

namespace component
{

extern "C" {
    SOFA_SOFAPYTHON_API void initExternalModule();
    SOFA_SOFAPYTHON_API const char* getModuleName();
    SOFA_SOFAPYTHON_API const char* getModuleVersion();
    SOFA_SOFAPYTHON_API const char* getModuleLicense();
    SOFA_SOFAPYTHON_API const char* getModuleDescription();
    SOFA_SOFAPYTHON_API const char* getModuleComponentList();
}

}

}

/** \mainpage
  This is a the starting page of the plugin documentation, defined in file initEmptyPlugin.h
  */

#endif // INITEmptyPlugin_H
