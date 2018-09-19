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
#include "config.h"

#include "PythonEnvironment.h"
using sofapython3::PythonEnvironment;

extern "C" {

SOFAPYTHON3_API void initExternalModule();
SOFAPYTHON3_API const char* getModuleName();
SOFAPYTHON3_API const char* getModuleVersion();
SOFAPYTHON3_API const char* getModuleLicense();
SOFAPYTHON3_API const char* getModuleDescription();
SOFAPYTHON3_API const char* getModuleComponentList();

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        PythonEnvironment::Init();
        first = false;
    }
}

const char* getModuleName()
{
    return "SofaPython3";
}

const char* getModuleVersion()
{
    return "1.0";
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "This plugin contains the interpreter for python3.";
}

const char* getModuleComponentList()
{
    return "";
}

}
