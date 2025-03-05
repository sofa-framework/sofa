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
#include <SofaCarving/init.h>

namespace sofacarving
{
//Here are just several convenient functions to help user to know what contains the plugin

extern "C" {
    SOFA_SOFACARVING_API void initExternalModule();
    SOFA_SOFACARVING_API const char* getModuleName();
    SOFA_SOFACARVING_API const char* getModuleVersion();
    SOFA_SOFACARVING_API const char* getModuleLicense();
    SOFA_SOFACARVING_API const char* getModuleDescription();
    SOFA_SOFACARVING_API const char* getModuleComponentList();
}

void initExternalModule()
{
    init();
}

const char* getModuleName()
{
    return "SofaCarving";
}

const char* getModuleVersion()
{
    return "0.3";
}

const char* getModuleLicense()
{
    return "LGPL";
}


const char* getModuleDescription()
{
    return "Manager handling carving operations between a tool and an object.";
}

const char* getModuleComponentList()
{
    return "CarvingManager";
}

void init()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

} // namespace sofacarving
