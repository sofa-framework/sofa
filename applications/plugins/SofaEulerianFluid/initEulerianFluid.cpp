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
#include <sofa/helper/system/config.h>
#include <SofaEulerianFluid/initEulerianFluid.h>


namespace sofa
{

namespace component
{

/// Convenient functions to help user to know what contains the plugin
extern "C" {
    SOFA_EULERIAN_FLUID_API void initExternalModule();
    SOFA_EULERIAN_FLUID_API const char* getModuleName();
    SOFA_EULERIAN_FLUID_API const char* getModuleVersion();
    SOFA_EULERIAN_FLUID_API const char* getModuleLicense();
    SOFA_EULERIAN_FLUID_API const char* getModuleDescription();
    SOFA_EULERIAN_FLUID_API const char* getModuleComponentList();
}

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

const char* getModuleName()
{
    return "SofaEulerianFluid";
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
    return "This plugin expose component implementing fluid simulation in 2D and 3D.";
}

const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    return "Fluid2D Fluid3D";
}


SOFA_LINK_CLASS(Fluid2D)
SOFA_LINK_CLASS(Fluid3D)

} /// namespace component

} /// namespace sofa
