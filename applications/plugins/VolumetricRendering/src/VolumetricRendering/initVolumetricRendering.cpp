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
#include <VolumetricRendering/initVolumetricRendering.h>
#include <sofa/core/ObjectFactory.h>

namespace volumetricrendering
{

//Here are just several convenient functions to help user to know what contains the plugin

extern "C" {
    SOFA_VOLUMETRICRENDERING_API void initExternalModule();
    SOFA_VOLUMETRICRENDERING_API const char* getModuleName();
    SOFA_VOLUMETRICRENDERING_API const char* getModuleVersion();
    SOFA_VOLUMETRICRENDERING_API const char* getModuleLicense();
    SOFA_VOLUMETRICRENDERING_API const char* getModuleDescription();
    SOFA_VOLUMETRICRENDERING_API const char* getModuleComponentList();
}

void init()
{
    initExternalModule();
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
    return volumetricrendering::MODULE_NAME;
}

const char* getModuleVersion()
{
    return volumetricrendering::MODULE_VERSION;
}

const char* getModuleLicense()
{
    return "LGPL";
}


const char* getModuleDescription()
{
    return "A plugin for volumetric rendering (tetrahedron, hexahedron)";
}

const char* getModuleComponentList()
{
    static std::string classes = sofa::core::ObjectFactory::getInstance()->listClassesFromTarget(volumetricrendering::MODULE_NAME);
    return classes.c_str();
}


} // namespace volumetricrendering
