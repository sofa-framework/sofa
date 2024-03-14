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
#include <sofa/component/linearsystem/init.h>

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory;

namespace sofa::component::linearsystem
{

void init()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

extern "C" {
    SOFA_COMPONENT_LINEARSYSTEM_API void initExternalModule();
    SOFA_COMPONENT_LINEARSYSTEM_API const char* getModuleName();
    SOFA_COMPONENT_LINEARSYSTEM_API const char* getModuleVersion();
    SOFA_COMPONENT_LINEARSYSTEM_API const char* getModuleLicense();
    SOFA_COMPONENT_LINEARSYSTEM_API const char* getModuleDescription();
    SOFA_COMPONENT_LINEARSYSTEM_API const char* getModuleComponentList();
}

void initExternalModule()
{
    init();
}

const char* getModuleName()
{
    return MODULE_NAME;
}

const char* getModuleVersion()
{
    return MODULE_VERSION;
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "This plugin defines a linear system and provides components able to assemble one from the scene.";
}

const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    static std::string classes = ObjectFactory::getInstance()->listClassesFromTarget(MODULE_NAME);
    return classes.c_str();
}

} // namespace sofa::component::linearsystem
