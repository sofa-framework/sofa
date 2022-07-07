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
#include <CollisionOBBCapsule/init.h>

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory;

namespace collisionobbcapsule
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
COLLISIONOBBCAPSULE_API void initExternalModule();
COLLISIONOBBCAPSULE_API const char* getModuleName();
COLLISIONOBBCAPSULE_API const char* getModuleVersion();
COLLISIONOBBCAPSULE_API const char* getModuleLicense();
COLLISIONOBBCAPSULE_API const char* getModuleDescription();
COLLISIONOBBCAPSULE_API const char* getModuleComponentList();
}

void initExternalModule()
{
    init();
}

const char* getModuleName()
{
    return sofa_tostring(SOFA_TARGET);
}

const char* getModuleVersion()
{
    return sofa_tostring(COLLISIONOBBCAPSULE_VERSION);
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "This plugin contains OBB and capsule collision components.";
}

const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    static std::string classes = ObjectFactory::getInstance()->listClassesFromTarget(sofa_tostring(SOFA_TARGET));
    return classes.c_str();
}

} // namespace collisionobbcapsule
