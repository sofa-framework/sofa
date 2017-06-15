/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <BulletCollisionDetection/config.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

//Here are just several convenient functions to help user to know what contains the plugin

extern "C" {
SOFA_BULLETCOLLISIONDETECTION_API void initExternalModule();
SOFA_BULLETCOLLISIONDETECTION_API const char* getModuleName();
SOFA_BULLETCOLLISIONDETECTION_API const char* getModuleVersion();
SOFA_BULLETCOLLISIONDETECTION_API const char* getModuleLicense();
SOFA_BULLETCOLLISIONDETECTION_API const char* getModuleDescription();
SOFA_BULLETCOLLISIONDETECTION_API const char* getModuleComponentList();
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
    return "BulletCollisionDetection";
}

const char* getModuleVersion()
{
    return "1.0";
}

const char* getModuleLicense()
{
    return "Private";
}

const char* getModuleDescription()
{
    return "Bullet Collision Detection";
}

const char* getModuleComponentList()
{
    /// string containing the names of the classes provided by the plugin
    static std::string classes = sofa::core::ObjectFactory::getInstance()->listClassesFromTarget(sofa_tostring(SOFA_TARGET));
    return classes.c_str();
}

}

}
