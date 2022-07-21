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
#include <SofaMiscCollision/initSofaMiscCollision.h>

#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace component
{

void initSofaMiscCollision()
{
    static bool first = true;
    if (first)
    {
        msg_deprecated("SofaMiscCollision") << "SofaMiscCollision is deprecated. It will be removed at v23.06. Use Sofa.Component.Collision.Geometry, Sofa.Component.Collision.Detection.Intersection and Sofa.Component.Collision.Response.Contact instead.";
        msg_deprecated("SofaMiscCollision") << "If you are looking for OBB and Capsule-related components, please use the external plugin CollisionOBBCapsule.";

        sofa::helper::system::PluginManager::getInstance().loadPlugin("Sofa.Component.Collision.Geometry");
        sofa::helper::system::PluginManager::getInstance().loadPlugin("Sofa.Component.Collision.Detection.Intersection");
        sofa::helper::system::PluginManager::getInstance().loadPlugin("Sofa.Component.Collision.Response.Contact");

        first = false;
    }
}

extern "C" {
SOFA_MISC_COLLISION_API void initExternalModule();
SOFA_MISC_COLLISION_API const char* getModuleName();
SOFA_MISC_COLLISION_API const char* getModuleVersion();
SOFA_MISC_COLLISION_API const char* getModuleLicense();
SOFA_MISC_COLLISION_API const char* getModuleDescription();
SOFA_MISC_COLLISION_API const char* getModuleComponentList();
}

void initExternalModule()
{
    initSofaMiscCollision();
}

const char* getModuleName()
{
    return "SofaMiscCollision";
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
    return "This plugin contains collision components.";
}

const char* getModuleComponentList()
{
    return "DistanceGridCollisionModel FFDDistanceGridDiscreteIntersection RayDistanceGridContact "
           "RigidDistanceGridDiscreteIntersection DistanceGridForceField";
}

} // component

} // sofa
