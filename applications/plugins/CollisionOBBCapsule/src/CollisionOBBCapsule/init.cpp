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

namespace sofa
{

namespace component
{

void initSofaMiscCollision()
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
    initSofaMiscCollision();
}

const char* getModuleName()
{
    return "CollisionOBB";
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
    return "This plugin contains OBB collision components.";
}

const char* getModuleComponentList()
{
    return "DistanceGridCollisionModel FFDDistanceGridDiscreteIntersection RayDistanceGridContact "
           "RigidDistanceGridDiscreteIntersection DistanceGridForceField";
}

} // component

} // sofa
