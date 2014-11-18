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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_RAYNEWPROXIMITYINTERSECTION_H
#define SOFA_COMPONENT_COLLISION_RAYNEWPROXIMITYINTERSECTION_H

#include <sofa/SofaGeneral.h>
#include <SofaBaseCollision/NewProximityIntersection.h>
#include <sofa/helper/FnDispatcher.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaMeshCollision/LineModel.h>
#include <SofaMeshCollision/PointModel.h>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaUserInteraction/RayModel.h>
#include <SofaBaseCollision/OBBModel.h>

namespace sofa
{

namespace component
{

namespace collision
{

class SOFA_USER_INTERACTION_API RayNewProximityIntersection : public core::collision::BaseIntersector
{
    typedef NewProximityIntersection::OutputVector OutputVector;

public:
    RayNewProximityIntersection(NewProximityIntersection* object, bool addSelf=true);

    bool testIntersection(Ray& t1, Triangle& t2);
    bool testIntersection(Ray& rRay, OBB& rOBB);
    bool testIntersection(Ray& rRay, RigidSphere& rSphere);

    int computeIntersection(Ray& t1, Triangle& t2, OutputVector*);
    int computeIntersection(Ray& rRay, OBB& rOBB, OutputVector*);
    int computeIntersection(Ray& rRay, RigidSphere& rSphere, OutputVector*);


protected:

    NewProximityIntersection* intersection;
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
