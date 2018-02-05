/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_COLLISION_RAYNEWPROXIMITYINTERSECTION_H
#define SOFA_COMPONENT_COLLISION_RAYNEWPROXIMITYINTERSECTION_H
#include "config.h"

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
    int computeIntersection(Ray& t1, Triangle& t2, OutputVector*);

    // why rigidsphere has a different collision detection compared to RayDiscreteIntersection?
    bool testIntersection(Ray& rRay, RigidSphere& rSphere);
    int computeIntersection(Ray& rRay, RigidSphere& rSphere, OutputVector*);


protected:

    NewProximityIntersection* intersection;
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
