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
#ifndef SOFA_COMPONENT_COLLISION_RIGIDDISTANCEGRIDDISCRETEINTERSECTION_H
#define SOFA_COMPONENT_COLLISION_RIGIDDISTANCEGRIDDISCRETEINTERSECTION_H
#include "config.h"

#include <sofa/core/collision/Intersection.h>
#include <sofa/helper/FnDispatcher.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaMeshCollision/PointModel.h>
#include <SofaMeshCollision/LineModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaUserInteraction/RayModel.h>
#include <SofaVolumetricData/DistanceGridCollisionModel.h>
#include <SofaBaseCollision/DiscreteIntersection.h>

namespace sofa
{

namespace component
{

namespace collision
{
class SOFA_VOLUMETRIC_DATA_API RigidDistanceGridDiscreteIntersection : public core::collision::BaseIntersector
{

    typedef DiscreteIntersection::OutputVector OutputVector;

public:
    RigidDistanceGridDiscreteIntersection(DiscreteIntersection* object);

    bool testIntersection(RigidDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&);
    bool testIntersection(RigidDistanceGridCollisionElement&, Point&);
    template<class T> bool testIntersection(RigidDistanceGridCollisionElement&, TSphere<T>&);
    bool testIntersection(RigidDistanceGridCollisionElement&, Line&);
    bool testIntersection(RigidDistanceGridCollisionElement&, Triangle&);
    bool testIntersection(Ray&, RigidDistanceGridCollisionElement&);

    int computeIntersection(RigidDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&, OutputVector*);
    int computeIntersection(RigidDistanceGridCollisionElement&, Point&, OutputVector*);
    template<class T> int computeIntersection(RigidDistanceGridCollisionElement&, TSphere<T>&, OutputVector*);
    int computeIntersection(RigidDistanceGridCollisionElement&, Line&, OutputVector*);
    int computeIntersection(RigidDistanceGridCollisionElement&, Triangle&, OutputVector*);
    int computeIntersection(Ray&, RigidDistanceGridCollisionElement&, OutputVector*);

protected:

    DiscreteIntersection* intersection;

};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
