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
#ifndef SOFA_COMPONENT_COLLISION_FFDDISTANCEGRIDDISCRETEINTERSECTION_H
#define SOFA_COMPONENT_COLLISION_FFDDISTANCEGRIDDISCRETEINTERSECTION_H
#include <SofaDistanceGrid/config.h>

#include <sofa/helper/FnDispatcher.h>
#include <sofa/core/collision/Intersection.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaMeshCollision/PointModel.h>
#include <SofaMeshCollision/LineModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaUserInteraction/RayModel.h>
#include <SofaBaseCollision/DiscreteIntersection.h>

#include <SofaDistanceGrid/components/collision/DistanceGridCollisionModel.h>

namespace sofa
{

namespace component
{

namespace collision
{
class SOFA_SOFADISTANCEGRID_API FFDDistanceGridDiscreteIntersection : public core::collision::BaseIntersector
{

    typedef DiscreteIntersection::OutputVector OutputVector;

public:
    FFDDistanceGridDiscreteIntersection(DiscreteIntersection* object);

    bool testIntersection(FFDDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&);
    bool testIntersection(FFDDistanceGridCollisionElement&, FFDDistanceGridCollisionElement&);
    bool testIntersection(FFDDistanceGridCollisionElement&, Point&);
    template<class T> bool testIntersection(FFDDistanceGridCollisionElement&, TSphere<T>&);
    bool testIntersection(FFDDistanceGridCollisionElement&, Triangle&);
    bool testIntersection(Ray&, FFDDistanceGridCollisionElement&);

    int computeIntersection(FFDDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&, OutputVector*);
    int computeIntersection(FFDDistanceGridCollisionElement&, FFDDistanceGridCollisionElement&, OutputVector*);
    int computeIntersection(FFDDistanceGridCollisionElement&, Point&, OutputVector*);
    template<class T> int computeIntersection(FFDDistanceGridCollisionElement&, TSphere<T>&, OutputVector*);
    int computeIntersection(FFDDistanceGridCollisionElement&, Triangle&, OutputVector*);
    int computeIntersection(Ray&, FFDDistanceGridCollisionElement&, OutputVector*);

protected:

    DiscreteIntersection* intersection;

};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
