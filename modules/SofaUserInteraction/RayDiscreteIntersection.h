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
#ifndef SOFA_COMPONENT_COLLISION_RAYDISCRETEINTERSECTION_H
#define SOFA_COMPONENT_COLLISION_RAYDISCRETEINTERSECTION_H
#include "config.h"

#include <sofa/core/collision/Intersection.h>
#include <sofa/helper/FnDispatcher.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaMeshCollision/PointModel.h>
#include <SofaMeshCollision/LineModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaUserInteraction/RayModel.h>
#include <SofaBaseCollision/DiscreteIntersection.h>

namespace sofa
{

namespace component
{

namespace collision
{
class SOFA_USER_INTERACTION_API RayDiscreteIntersection : public core::collision::BaseIntersector
{

    typedef DiscreteIntersection::OutputVector OutputVector;

public:
    RayDiscreteIntersection(DiscreteIntersection* object, bool addSelf=true);

    template<class T> bool testIntersection(Ray&, TSphere<T>&);
    bool testIntersection(Ray&, Triangle&);

    template<class T> int computeIntersection(Ray&, TSphere<T>&, OutputVector*);
    int computeIntersection(Ray&, Triangle&, OutputVector*);

    bool testIntersection(Ray& rRay, OBB& rOBB);
    int computeIntersection(Ray& rRay, OBB& rOBB, OutputVector*);

protected:

    DiscreteIntersection* intersection;

};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
