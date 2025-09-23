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
#pragma once

#include <sofa/component/collision/detection/intersection/config.h>

#include <sofa/core/collision/Intersection.h>

#include <sofa/component/collision/geometry/LineModel.h>
#include <sofa/component/collision/geometry/TriangleModel.h>
#include <sofa/component/collision/geometry/SphereModel.h>

namespace sofa::component::collision::detection::intersection
{

class DiscreteIntersection;

class SOFA_COMPONENT_COLLISION_DETECTION_INTERSECTION_API MeshDiscreteIntersection : public core::collision::BaseIntersector
{
    typedef core::collision::BaseIntersector::OutputVector OutputVector;

public:
    MeshDiscreteIntersection(DiscreteIntersection* intersection, bool addSelf=true);
    
    template<class T> bool testIntersection(collision::geometry::TSphere<T>&, collision::geometry::Triangle&, const core::collision::Intersection* currentIntersection);
    template<class T> int computeIntersection(collision::geometry::TSphere<T>&, collision::geometry::Triangle&, OutputVector*, const core::collision::Intersection* currentIntersection);
    bool testIntersection(collision::geometry::Triangle&, collision::geometry::Line&, const core::collision::Intersection* currentIntersection);
    int computeIntersection(collision::geometry::Triangle& e1, collision::geometry::Line& e2, OutputVector* contacts, const core::collision::Intersection* currentIntersection);
};

} // namespace sofa::component::collision::detection::intersection
