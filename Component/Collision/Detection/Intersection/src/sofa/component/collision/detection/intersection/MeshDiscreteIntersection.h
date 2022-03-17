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

#include <sofa/component/collision/model/LineModel.h>
#include <sofa/component/collision/model/TriangleModel.h>
#include <sofa/component/collision/model/SphereModel.h>

namespace sofa::component::collision::detection::intersection
{

class DiscreteIntersection;

class SOFA_COMPONENT_COLLISION_DETECTION_INTERSECTION_API MeshDiscreteIntersection : public core::collision::BaseIntersector
{
    typedef core::collision::BaseIntersector::OutputVector OutputVector;

public:
    MeshDiscreteIntersection(DiscreteIntersection* object, bool addSelf=true);

    bool testIntersection(model::Triangle&, model::Line&);
    template<class T> bool testIntersection(model::TSphere<T>&, model::Triangle&);

    int computeIntersection(model::Triangle& e1, model::Line& e2, OutputVector* contacts);
    template<class T> int computeIntersection(model::TSphere<T>&, model::Triangle&, OutputVector*);

protected:
    DiscreteIntersection* intersection;

};

} // namespace sofa::component::collision::detection::intersection
