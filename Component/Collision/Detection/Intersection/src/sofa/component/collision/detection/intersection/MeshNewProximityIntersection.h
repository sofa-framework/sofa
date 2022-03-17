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

#include <sofa/component/collision/detection/intersection/NewProximityIntersection.h>
#include <sofa/component/collision/model/TriangleModel.h>
#include <sofa/component/collision/model/LineModel.h>
#include <sofa/component/collision/model/PointModel.h>

namespace sofa::component::collision::detection::intersection
{

class SOFA_COMPONENT_COLLISION_DETECTION_INTERSECTION_API MeshNewProximityIntersection : public core::collision::BaseIntersector
{
    typedef NewProximityIntersection::OutputVector OutputVector;

public:
    MeshNewProximityIntersection(NewProximityIntersection* object, bool addSelf=true);

    bool testIntersection(model::Point&, model::Point&);
    int computeIntersection(model::Point&, model::Point&, OutputVector*);
    bool testIntersection(model::Line&, model::Point&);
    int computeIntersection(model::Line&, model::Point&, OutputVector*);
    bool testIntersection(model::Line&, model::Line&);
    int computeIntersection(model::Line&, model::Line&, OutputVector*);
    bool testIntersection(model::Triangle&, model::Point&);
    int computeIntersection(model::Triangle&, model::Point&, OutputVector*);
    bool testIntersection(model::Triangle&, model::Line&);
    int computeIntersection(model::Triangle&, model::Line&, OutputVector*);
    bool testIntersection(model::Triangle&, model::Triangle&);
    int computeIntersection(model::Triangle&, model::Triangle&, OutputVector*);

    template <class T>
    bool testIntersection(model::TSphere<T>& sph, model::Point& pt);
    template <class T> 
    int computeIntersection(model::TSphere<T>& sph, model::Point& pt, OutputVector*);
    template <class T>
    bool testIntersection(model::Line&, model::TSphere<T>&);
    template <class T> 
    int computeIntersection(model::Line& line, model::TSphere<T>& sph, OutputVector*);
    template <class T>
    bool testIntersection(model::Triangle&, model::TSphere<T>&);
    template <class T> 
    int computeIntersection(model::Triangle& tri, model::TSphere<T>& sph, OutputVector*);

    static inline int doIntersectionLineLine(SReal dist2, const type::Vector3& p1, const type::Vector3& p2, const type::Vector3& q1, const type::Vector3& q2, OutputVector* contacts, int id, const type::Vector3& n=type::Vector3(), bool useNormal=false);
    static inline int doIntersectionLinePoint(SReal dist2, const type::Vector3& p1, const type::Vector3& p2, const type::Vector3& q, OutputVector* contacts, int id, bool swapElems = false);
    static inline int doIntersectionTrianglePoint(SReal dist2, int flags, const type::Vector3& p1, const type::Vector3& p2, const type::Vector3& p3, const type::Vector3& n, const type::Vector3& q, OutputVector* contacts, int id, bool swapElems = false, bool useNormal=false);
    static inline int doIntersectionTrianglePoint2(SReal dist2, int flags, const type::Vector3& p1, const type::Vector3& p2, const type::Vector3& p3, const type::Vector3& n, const type::Vector3& q, OutputVector* contacts, int id, bool swapElems = false);

protected:

    NewProximityIntersection* intersection;
};

} // namespace sofa::component::collision::detection::intersection
