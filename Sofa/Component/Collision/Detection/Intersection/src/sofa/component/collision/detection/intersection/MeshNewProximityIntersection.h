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
#include <sofa/component/collision/geometry/TriangleCollisionModel.h>
#include <sofa/component/collision/geometry/LineCollisionModel.h>
#include <sofa/component/collision/geometry/PointCollisionModel.h>

namespace sofa::component::collision::detection::intersection
{

class SOFA_COMPONENT_COLLISION_DETECTION_INTERSECTION_API MeshNewProximityIntersection : public core::collision::BaseIntersector
{
    typedef NewProximityIntersection::OutputVector OutputVector;

public:
    MeshNewProximityIntersection(NewProximityIntersection* intersection, bool addSelf=true);
    
    bool testIntersection(collision::geometry::Point&, collision::geometry::Point&, const core::collision::Intersection* currentIntersection);
    int computeIntersection(collision::geometry::Point&, collision::geometry::Point&, OutputVector*, const core::collision::Intersection* currentIntersection);
    bool testIntersection(collision::geometry::Line&, collision::geometry::Point&, const core::collision::Intersection* currentIntersection);
    int computeIntersection(collision::geometry::Line&, collision::geometry::Point&, OutputVector*, const core::collision::Intersection* currentIntersection);
    bool testIntersection(collision::geometry::Line&, collision::geometry::Line&, const core::collision::Intersection* currentIntersection);
    int computeIntersection(collision::geometry::Line&, collision::geometry::Line&, OutputVector*, const core::collision::Intersection* currentIntersection);
    bool testIntersection(collision::geometry::Triangle&, collision::geometry::Point&, const core::collision::Intersection* currentIntersection);
    int computeIntersection(collision::geometry::Triangle&, collision::geometry::Point&, OutputVector*, const core::collision::Intersection* currentIntersection);
    bool testIntersection(collision::geometry::Triangle&, collision::geometry::Line&, const core::collision::Intersection* currentIntersection);
    int computeIntersection(collision::geometry::Triangle&, collision::geometry::Line&, OutputVector*, const core::collision::Intersection* currentIntersection);
    bool testIntersection(collision::geometry::Triangle&, collision::geometry::Triangle&, const core::collision::Intersection* currentIntersection);
    int computeIntersection(collision::geometry::Triangle&, collision::geometry::Triangle&, OutputVector*, const core::collision::Intersection* currentIntersection);
    
    template <class T>
    bool testIntersection(collision::geometry::TSphere<T>& sph, collision::geometry::Point& pt, const core::collision::Intersection* currentIntersection);
    template <class T>
    int computeIntersection(collision::geometry::TSphere<T>& sph, collision::geometry::Point& pt, OutputVector*, const core::collision::Intersection* currentIntersection);
    template <class T>
    bool testIntersection(collision::geometry::Line&, collision::geometry::TSphere<T>&, const core::collision::Intersection* currentIntersection);
    template <class T>
    int computeIntersection(collision::geometry::Line& line, collision::geometry::TSphere<T>& sph, OutputVector*, const core::collision::Intersection* currentIntersection);
    template <class T>
    bool testIntersection(collision::geometry::Triangle&, collision::geometry::TSphere<T>&, const core::collision::Intersection* currentIntersection);
    template <class T>
    int computeIntersection(collision::geometry::Triangle& tri, collision::geometry::TSphere<T>& sph, OutputVector*, const core::collision::Intersection* currentIntersection);

    static inline int doIntersectionLineLine(SReal dist2, const type::Vec3& p1, const type::Vec3& p2, const type::Vec3& q1, const type::Vec3& q2, OutputVector* contacts, int id, const type::Vec3& n=type::Vec3(), bool useNormal=false);
    static inline int doIntersectionLinePoint(SReal dist2, const type::Vec3& p1, const type::Vec3& p2, const type::Vec3& q, OutputVector* contacts, int id, bool swapElems = false);
    static inline int doIntersectionTrianglePoint(SReal dist2, int flags, const type::Vec3& p1, const type::Vec3& p2, const type::Vec3& p3, const type::Vec3& n, const type::Vec3& q, OutputVector* contacts, int id, bool swapElems = false, bool useNormal=false);
    static inline int doIntersectionTrianglePoint2(SReal dist2, int flags, const type::Vec3& p1, const type::Vec3& p2, const type::Vec3& p3, const type::Vec3& n, const type::Vec3& q, OutputVector* contacts, int id, bool swapElems = false);
};

} // namespace sofa::component::collision::detection::intersection
