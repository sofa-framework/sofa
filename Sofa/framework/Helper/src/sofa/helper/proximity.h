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
#ifndef SOFA_COMPONENT_COLLISION_PROXIMITY_H
#define SOFA_COMPONENT_COLLISION_PROXIMITY_H

#include <sofa/type/Vec.h>
#include <sofa/helper/config.h>

#ifndef SOFA_BUILD_SOFA_HELPER
SOFA_DEPRECATED_HEADER("v23.06", "v23.12", "sofa/geometry/proximity/PointTriangle.h, sofa/geometry/proximity/SegmentTriangle.h or sofa/geometry/proximity/TriangleTriangle.h")
#endif

namespace sofa
{

namespace helper
{

//-----------------------------------------------------------------------------
//--DistanceTriTri--
//------------------
// this class computes the shortest distance between a triangle P and a triangle Q
//-----------------------------------------------------------------------------
class SOFA_HELPER_API DistanceTriTri
{
public:
    DistanceTriTri(); // start a Proximity solver
    ~DistanceTriTri();

    // init the solver with the new coordinates of the triangle & the segment
    // solve the lcp
    SOFA_PROXIMITY_CLASSES_DEPRECATED()
    void NewComputation(const sofa::type::Vec3& P1, const sofa::type::Vec3& P2, const sofa::type::Vec3& P3, const sofa::type::Vec3& Q1, const sofa::type::Vec3& Q2, const sofa::type::Vec3& Q3, sofa::type::Vec3 &Presult, sofa::type::Vec3 &Qresult);

};

//-----------------------------------------------------------------------------
//--DistanceSegTri--
//------------------
// this class compute the shortest distance between a triangle and a segment
//-----------------------------------------------------------------------------
class SOFA_HELPER_API DistanceSegTri
{
public:
    DistanceSegTri(); // start a Proximity solver
    ~DistanceSegTri();

    // init the solver with the new coordinates of the triangle & the segment
    // solve the lcp
    SOFA_PROXIMITY_CLASSES_DEPRECATED()
    void NewComputation(const sofa::type::Vec3 &P1, const sofa::type::Vec3 &P2, const sofa::type::Vec3 &P3, const sofa::type::Vec3 &Q1, const sofa::type::Vec3 &Q2, sofa::type::Vec3 &Presult, sofa::type::Vec3 &Qresult);
};

//-----------------------------------------------------------------------------
//--DistancePointTri--
//------------------
// this class compute the shortest distance between a triangle and a Point
//-----------------------------------------------------------------------------
class SOFA_HELPER_API DistancePointTri
{
public:
    DistancePointTri(); // start a Proximity solver
    ~DistancePointTri();

    // init the solver with the new coordinates of the triangle & the segment
    // solve the lcp
    SOFA_PROXIMITY_CLASSES_DEPRECATED()
    void NewComputation(const sofa::type::Vec3 &P1, const sofa::type::Vec3 &P2, const sofa::type::Vec3 &P3, const sofa::type::Vec3 &Q, sofa::type::Vec3 &Presult);
};

} // namespace helper

} // namespace sofa


#endif

