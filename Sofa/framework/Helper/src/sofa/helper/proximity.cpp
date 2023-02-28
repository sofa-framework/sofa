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
#include <sofa/geometry/proximity/PointTriangle.h>
#include <sofa/geometry/proximity/SegmentTriangle.h>
#include <sofa/geometry/proximity/TriangleTriangle.h>
#include <sofa/helper/proximity.h>
#include <sofa/type/Mat_solve_LCP.h>

namespace sofa
{

namespace helper
{
using namespace sofa::helper;
using namespace sofa::type;

//------------------------
// DISTANCETRITRI
/*
 aP = alphaP, bP = betaP are the barycentric coordinates of point P on triangle P
 aQ = alphaQ, bQ = betaQ are the barycentric coordinates of point Q on triangle Q

 the computation of the distance between P and Q can be done with:

QP^2 = [aP bP aQ bQ] * [P1P2*P1P2 P1P2*P1P3 -P1P2*Q1Q2 -P1P2*Q1Q3 * [aP bP aQ bQ]^T
								   P1P3*P1P3 -P1P3*Q1Q2 -P1P3*Q1Q3
						sym					 Q1Q2*Q1Q2  Q1Q2*Q1Q3
													    Q1Q3*Q1Q3]
  + [Q1P1*P1P2 Q1P1*P1P3 -Q1P1*P1Q2 -Q1P1*P1Q3] *[aP bP aQ bQ]^T + (Q1P1)^2

  P and Q are on triangles, so we have the constraints:
aP>=0, bP>=0, aQ>=0, bQ>=0
aP+bP<=1, aQ+bQ<=1 <=> -aP-bP>=-1 , -aQ-bQ>=-1

To perform min(QP^2) we use LCP formalism

*/
//------------------------

DistanceTriTri::
DistanceTriTri()
{
}

DistanceTriTri::
~DistanceTriTri()
{
}



void DistanceTriTri::
NewComputation(const Vec3& P1, const Vec3& P2, const Vec3& P3, const Vec3& Q1, const Vec3& Q2, const Vec3& Q3, Vec3 &Presult, Vec3 &Qresult)
{
    const auto r = sofa::geometry::proximity::computeClosestPointsInTwoTriangles(P1, P2, P3, Q1, Q2, Q3, Presult, Qresult);

    if (!r)
    {
        printf(" no result from LCP !\n");
    }
}


//------------------------
// DISTANCESEGTRI
//------------------------

DistanceSegTri::
DistanceSegTri()
{
}

DistanceSegTri::
~DistanceSegTri()
{
}

void DistanceSegTri::
NewComputation(const Vec3 &P1, const Vec3 &P2, const Vec3 &P3, const Vec3 &Q1, const Vec3 &Q2, Vec3 &Presult, Vec3 &Qresult)
{
    const bool r = geometry::proximity::computeClosestPointsSegmentAndTriangle(P1, P2, P3, Q1, Q2, Presult, Qresult);

    if (!r)
    {
        printf(" no result from LCP !\n");
    }
}

//------------------------------------------------------------------------------------------------------ //
// ---------------------------------------- DISTANCEPOINTTRI ------------------------------------------- //
//------------------------------------------------------------------------------------------------------ //


DistancePointTri::
DistancePointTri()
{
}

DistancePointTri::
~DistancePointTri()
{
}

void DistancePointTri::
NewComputation(const Vec3 &P1, const Vec3 &P2, const Vec3 &P3, const Vec3 &Q, Vec3 &Presult)
{
    const bool r = geometry::proximity::computeClosestPointOnTriangleToPoint(P1, P2, P3, Q, Presult);

    if (!r)
    {
        printf(" no result from LCP !\n");
    }
}


} // namespace helper

} // namespace sofa
