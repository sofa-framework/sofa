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
#include <sofa/component/collision/detection/intersection/MeshDiscreteIntersection.inl>

#include <sofa/core/collision/Intersection.inl>
#include <sofa/core/collision/IntersectorFactory.h>

namespace sofa::component::collision::detection::intersection
{

using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace sofa::component::collision::geometry;

IntersectorCreator<DiscreteIntersection, MeshDiscreteIntersection> MeshDiscreteIntersectors("Mesh");

MeshDiscreteIntersection::MeshDiscreteIntersection(DiscreteIntersection* intersection, bool addSelf)
{
    if (addSelf)
    {
        intersection->intersectors.add<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>, MeshDiscreteIntersection>  (this);
    }
}

bool MeshDiscreteIntersection::testIntersection(Triangle&, Line&, const sofa::core::collision::Intersection*)
{
    return true;
}

int MeshDiscreteIntersection::computeIntersection(Triangle& e1, Line& e2, OutputVector* contacts, const core::collision::Intersection* currentIntersection)
{
    SOFA_UNUSED(currentIntersection);

    static_assert(std::is_same_v<Triangle::Coord, Line::Coord>, "Data mismatch");
    static_assert(Triangle::Coord::total_size == 3, "Must be a vec type");

    const Triangle::Coord& A = e1.p1();
    const Triangle::Coord AB = e1.p2()-A;
    const Triangle::Coord AC = e1.p3()-A;
    const Line::Coord& P = e2.p1();
    const Line::Coord PQ = e2.p2()-P;
    Mat<3, 3, Triangle::Coord::value_type> M(NOINIT);
    Mat<3, 3, Triangle::Coord::value_type> Minv(NOINIT);
    Triangle::Coord right(NOINIT);
    for (int i=0; i<3; i++)
    {
        M(i,0) = AB[i];
        M(i,1) = AC[i];
        M(i,2) = -PQ[i];
        right[i] = P[i]-A[i];
    }
    if (!Minv.invert(M))
        return 0;
    const Triangle::Coord baryCoords = Minv * right;
    if (baryCoords[0] < 0 || baryCoords[1] < 0 || baryCoords[0]+baryCoords[1] > 1)
        return 0; // out of the triangle
    if (baryCoords[2] < 0 || baryCoords[2] > 1)
        return 0; // out of the line

    const Triangle::Coord X = P+PQ*baryCoords[2];

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    detection->point[0] = X;
    detection->point[1] = X;
    detection->normal = e1.n();
    detection->value = 0;
    detection->elem.first = e1;
    detection->elem.second = e2;
    detection->id = e2.getIndex();
    return 1;
}

} // namespace sofa::component::collision::detection::intersection
