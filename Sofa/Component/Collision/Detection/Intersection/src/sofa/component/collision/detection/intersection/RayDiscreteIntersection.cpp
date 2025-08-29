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
#include <sofa/component/collision/detection/intersection/RayDiscreteIntersection.inl>

#include <sofa/core/collision/Intersection.inl>
#include <iostream>
#include <algorithm>
#include <sofa/core/collision/IntersectorFactory.h>

#include <sofa/component/collision/detection/intersection/MinProximityIntersection.h>

namespace sofa::component::collision::detection::intersection
{

using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace sofa::component::collision::geometry;

IntersectorCreator<DiscreteIntersection, RayDiscreteIntersection> RayDiscreteIntersectors("Ray");

// since MinProximityIntersection inherits from DiscreteIntersection, should not this line be implicit? (but it is not the case...)
IntersectorCreator<MinProximityIntersection, RayDiscreteIntersection> RayMinProximityIntersectors("Ray");

RayDiscreteIntersection::RayDiscreteIntersection(DiscreteIntersection* intersection, bool addSelf)
{
    if (addSelf)
    {
        intersection->intersectors.add<RayCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>,       RayDiscreteIntersection>(this);
        intersection->intersectors.add<RayCollisionModel, RigidSphereModel,  RayDiscreteIntersection>(this);
        intersection->intersectors.add<RayCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>,     RayDiscreteIntersection>(this);

        intersection->intersectors.ignore<RayCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>>();
        intersection->intersectors.ignore<RayCollisionModel, LineCollisionModel<sofa::defaulttype::Vec3Types>>();
    }
}

bool RayDiscreteIntersection::testIntersection(Ray&, Triangle&, const sofa::core::collision::Intersection*)
{
    return true;
}

int RayDiscreteIntersection::computeIntersection(Ray& e1, Triangle& e2, OutputVector* contacts, const sofa::core::collision::Intersection*)
{
    Vec3 A = e2.p1();
    Vec3 AB = e2.p2()-A;
    Vec3 AC = e2.p3()-A;
    Vec3 P = e1.origin();
    Vec3 PQ = e1.direction();
    Matrix3 M, Minv;
    Vec3 right;
    for (int i=0; i<3; i++)
    {
        M(i,0) = AB[i];
        M(i,1) = AC[i];
        M(i,2) = -PQ[i];
        right[i] = P[i]-A[i];
    }
    if (!Minv.invert(M))
        return 0;
    Vec3 baryCoords = Minv * right;
    if (baryCoords[0] < 0 || baryCoords[1] < 0 || baryCoords[0]+baryCoords[1] > 1)
        return 0; // out of the triangle
    if (baryCoords[2] < 0 || baryCoords[2] > e1.l())
        return 0; // out of the line

    const Vec3 X = P+PQ*baryCoords[2];

    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);
    detection->point[0] = X;
    detection->point[1] = X;
    detection->normal = -e2.n();
    detection->value = 0;
    detection->elem.first = e1;
    detection->elem.second = e2;
    detection->id = e1.getIndex();
    return 1;
}

} //namespace sofa::component::collision::detection::intersection
