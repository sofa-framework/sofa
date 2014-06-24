/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <SofaUserInteraction/RayDiscreteIntersection.inl>
#include <sofa/helper/system/config.h>
#include <sofa/helper/FnDispatcher.inl>
#include <sofa/core/collision/Intersection.inl>
//#include <sofa/component/collision/ProximityIntersection.h>
#include <sofa/helper/proximity.h>
#include <iostream>
#include <algorithm>
#include <sofa/core/collision/IntersectorFactory.h>

#include <SofaBaseCollision/MinProximityIntersection.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;

SOFA_DECL_CLASS(RayDiscreteIntersection)

IntersectorCreator<DiscreteIntersection, RayDiscreteIntersection> RayDiscreteIntersectors("Ray");

// since MinProximityIntersection inherits from DiscreteIntersection, should not this line be implicit? (but it is not the case...)
IntersectorCreator<MinProximityIntersection, RayDiscreteIntersection> RayMinProximityIntersectors("Ray");

RayDiscreteIntersection::RayDiscreteIntersection(DiscreteIntersection* object, bool addSelf)
    : intersection(object)
{
    if (addSelf)
    {
        intersection->intersectors.add<RayModel, SphereModel,                     RayDiscreteIntersection>  (this);
        intersection->intersectors.add<RayModel, RigidSphereModel,                RayDiscreteIntersection>  (this);
        intersection->intersectors.add<RayModel, TriangleModel,                   RayDiscreteIntersection>  (this);

        intersection->intersectors.ignore<RayModel, PointModel>();
        intersection->intersectors.ignore<RayModel, LineModel>();
    }
}

bool RayDiscreteIntersection::testIntersection(Ray&, Triangle&)
{
    return true;
}

int RayDiscreteIntersection::computeIntersection(Ray& e1, Triangle& e2, OutputVector* contacts)
{
    Vector3 A = e2.p1();
    Vector3 AB = e2.p2()-A;
    Vector3 AC = e2.p3()-A;
    Vector3 P = e1.origin();
    Vector3 PQ = e1.direction();
    Matrix3 M, Minv;
    Vector3 right;
    for (int i=0; i<3; i++)
    {
        M[i][0] = AB[i];
        M[i][1] = AC[i];
        M[i][2] = -PQ[i];
        right[i] = P[i]-A[i];
    }
    if (!Minv.invert(M))
        return 0;
    Vector3 baryCoords = Minv * right;
    if (baryCoords[0] < 0 || baryCoords[1] < 0 || baryCoords[0]+baryCoords[1] > 1)
        return 0; // out of the triangle
    if (baryCoords[2] < 0 || baryCoords[2] > e1.l())
        return 0; // out of the line

    Vector3 X = P+PQ*baryCoords[2];

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

} // namespace collision

} // namespace component

} // namespace sofa

