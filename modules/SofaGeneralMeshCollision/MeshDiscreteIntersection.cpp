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
#include <SofaGeneralMeshCollision/MeshDiscreteIntersection.inl>
#include <sofa/helper/system/config.h>
#include <sofa/helper/FnDispatcher.inl>
#include <sofa/core/collision/Intersection.inl>
#include <sofa/helper/proximity.h>
#include <iostream>
#include <algorithm>
#include <sofa/core/collision/IntersectorFactory.h>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;

SOFA_DECL_CLASS(MeshDiscreteIntersection)

IntersectorCreator<DiscreteIntersection, MeshDiscreteIntersection> MeshDiscreteIntersectors("Mesh");

MeshDiscreteIntersection::MeshDiscreteIntersection(DiscreteIntersection* object, bool addSelf)
    : intersection(object)
{
    if (addSelf)
    {
        intersection->intersectors.add<TriangleModel,     LineModel,       MeshDiscreteIntersection>  (this);
        intersection->intersectors.add<CapsuleModel,LineModel,MeshDiscreteIntersection>(this);
        intersection->intersectors.add<CapsuleModel,TriangleModel,MeshDiscreteIntersection>(this);
    }
}

bool MeshDiscreteIntersection::testIntersection(Triangle&, Line&)
{
    return true;
}

bool MeshDiscreteIntersection::testIntersection(Capsule&,Triangle&){
    return true;
}

bool MeshDiscreteIntersection::testIntersection(Capsule&,Line&){
    return true;
}

int MeshDiscreteIntersection::computeIntersection(Triangle& e1, Line& e2, OutputVector* contacts)
{
    Vector3 A = e1.p1();
    Vector3 AB = e1.p2()-A;
    Vector3 AC = e1.p3()-A;
    Vector3 P = e2.p1();
    Vector3 PQ = e2.p2()-P;
    Matrix3 M, Minv;
    Vector3 right;
    for (int i=0; i<3; i++)
    {
        M[i][0] = AB[i];
        M[i][1] = AC[i];
        M[i][2] = -PQ[i];
        right[i] = P[i]-A[i];
    }
    //sout << "M="<<M<<sendl;
    if (!Minv.invert(M))
        return 0;
    Vector3 baryCoords = Minv * right;
    if (baryCoords[0] < 0 || baryCoords[1] < 0 || baryCoords[0]+baryCoords[1] > 1)
        return 0; // out of the triangle
    if (baryCoords[2] < 0 || baryCoords[2] > 1)
        return 0; // out of the line

    Vector3 X = P+PQ*baryCoords[2];

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


} // namespace collision

} // namespace component

} // namespace sofa

