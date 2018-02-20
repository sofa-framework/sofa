/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaUserInteraction/RayDiscreteIntersection.inl>
#include <sofa/helper/system/config.h>
#include <sofa/helper/FnDispatcher.inl>
#include <sofa/core/collision/Intersection.inl>
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
        intersection->intersectors.add<RayModel, SphereModel,       RayDiscreteIntersection>(this);
        intersection->intersectors.add<RayModel, RigidSphereModel,  RayDiscreteIntersection>(this);
        intersection->intersectors.add<RayModel, TriangleModel,     RayDiscreteIntersection>(this);
        intersection->intersectors.add<RayModel, OBBModel,          RayDiscreteIntersection>(this);

        // TODO implement ray vs capsule
        intersection->intersectors.ignore<RayModel, CapsuleModel>();
        intersection->intersectors.ignore<RayModel, RigidCapsuleModel>();
//        intersection->intersectors.add<RayModel, CapsuleModel,      RayDiscreteIntersection>(this);
//        intersection->intersectors.add<RayModel, RigidCapsuleModel, RayDiscreteIntersection>(this);

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


bool RayDiscreteIntersection::testIntersection( Ray& /*rRay*/, OBB& /*rOBB*/ )
{
    return false;
}




static float const c_fMin = -3.402823466e+38f;
static float const c_fMax = 3.402823466e+38f;
typedef Mat<3, 3, SReal> Mat33;

int  RayDiscreteIntersection::computeIntersection(Ray& rRay, OBB& rObb, OutputVector* contacts)
{
    //Near intersection: ray to closest plat of slab
    float fNear = c_fMin;

    //Far intersection: ray to farthest plane of slab
    float fFar = c_fMax;

    //Epsilon: for checks between planes and direction of ray
    float fEPSILON = 1e-4f;

    //Ray
    const Vector3& v3Origin = rRay.origin();
    const Vector3& v3Direction = rRay.direction();

    //Box
    const Vector3 v3HalfExtents = rObb.extents();
    const Vector3& v3BoxCenter = rObb.center();
    const Quaternion& qOrientation = rObb.orientation();
    Mat33 m33Orientation;
    qOrientation.toMatrix(m33Orientation);

    //Vector from origin of ray to center of box
    Vector3 v3RayOriginToBoxCenter = v3BoxCenter - v3Origin;

    //Normal at near and far intersection points
    Vector3 v3NormalAtNear(0.0f, 1.0f, 0.0f);
    Vector3 v3NormalAtFar(0.0f, 1.0f, 0.0f);

    //For the 3 slabs
    for(unsigned int i = 0; i < 3; i++)
    {
        Vector3 v3CurrAxis = m33Orientation.col(i); //TODO: implement the return of a reference instead of a copy of the column
        float fR = (float)(v3CurrAxis*v3RayOriginToBoxCenter);
        float fBoxMax = (float)v3HalfExtents[i];
        float fBoxMin = (float)-v3HalfExtents[i];
        float fNormal = 1.f;

        //Check if planes are parallel
        if(fabs(v3Direction*v3CurrAxis) < fEPSILON)
        {
            if(fBoxMin > fR || fBoxMax < fR)
            {
                return 0;
            }
        }
        else
        {
            // Ray not parallel to planes, so find intersection parameters
            float fS = (float)(v3Direction*v3CurrAxis);
            float fT0 = (fR + fBoxMax)/ fS;
            float fT1 = (fR + fBoxMin)/ fS;

            // Check ordering
            if(fT0 > fT1)
            {
                //Swap them
                float fTemp = fT0;
                fT0 = fT1;
                fT1 = fTemp;
                fNormal = -1.f;
            }

            //Compare with current values
            if(fT0 > fNear)
            {
                fNear = fT0;
                v3NormalAtNear = v3CurrAxis * fNormal;
            }
            if(fT1 < fFar)
            {
                fFar = fT1;
                v3NormalAtFar = v3CurrAxis * fNormal;
            }

            //Check if ray misses entirely, i.e. Exits with no collision as soon as slab intersection becomes empty
            if(fNear > fFar)
            {
                return 0;
            }

            // Ray starts after box, returns directly
            if(fFar < 0)
            {
                return 0;
            }

            // Ray ends before box, returns directly
            if(fNear > rRay.l())
            {
                return 0;
            }
        }
    }  //end of slabs checking

    bool bHit = false;

    float fHitFraction = 0;
    Vector3 v3Normal;
    Vector3 v3HitLocation;
    // If ray starts inside box
    if(fNear < 0.f)
    {
        // Make sure it does not ends strictly inside box
        if(fFar <= rRay.l())
        {
            bHit = true;
            fHitFraction = fFar / (float)rRay.l();
            v3Normal = v3NormalAtFar;
            v3HitLocation = rRay.origin() + rRay.direction() * rRay.l() * fHitFraction;
        }

    }
    else
    {
        // Filter hits if very close to each other
        if( fabs(fNear-fFar) < fEPSILON )
        {
            bHit = true;
            fHitFraction = fNear / (float)rRay.l();
            v3Normal = v3NormalAtNear;
            v3HitLocation = rRay.origin() + rRay.direction() * rRay.l();
        }
        else
        {
            bHit = true;
            fHitFraction =  fNear / (float)rRay.l();
            v3Normal = v3NormalAtNear;
            v3HitLocation =  rRay.origin() + rRay.direction() * fNear;

            // Ignore far hit if ends inside box
            //if(fFar <= rRay.l())
            //{
            //	bHit = true;
            //	fHitFraction = fFar / rRay.l();
            //	v3Normal = v3NormalAtFar;
            //	v3HitLocation = rRay.origin() + rRay.direction() * fFar;
            //}
        }
    }

    if (bHit)
    {
//		const SReal contactDist = fHitFraction;
        contacts->resize(contacts->size()+1);
        DetectionOutput *detection = &*(contacts->end()-1);

        detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(rRay, rObb);
        detection->point[1] = v3HitLocation;
        detection->point[0] = v3HitLocation;
        detection->normal = v3Normal;
        detection->value = fHitFraction;
        detection->value -= fHitFraction;

        return 1;
    }


    return 0;

}



} // namespace collision

} // namespace component

} // namespace sofa

