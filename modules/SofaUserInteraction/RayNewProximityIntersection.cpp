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
#include <SofaUserInteraction/RayNewProximityIntersection.h>
#include <sofa/helper/system/config.h>
#include <sofa/helper/FnDispatcher.inl>
#include <sofa/core/collision/Intersection.inl>
#include <sofa/helper/proximity.h>
#include <iostream>
#include <algorithm>
#include <sofa/core/collision/IntersectorFactory.h>
#include <sofa/defaulttype/Mat.h>

#include <SofaUserInteraction/RayContact.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using sofa::helper::DistanceSegTri;

SOFA_DECL_CLASS(RayNewProximityIntersection)

IntersectorCreator<NewProximityIntersection, RayNewProximityIntersection> RayNewProximityIntersectors("Ray");

RayNewProximityIntersection::RayNewProximityIntersection(NewProximityIntersection* object, bool addSelf)
    : intersection(object)
{
    if (addSelf)
    {
        intersection->intersectors.ignore<RayModel, PointModel>();
        intersection->intersectors.ignore<RayModel, LineModel>();

        // why rigidsphere has a different collision detection compared to RayDiscreteIntersection?
        intersection->intersectors.add<RayModel, RigidSphereModel, RayNewProximityIntersection>(this);

        intersection->intersectors.add<RayModel, TriangleModel, RayNewProximityIntersection>(this);
    }
}

bool RayNewProximityIntersection::testIntersection(Ray &t1,Triangle &t2)
{
    Vector3 P,Q,PQ;
    static DistanceSegTri proximitySolver;
    const SReal alarmDist = intersection->getAlarmDistance() + t1.getProximity() + t2.getProximity();

    if (fabs(t2.n() * t1.direction()) < 0.000001)
        return false; // no intersection for edges parallel to the triangle

    Vector3 A = t1.origin();
    Vector3 B = A + t1.direction() * t1.l();

    proximitySolver.NewComputation( t2.p1(), t2.p2(), t2.p3(), A, B,P,Q);
    PQ = Q-P;

    if (PQ.norm2() < alarmDist*alarmDist)
    {
        //sout<<"Collision between Line - Triangle"<<sendl;
        return true;
    }
    else
        return false;
}


int RayNewProximityIntersection::computeIntersection(Ray &t1, Triangle &t2, OutputVector* contacts)
{
    const SReal alarmDist = intersection->getAlarmDistance() + t1.getProximity() + t2.getProximity();

    if (fabs(t2.n() * t1.direction()) < 0.000001)
        return false; // no intersection for edges parallel to the triangle

    Vector3 A = t1.origin();
    Vector3 B = A + t1.direction() * t1.l();

    Vector3 P,Q,PQ;
    static DistanceSegTri proximitySolver;

    proximitySolver.NewComputation( t2.p1(), t2.p2(), t2.p3(), A,B,P,Q);
    PQ = Q-P;

    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    const SReal contactDist = alarmDist;
    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(t1, t2);
    detection->point[1]=P;
    detection->point[0]=Q;
#ifdef DETECTIONOUTPUT_FREEMOTION
    detection->freePoint[1] = P;
    detection->freePoint[0] = Q;
#endif
    detection->normal=-t2.n();
    detection->value = PQ.norm();
    detection->value -= contactDist;
    return 1;
}




bool RayNewProximityIntersection::testIntersection( Ray& /*rRay*/, RigidSphere&)
{
    return false;
}


int RayNewProximityIntersection::computeIntersection(Ray& rRay, RigidSphere& rSphere, OutputVector* contacts)
{

    Vector3 v3SphereCenter = rSphere.center( );
    SReal fSphereRadii = rSphere.r();

    Vector3 v3RayOriginToSphereCenter = rRay.origin() - v3SphereCenter;
    SReal fB = v3RayOriginToSphereCenter * rRay.direction();
    SReal fC = v3RayOriginToSphereCenter * v3RayOriginToSphereCenter - fSphereRadii * fSphereRadii;

    // Exit if ray's origin outside sphere & ray's pointing away from sphere
    if((fC > 0.f) && (fB > 0.f))
    {
        return false;
    }

    // A negative discriminant corresponds to ray missing sphere
    SReal fDiscr = fB * fB - fC;
    if(fDiscr < 0.f)
    {
        return false;
    }

    // Ray intersects sphere, compute hits values
    int iHit = 0;
    Vector3 v3RayVector =  rRay.origin() + rRay.direction() * rRay.l();

    if(fDiscr < 1e-6f)
    {
        // One hit (on tangent)
        SReal fHitLength = -fB;

        // Make sure hit is on ray
        if((fHitLength < 0.f) || (fHitLength > rRay.l()))
        {
            return false;
        }


        SReal fHitFraction = fHitLength * (1.f / rRay.l() );
        Vector3 v3ContactPoint = rRay.origin() + v3RayVector * fHitFraction;
        Vector3 v3Normal = (v3ContactPoint - v3SphereCenter)/ fSphereRadii;

//		const SReal contactDist = fHitFraction;
        contacts->resize(contacts->size()+1);
        DetectionOutput *detection = &*(contacts->end()-1);

        detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(rRay, rSphere);
        detection->point[1] = v3ContactPoint;
        detection->point[0] = v3ContactPoint;
        detection->normal = v3Normal;
        detection->value = fHitFraction;
        detection->value -= fHitFraction;



        iHit = 1;
    }
    else
    {
        // Two hits, add contacts if on ray
        SReal fDiscrSqrt =   sqrt(fDiscr); //gnSqrt(fDiscr);
        SReal fHitLengthMin = -fB - fDiscrSqrt;
        SReal fHitLengthMax = -fB + fDiscrSqrt;

        if(( fHitLengthMin >= 0.f ) && ( fHitLengthMin <= rRay.l() ))
        {
            iHit = 1;

            //Contact 1
            SReal fHitFraction = fHitLengthMin  * ( 1.0f/rRay.l() );
            Vector3 v3ContactPoint = rRay.origin() + v3RayVector * fHitFraction;
            Vector3 v3Normal = ( v3ContactPoint - v3SphereCenter ) / fSphereRadii;

//			const SReal contactDist = fHitFraction;
            contacts->resize(contacts->size()+1);
            DetectionOutput *detection = &*(contacts->end()-1);

            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(rRay, rSphere);
            detection->point[1] = v3ContactPoint;
            detection->point[0] = v3ContactPoint;
            detection->normal = v3Normal;
            detection->value = fHitFraction;
            detection->value -= fHitFraction;


        }


        if((fHitLengthMax >= 0.f) && (fHitLengthMax <= rRay.l()))
        {
            iHit = 1;

            //Contact 2
            SReal fHitFraction = fHitLengthMax * ( 1.0f/rRay.l() );
            Vector3 v3ContactPoint = rRay.origin() + v3RayVector * fHitFraction;
            Vector3 v3Normal = ( v3ContactPoint - v3SphereCenter ) / fSphereRadii;


//			const SReal contactDist = fHitFraction;
            contacts->resize(contacts->size()+1);
            DetectionOutput *detection = &*(contacts->end()-1);

            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(rRay, rSphere);
            detection->point[1] = v3ContactPoint;
            detection->point[0] = v3ContactPoint;
            detection->normal = v3Normal;
            detection->value = fHitFraction;
            detection->value -= fHitFraction;


        }
    }





    return iHit;

}


} // namespace collision

} // namespace component

} // namespace sofa
