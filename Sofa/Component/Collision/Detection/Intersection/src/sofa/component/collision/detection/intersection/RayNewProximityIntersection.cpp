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
#include <sofa/component/collision/detection/intersection/RayNewProximityIntersection.h>

#include <sofa/core/collision/Intersection.inl>
#include <iostream>
#include <algorithm>
#include <sofa/core/collision/IntersectorFactory.h>
#include <sofa/geometry/proximity/PointTriangle.h>
#include <sofa/geometry/proximity/SegmentTriangle.h>
#include <sofa/type/Mat.h>

namespace sofa::component::collision::detection::intersection
{

using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace sofa::component::collision::geometry;

IntersectorCreator<NewProximityIntersection, RayNewProximityIntersection> RayNewProximityIntersectors("Ray");

RayNewProximityIntersection::RayNewProximityIntersection(NewProximityIntersection* object, bool addSelf)
    : intersection(object)
{
    if (addSelf)
    {
        intersection->intersectors.ignore<RayCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>>();
        intersection->intersectors.ignore<RayCollisionModel, LineCollisionModel<sofa::defaulttype::Vec3Types>>();

        // why rigidsphere has a different collision detection compared to RayDiscreteIntersection?
        intersection->intersectors.add<RayCollisionModel, RigidSphereModel, RayNewProximityIntersection>(this);

        intersection->intersectors.add<RayCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>, RayNewProximityIntersection>(this);
    }
}

bool RayNewProximityIntersection::testIntersection(Ray &t1,Triangle &t2)
{
    Vec3 P,Q,PQ;
    const SReal alarmDist = intersection->getAlarmDistance() + t1.getProximity() + t2.getProximity();

    if (fabs(t2.n() * t1.direction()) < 0.000001)
        return false; // no intersection for edges parallel to the triangle

    const Vec3 A = t1.origin();
    const Vec3 B = A + t1.direction() * t1.l();

    const auto r = sofa::geometry::proximity::computeClosestPointsSegmentAndTriangle(t2.p1(), t2.p2(), t2.p3(), A, B,P,Q);
    msg_warning_when(!r, "RayNewProximityIntersection") << "Failed to compute distance between ray ["
        << A << "," << B <<"] and triangle [" << t2.p1() << ", " << t2.p2() << ", " << t2.p3() << "]";

    PQ = Q-P;

    if (PQ.norm2() < alarmDist*alarmDist)
    {
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

    const Vec3 A = t1.origin();
    const Vec3 B = A + t1.direction() * t1.l();

    Vec3 P,Q,PQ;

    const auto r = sofa::geometry::proximity::computeClosestPointsSegmentAndTriangle(t2.p1(), t2.p2(), t2.p3(), A, B,P,Q);
    msg_warning_when(!r, "RayNewProximityIntersection") << "Failed to compute distance between ray ["
        << A << "," << B <<"] and triangle [" << t2.p1() << ", " << t2.p2() << ", " << t2.p3() << "]";

    PQ = Q-P;

    if (PQ.norm2() >= alarmDist*alarmDist)
        return 0;

    const SReal contactDist = alarmDist;
    contacts->resize(contacts->size()+1);
    DetectionOutput *detection = &*(contacts->end()-1);

    detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(t1, t2);
    detection->point[1]=P;
    detection->point[0]=Q;
#ifdef SOFA_DETECTIONOUTPUT_FREEMOTION
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
    const Vec3 v3SphereCenter = rSphere.center( );
    const SReal fSphereRadii = rSphere.r();

    const Vec3 v3RayOriginToSphereCenter = rRay.origin() - v3SphereCenter;
    const SReal fB = v3RayOriginToSphereCenter * rRay.direction();
    const SReal fC = v3RayOriginToSphereCenter * v3RayOriginToSphereCenter - fSphereRadii * fSphereRadii;

    // Exit if ray's origin outside sphere & ray's pointing away from sphere
    if((fC > 0.f) && (fB > 0.f))
    {
        return false;
    }

    // A negative discriminant corresponds to ray missing sphere
    const SReal fDiscr = fB * fB - fC;
    if(fDiscr < 0.f)
    {
        return false;
    }

    // Ray intersects sphere, compute hits values
    int iHit = 0;
    const Vec3 v3RayVector =  rRay.origin() + rRay.direction() * rRay.l();

    if(fDiscr < 1e-6f)
    {
        // One hit (on tangent)
        const SReal fHitLength = -fB;

        // Make sure hit is on ray
        if((fHitLength < 0.f) || (fHitLength > rRay.l()))
        {
            return false;
        }


        const SReal fHitFraction = fHitLength * (1.f / rRay.l() );
        const Vec3 v3ContactPoint = rRay.origin() + v3RayVector * fHitFraction;
        const Vec3 v3Normal = (v3ContactPoint - v3SphereCenter)/ fSphereRadii;

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
        const SReal fDiscrSqrt =   sqrt(fDiscr); //gnSqrt(fDiscr);
        const SReal fHitLengthMin = -fB - fDiscrSqrt;
        const SReal fHitLengthMax = -fB + fDiscrSqrt;

        if(( fHitLengthMin >= 0.f ) && ( fHitLengthMin <= rRay.l() ))
        {
            iHit = 1;

            //Contact 1
            const SReal fHitFraction = fHitLengthMin  * ( 1.0f/rRay.l() );
            const Vec3 v3ContactPoint = rRay.origin() + v3RayVector * fHitFraction;
            const Vec3 v3Normal = ( v3ContactPoint - v3SphereCenter ) / fSphereRadii;

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
            const SReal fHitFraction = fHitLengthMax * ( 1.0f/rRay.l() );
            const Vec3 v3ContactPoint = rRay.origin() + v3RayVector * fHitFraction;
            const Vec3 v3Normal = ( v3ContactPoint - v3SphereCenter ) / fSphereRadii;


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


} //namespace sofa::component::collision::detection::intersection
