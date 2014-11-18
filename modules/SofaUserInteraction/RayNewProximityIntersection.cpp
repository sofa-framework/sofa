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
#include <SofaUserInteraction/RayNewProximityIntersection.h>
#include <sofa/helper/system/config.h>
#include <sofa/helper/FnDispatcher.inl>
#include <sofa/core/collision/Intersection.inl>
//#include <sofa/component/collision/ProximityIntersection.h>
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

	//	intersection->intersectors.add<RayModel, RigidSphere, RayNewProximityIntersection>(this);

	//	intersection->intersectors.add<RayModel, SphereModel, RayNewProximityIntersection>(this);
		intersection->intersectors.add<RayModel, RigidSphereModel, RayNewProximityIntersection>(this);

		intersection->intersectors.add<RayModel, OBBModel, RayNewProximityIntersection>(this);
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

bool RayNewProximityIntersection::testIntersection( Ray& /*rRay*/, OBB& /*rOBB*/ )
{
	return false;
}

bool RayNewProximityIntersection::testIntersection( Ray& /*rRay*/, RigidSphere& /*rSphere*/ )
{
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

static float const c_fMin = -3.402823466e+38f;
static float const c_fMax = 3.402823466e+38f;
typedef Mat<3, 3, SReal> Mat33;

int  RayNewProximityIntersection::computeIntersection(Ray& rRay, OBB& rObb, OutputVector* contacts)
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
		float fR = v3CurrAxis*v3RayOriginToBoxCenter;
		float fBoxMax = v3HalfExtents[i];
		float fBoxMin = - v3HalfExtents[i];  
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
			float fS = v3Direction*v3CurrAxis; 
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
			fHitFraction = fFar / rRay.l();
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
			fHitFraction = fNear / rRay.l();	
			v3Normal = v3NormalAtNear;			
			v3HitLocation = rRay.origin() + rRay.direction() * rRay.l();
		}
		else
		{
			bHit = true;
			fHitFraction =  fNear / rRay.l();	
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

