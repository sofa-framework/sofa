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
#include <sofa/helper/system/config.h>
#include <sofa/component/collision/MinProximityIntersection.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/proximity.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/collision/Intersection.inl>
#include <iostream>
#include <algorithm>
#include <sofa/helper/gl/template.h>
#include <sofa/component/collision/BaseIntTool.h>

#define DYNAMIC_CONE_ANGLE_COMPUTATION

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace helper;

SOFA_DECL_CLASS(MinProximityIntersection)

int MinProximityIntersectionClass = core::RegisterObject("A set of methods to compute if two primitives are close enougth to consider they collide")
        .add< MinProximityIntersection >()
        ;

MinProximityIntersection::MinProximityIntersection()
    : BaseProximityIntersection()
    , useSphereTriangle(initData(&useSphereTriangle, true, "useSphereTriangle","activate Sphere-Triangle intersection tests"))
    , usePointPoint(initData(&usePointPoint, true, "usePointPoint","activate Point-Point intersection tests"))
{
}

void MinProximityIntersection::init()
{
    intersectors.add<CubeModel, CubeModel, MinProximityIntersection>(this);
//        intersectors.add<RayModel, TriangleModel, MinProximityIntersection>(this);
    IntersectorFactory::getInstance()->addIntersectors(this);
}

bool MinProximityIntersection::testIntersection(Cube &cube1, Cube &cube2)
{
    return BaseIntTool::testIntersection(cube1,cube2,getAlarmDistance() + cube1.getProximity() + cube2.getProximity());
}

int MinProximityIntersection::computeIntersection(Cube&, Cube&, OutputVector* /*contacts*/)
{
    return 0; /// \todo
}

int MinProximityIntersection::computeIntersection(Capsule & e1,Capsule & e2,OutputVector * contacts){
    return CapsuleIntTool::computeIntersection(e1,e2,e1.getProximity() + e2.getProximity() + getAlarmDistance(),e1.getProximity() + e2.getProximity() + getContactDistance(),contacts);
}

int MinProximityIntersection::computeIntersection(Capsule & cap, Sphere & sph,OutputVector* contacts){
    return CapsuleIntTool::computeIntersection(cap,sph,getAlarmDistance() + cap.getProximity() + sph.getProximity(),getContactDistance() + cap.getProximity() + sph.getProximity(),contacts);
}

int MinProximityIntersection::computeIntersection(Capsule& cap, OBB& obb,OutputVector* contacts){
    return CapsuleIntTool::computeIntersection(cap,obb,getAlarmDistance() + cap.getProximity() + obb.getProximity(),getContactDistance()+ cap.getProximity() + obb.getProximity(),contacts);
}

int MinProximityIntersection::computeIntersection( Sphere& sph,OBB& obb,OutputVector* contacts){
    return OBBIntTool::computeIntersection(sph,obb,getAlarmDistance() + sph.getProximity() + obb.getProximity(),getContactDistance()+ sph.getProximity() + obb.getProximity(),contacts);
}

int MinProximityIntersection::computeIntersection(OBB& obb0,OBB& obb1,OutputVector* contacts){
    return OBBIntTool::computeIntersection(obb0,obb1,getAlarmDistance() + obb0.getProximity() + obb1.getProximity(),getContactDistance()+ obb0.getProximity() + obb1.getProximity(),contacts);
}

/*
bool MinProximityIntersection::testIntersection(Ray &t1,Triangle &t2)
{
	Vector3 P,Q,PQ;
	static DistanceSegTri proximitySolver;

	const double alarmDist = getAlarmDistance() + t1.getProximity() + t2.getProximity();

	if (fabs(t2.n() * t1.direction()) < 0.000001)
		return false; // no intersection for edges parallel to the triangle

	Vector3 A = t1.origin();
	Vector3 B = A + t1.direction() * t1.l();

	proximitySolver.NewComputation( &t2, A, B,P,Q);
	PQ = Q-P;

	if (PQ.norm2() < alarmDist*alarmDist)
	{
		//sout<<"Collision between Line - Triangle"<<sendl;
		return true;
	}
	else
		return false;
}

int MinProximityIntersection::computeIntersection(Ray &t1, Triangle &t2, OutputVector* contacts)
{
	const double alarmDist = getAlarmDistance() + t1.getProximity() + t2.getProximity();


	if (fabs(t2.n() * t1.direction()) < 0.000001)
		return false; // no intersection for edges parallel to the triangle

	Vector3 A = t1.origin();
	Vector3 B = A + t1.direction() * t1.l();

	Vector3 P,Q,PQ;
	static DistanceSegTri proximitySolver;

	proximitySolver.NewComputation( &t2, A,B,P,Q);
	PQ = Q-P;

	if (PQ.norm2() >= alarmDist*alarmDist)
		return 0;

	const double contactDist = alarmDist;
	contacts->resize(contacts->size()+1);
	DetectionOutput *detection = &*(contacts->end()-1);

	detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(t1, t2);
    detection->id = t1.getIndex();
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
*/
void MinProximityIntersection::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowCollisionModels())
        return;
}

} // namespace collision

} // namespace component

namespace core
{
namespace collision
{
template class SOFA_BASE_COLLISION_API IntersectorFactory<component::collision::MinProximityIntersection>;
}
}

} // namespace sofa

