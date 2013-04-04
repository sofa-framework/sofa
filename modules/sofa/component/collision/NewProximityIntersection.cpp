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
#include <sofa/component/collision/NewProximityIntersection.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/collision/Intersection.inl>
#include <iostream>
#include <algorithm>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace helper;

SOFA_DECL_CLASS(NewProximityIntersection)

int NewProximityIntersectionClass = core::RegisterObject("Optimized Proximity Intersection based on Triangle-Triangle tests, ignoring Edge-Edge cases")
        .add< NewProximityIntersection >()
        ;

NewProximityIntersection::NewProximityIntersection()
    : BaseProximityIntersection()
    , useLineLine(initData(&useLineLine, false, "useLineLine", "Line-line collision detection enabled"))
{
}

void NewProximityIntersection::init()
{
    intersectors.add<CubeModel, CubeModel, NewProximityIntersection>(this);
    intersectors.add<SphereModel, SphereModel, NewProximityIntersection>(this);
    intersectors.add<CapsuleModel,CapsuleModel, NewProximityIntersection> (this);
    intersectors.add<CapsuleModel,SphereModel, NewProximityIntersection> (this);
    intersectors.add<RigidSphereModel,RigidSphereModel, NewProximityIntersection> (this);
    intersectors.add<OBBModel,OBBModel, NewProximityIntersection> (this);
    intersectors.add<CapsuleModel,OBBModel, NewProximityIntersection> (this);
    intersectors.add<SphereModel,OBBModel, NewProximityIntersection> (this);

    IntersectorFactory::getInstance()->addIntersectors(this);
}

bool NewProximityIntersection::testIntersection(Cube &cube1, Cube &cube2)
{
    const Vector3& minVect1 = cube1.minVect();
    const Vector3& minVect2 = cube2.minVect();
    const Vector3& maxVect1 = cube1.maxVect();
    const Vector3& maxVect2 = cube2.maxVect();

    const double alarmDist = getAlarmDistance() + cube1.getProximity() + cube2.getProximity();

    for (int i = 0; i < 3; i++)
    {
        if ( minVect1[i] > maxVect2[i] + alarmDist || minVect2[i] > maxVect1[i] + alarmDist )
            return false;
    }

    return true;
}


bool NewProximityIntersection::testIntersection(Sphere& e1, Sphere& e2)
{
    OutputVector contacts;
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e1.r() + e2.r();
    int n = doIntersectionPointPoint(alarmDist*alarmDist, e1.center(), e2.center(), &contacts, -1);
    return n>0;
}

bool NewProximityIntersection::testIntersection(Capsule&, Sphere&){
    //you can do but not useful because it is not called
    return false;
}


bool NewProximityIntersection::testIntersection(Capsule&, Capsule&){    
    return true;
}


int NewProximityIntersection::computeIntersection(Sphere& e1, Sphere& e2, OutputVector* contacts)
{
    const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e1.r() + e2.r();
    int n = doIntersectionPointPoint(alarmDist*alarmDist, e1.center(), e2.center(), contacts, (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex());
    if (n>0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity() + e1.r() + e2.r();
        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->value -= contactDist;
        }
    }
    return n;
}

int NewProximityIntersection::computeIntersection(RigidSphere& e1, RigidSphere& e2, OutputVector* contacts)
{
    const double alarmDist = getContactDistance() + e1.getProximity() + e2.getProximity() + e1.r() + e2.r();
    //const double alarmDist = getAlarmDistance() + e1.getProximity() + e2.getProximity() + e1.r() + e2.r();
    int n = doIntersectionPointPoint(alarmDist*alarmDist, e1.center(), e2.center(), contacts, (e1.getCollisionModel()->getSize() > e2.getCollisionModel()->getSize()) ? e1.getIndex() : e2.getIndex());
    if (n>0)
    {
        const double contactDist = getContactDistance() + e1.getProximity() + e2.getProximity() + e1.r() + e2.r();
        for (OutputVector::iterator detection = contacts->end()-n; detection != contacts->end(); ++detection)
        {
            detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(e1, e2);
            detection->value -= contactDist;
        }
    }
    return n;
}


int NewProximityIntersection::computeIntersection(Capsule & e1,Capsule & e2,OutputVector * contacts){
    return CapsuleIntTool::computeIntersection(e1,e2,e1.getProximity() + e2.getProximity() + getAlarmDistance(),e1.getProximity() + e2.getProximity() + getContactDistance(),contacts);
}

int NewProximityIntersection::computeIntersection(Capsule & cap, Sphere & sph,OutputVector* contacts){
    return CapsuleIntTool::computeIntersection(cap,sph,getAlarmDistance(),getContactDistance(),contacts);
}


int NewProximityIntersection::computeIntersection(Cube&, Cube&, OutputVector* /*contacts*/)
{
    return 0; /// \todo
}

bool NewProximityIntersection::testIntersection(RigidSphere&, RigidSphere&){
    return true;
}

bool NewProximityIntersection::testIntersection(OBB&, OBB&){
    return true;
}

int NewProximityIntersection::computeIntersection(OBB & box0, OBB & box1,OutputVector* contacts){
    return OBBIntTool::computeIntersection(box0,box1,box0.getProximity() + box1.getProximity() + getAlarmDistance(),box0.getProximity() + box1.getProximity() + getContactDistance(),contacts);
}


int NewProximityIntersection::computeIntersection(Capsule& cap,OBB& obb,OutputVector * contacts){
    return CapsuleIntTool::computeIntersection(cap,obb,cap.getProximity() + obb.getProximity() + getAlarmDistance(),cap.getProximity() + obb.getProximity() + getContactDistance(),contacts);
}


bool NewProximityIntersection::testIntersection(Capsule&, OBB&){
    return true;
}

int NewProximityIntersection::computeIntersection(Sphere & sph, OBB & box,OutputVector* contacts){
    return OBBIntTool::computeIntersection(sph,box,sph.getProximity() + box.getProximity() + getAlarmDistance(),box.getProximity() + sph.getProximity() + getContactDistance(),contacts);
}

bool NewProximityIntersection::testIntersection(Sphere &,OBB &){
    return true;
}

} // namespace collision

} // namespace component

namespace core
{
namespace collision
{
template class SOFA_BASE_COLLISION_API IntersectorFactory<component::collision::NewProximityIntersection>;
}
}

} // namespace sofa

