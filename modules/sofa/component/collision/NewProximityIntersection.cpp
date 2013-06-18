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
    intersectors.add<OBBModel,OBBModel, NewProximityIntersection> (this);
    intersectors.add<CapsuleModel,OBBModel, NewProximityIntersection> (this);
    intersectors.add<SphereModel,OBBModel, NewProximityIntersection> (this);
    intersectors.add<RigidSphereModel,RigidSphereModel, NewProximityIntersection> (this);
    intersectors.add<SphereModel,RigidSphereModel, NewProximityIntersection> (this);
    intersectors.add<CapsuleModel,RigidSphereModel, NewProximityIntersection> (this);
    intersectors.add<RigidSphereModel,OBBModel, NewProximityIntersection> (this);

    IntersectorFactory::getInstance()->addIntersectors(this);
}

bool NewProximityIntersection::testIntersection(Cube &cube1, Cube &cube2)
{
    return BaseIntTool::testIntersection(cube1,cube2,getAlarmDistance() + cube1.getProximity() + cube2.getProximity());
}


bool NewProximityIntersection::testIntersection(Capsule&, Capsule&){    
    return true;
}


int NewProximityIntersection::computeIntersection(Capsule & e1,Capsule & e2,OutputVector * contacts){
    return CapsuleIntTool::computeIntersection(e1,e2,e1.getProximity() + e2.getProximity() + getAlarmDistance(),e1.getProximity() + e2.getProximity() + getContactDistance(),contacts);
}


int NewProximityIntersection::computeIntersection(Cube&, Cube&, OutputVector* /*contacts*/)
{
    return 0; /// \todo
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

