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

#ifndef SOFA_FLAG_SOFAPRO
#include <sofa/component/collision/CapsuleModel.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/component/collision/CapsuleIntTool.h>
#include <sofa/component/collision/OBBModel.h>
#include <sofa/component/collision/OBBIntTool.h>
#endif // SOFA_FLAG_SOFAPRO

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
    intersectors.add<SphereModel, SphereModel, NewProximityIntersection>(this);
    intersectors.add<CubeModel, CubeModel, NewProximityIntersection>(this);

#ifndef SOFA_FLAG_SOFAPRO
    intersectors.add<CapsuleModel,CapsuleModel, NewProximityIntersection> (this);
    intersectors.add<CapsuleModel,SphereModel, NewProximityIntersection> (this);
    intersectors.add<OBBModel,OBBModel, NewProximityIntersection> (this);
    intersectors.add<CapsuleModel,OBBModel, NewProximityIntersection> (this);
    intersectors.add<SphereModel,OBBModel, NewProximityIntersection> (this);
    intersectors.add<RigidSphereModel,RigidSphereModel, NewProximityIntersection> (this);
    intersectors.add<SphereModel,RigidSphereModel, NewProximityIntersection> (this);
    intersectors.add<CapsuleModel,RigidSphereModel, NewProximityIntersection> (this);
    intersectors.add<RigidSphereModel,OBBModel, NewProximityIntersection> (this);

    intersectors.add<CapsuleModel,RigidCapsuleModel, NewProximityIntersection> (this);
    intersectors.add<RigidCapsuleModel,SphereModel, NewProximityIntersection> (this);
    intersectors.add<RigidCapsuleModel,OBBModel, NewProximityIntersection> (this);
    intersectors.add<RigidCapsuleModel,RigidSphereModel, NewProximityIntersection> (this);
#endif // SOFA_FLAG_SOFAPRO

    IntersectorFactory::getInstance()->addIntersectors(this);
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

