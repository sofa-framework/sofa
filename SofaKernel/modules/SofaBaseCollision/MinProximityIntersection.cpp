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
#define SOFA_COMPONENT_COLLISION_MINPROXIMITYINTERSECTION_CPP
#include <sofa/helper/system/config.h>
#include <SofaBaseCollision/MinProximityIntersection.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/proximity.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/collision/Intersection.inl>
#include <iostream>
#include <algorithm>
#include <SofaBaseCollision/BaseIntTool.h>

#define DYNAMIC_CONE_ANGLE_COMPUTATION

namespace sofa
{

namespace core
{
	namespace collision
    {
        template class SOFA_BASE_COLLISION_API IntersectorFactory<component::collision::MinProximityIntersection>;
    }
}

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace helper;

SOFA_DECL_CLASS(MinProximityIntersection)

int MinProximityIntersectionClass = core::RegisterObject("A set of methods to compute if two primitives are close enough to consider they collide")
        .add< MinProximityIntersection >()
        ;

MinProximityIntersection::MinProximityIntersection()
    : BaseProximityIntersection()
    , useSphereTriangle(initData(&useSphereTriangle, true, "useSphereTriangle","activate Sphere-Triangle intersection tests"))
    , usePointPoint(initData(&usePointPoint, true, "usePointPoint","activate Point-Point intersection tests"))
    , useSurfaceNormals(initData(&useSurfaceNormals, false, "useSurfaceNormals", "Compute the norms of the Detection Outputs by considering the normals of the surfaces involved."))
    , useLinePoint(initData(&useLinePoint, true, "useLinePoint", "activate Line-Point intersection tests"))
    , useLineLine(initData(&useLineLine, true, "useLineLine", "activate Line-Line  intersection tests"))
{
}

void MinProximityIntersection::init()
{
    intersectors.add<CubeModel, CubeModel, MinProximityIntersection>(this);
    intersectors.add<SphereModel, SphereModel, MinProximityIntersection>(this);
    intersectors.add<CapsuleModel,CapsuleModel, MinProximityIntersection> (this);
    intersectors.add<CapsuleModel,SphereModel, MinProximityIntersection> (this);
    intersectors.add<OBBModel,OBBModel, MinProximityIntersection> (this);
    intersectors.add<CapsuleModel,OBBModel, MinProximityIntersection> (this);
    intersectors.add<SphereModel,OBBModel, MinProximityIntersection> (this);
    intersectors.add<RigidSphereModel,RigidSphereModel, MinProximityIntersection> (this);
    intersectors.add<SphereModel,RigidSphereModel, MinProximityIntersection> (this);
    intersectors.add<CapsuleModel,RigidSphereModel, MinProximityIntersection> (this);
    intersectors.add<RigidSphereModel,OBBModel, MinProximityIntersection> (this);

    IntersectorFactory::getInstance()->addIntersectors(this);

	BaseProximityIntersection::init();
}


bool MinProximityIntersection::getUseSurfaceNormals(){
    return useSurfaceNormals.getValue();
}

void MinProximityIntersection::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowCollisionModels())
        return;
}

} // namespace collision

} // namespace component

} // namespace sofa

