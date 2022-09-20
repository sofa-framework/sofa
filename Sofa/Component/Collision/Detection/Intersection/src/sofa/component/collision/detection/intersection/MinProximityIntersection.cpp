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
#define SOFA_COMPONENT_COLLISION_MINPROXIMITYINTERSECTION_CPP
#include <sofa/component/collision/detection/intersection/MinProximityIntersection.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/type/Vec.h>
#include <sofa/core/collision/Intersection.inl>

namespace sofa::core::collision
{
    template class SOFA_COMPONENT_COLLISION_DETECTION_INTERSECTION_API IntersectorFactory<component::collision::detection::intersection::MinProximityIntersection>;
} // namespace sofa::core::collision

namespace sofa::component::collision::detection::intersection
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace sofa::component::collision::geometry;
using namespace helper;

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

void MinProximityIntersection::doBaseObjectInit()
{
    intersectors.add<CubeCollisionModel, CubeCollisionModel, MinProximityIntersection>(this);
    intersectors.add<SphereCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>, MinProximityIntersection>(this);
    intersectors.add<RigidSphereModel,RigidSphereModel, MinProximityIntersection> (this);
    intersectors.add<SphereCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel, MinProximityIntersection>(this);

    //By default, all the previous pairs of collision models are supported,
    //but other C++ components are able to add a list of pairs to be supported.
    //In the following function, all the C++ components that registered to
    //MinProximityIntersection are created. In their constructors, they add
    //new supported pairs of collision models. For example, see MeshMinProximityIntersection.
    IntersectorFactory::getInstance()->addIntersectors(this);

	BaseProximityIntersection::doBaseObjectInit();
}

bool MinProximityIntersection::testIntersection(Cube& cube1, Cube& cube2)
{
    return BaseProximityIntersection::testIntersection(cube1, cube2);
}

int MinProximityIntersection::computeIntersection(Cube& cube1, Cube& cube2, OutputVector* contacts)
{
    return BaseProximityIntersection::computeIntersection(cube1, cube2, contacts);
}

bool MinProximityIntersection::getUseSurfaceNormals() const
{
    return useSurfaceNormals.getValue();
}

} // namespace sofa::component::collision::detection::intersection
