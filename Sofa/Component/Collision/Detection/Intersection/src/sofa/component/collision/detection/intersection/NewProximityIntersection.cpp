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
#define SOFA_COMPONENT_COLLISION_NEWPROXIMITYINTERSECTION_CPP
#include <sofa/component/collision/detection/intersection/NewProximityIntersection.inl>

#include <sofa/core/collision/Intersection.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa::core::collision
{
    template class SOFA_COMPONENT_COLLISION_DETECTION_INTERSECTION_API IntersectorFactory<component::collision::detection::intersection::NewProximityIntersection>;
} // namespace sofa::core::collision

namespace sofa::component::collision::detection::intersection
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace sofa::component::collision::geometry;
using namespace helper;

int NewProximityIntersectionClass = core::RegisterObject("Optimized Proximity Intersection based on Triangle-Triangle tests, ignoring Edge-Edge cases")
        .add< NewProximityIntersection >()
        ;

NewProximityIntersection::NewProximityIntersection()
    : BaseProximityIntersection()
    , d_useLineLine(initData(&d_useLineLine, false, "useLineLine", "Line-line collision detection enabled"))
{
    useLineLine.setParent(&d_useLineLine);
}

void NewProximityIntersection::init()
{
    intersectors.add<CubeCollisionModel, CubeCollisionModel, NewProximityIntersection>(this);
    intersectors.add<SphereCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>, NewProximityIntersection>(this);
    intersectors.add<RigidSphereModel, RigidSphereModel, NewProximityIntersection>(this);
    intersectors.add<SphereCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel, NewProximityIntersection>(this);

    //By default, all the previous pairs of collision models are supported,
    //but other C++ components are able to add a list of pairs to be supported.
    //In the following function, all the C++ components that registered to
    //NewProximityIntersection are created. In their constructors, they add
    //new supported pairs of collision models. For example, see MeshNewProximityIntersection.
    IntersectorFactory::getInstance()->addIntersectors(this);

	BaseProximityIntersection::init();
}

bool NewProximityIntersection::testIntersection(Cube& cube1, Cube& cube2, const core::collision::Intersection* currentIntersection)
{
    SOFA_UNUSED(currentIntersection);

    return BaseProximityIntersection::testIntersection(cube1, cube2, currentIntersection);
}

int NewProximityIntersection::computeIntersection(Cube& cube1, Cube& cube2, OutputVector* contacts, const core::collision::Intersection* currentIntersection)
{
    SOFA_UNUSED(currentIntersection);

    return BaseProximityIntersection::computeIntersection(cube1, cube2, contacts, currentIntersection);
}

bool NewProximityIntersection::testIntersection(Cube& cube1, Cube& cube2)
{
    return testIntersection(cube1, cube2, this );
}

int NewProximityIntersection::computeIntersection(Cube& cube1, Cube& cube2, OutputVector* contacts)
{
    return computeIntersection(cube1, cube2, contacts, this);
}


} // namespace sofa::component::collision::detection::intersection
