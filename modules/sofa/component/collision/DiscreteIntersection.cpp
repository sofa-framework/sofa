/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/helper/FnDispatcher.inl>
#include <sofa/component/collision/DiscreteIntersection.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/Intersection.inl>
//#include <sofa/component/collision/ProximityIntersection.h>
#include <sofa/helper/proximity.h>
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

SOFA_DECL_CLASS(DiscreteIntersection)

int DiscreteIntersectionClass = core::RegisterObject("TODO-DiscreteIntersectionClass")
        .add< DiscreteIntersection >()
        ;


DiscreteIntersection::DiscreteIntersection()
{
    intersectors.add<CubeModel,       CubeModel,         DiscreteIntersection> (this);
    intersectors.add<SphereModel,     SphereModel,       DiscreteIntersection> (this);
    IntersectorFactory::getInstance()->addIntersectors(this);
}

/// Return the intersector class handling the given pair of collision models, or NULL if not supported.
ElementIntersector* DiscreteIntersection::findIntersector(core::CollisionModel* object1, core::CollisionModel* object2, bool& swapModels)
{
    return intersectors.get(object1, object2, swapModels);
}

bool DiscreteIntersection::testIntersection(Cube& cube1, Cube& cube2)
{
    const Vector3& minVect1 = cube1.minVect();
    const Vector3& minVect2 = cube2.minVect();
    const Vector3& maxVect1 = cube1.maxVect();
    const Vector3& maxVect2 = cube2.maxVect();

    for (int i=0; i<3; i++)
    {
        if (minVect1[i] > maxVect2[i] || minVect2[i] > maxVect1[i])
            return false;
    }

    //sout << "Box <"<<minVect1[0]<<","<<minVect1[1]<<","<<minVect1[2]<<">-<"<<maxVect1[0]<<","<<maxVect1[1]<<","<<maxVect1[2]
    //  <<"> collide with Box "<<minVect2[0]<<","<<minVect2[1]<<","<<minVect2[2]<<">-<"<<maxVect2[0]<<","<<maxVect2[1]<<","<<maxVect2[2]<<">"<<sendl;
    return true;
}

//bool DiscreteIntersection::testIntersection(Triangle& t1, Triangle& t2)
//{
//	sout<<"Collision between Triangle - Triangle"<<sendl;
//	return false;
//}

int DiscreteIntersection::computeIntersection(Cube&, Cube&, OutputVector*)
{
    return 0; /// \todo
}

//int DiscreteIntersection::computeIntersection(Triangle&, Triangle&, OutputVector*)
//{
//	sout<<"Distance correction between Triangle - Triangle"<<sendl;
//	return 0;
//}


} // namespace collision

} // namespace component

} // namespace sofa

