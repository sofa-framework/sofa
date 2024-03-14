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
#ifndef SOFA_COMPONENT_COLLISION_RIGIDDISTANCEGRIDDISCRETEINTERSECTION_INL
#define SOFA_COMPONENT_COLLISION_RIGIDDISTANCEGRIDDISCRETEINTERSECTION_INL
#include <iostream>
#include <algorithm>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/Intersection.inl>

#include "RigidDistanceGridDiscreteIntersection.h"

namespace sofa
{

namespace component
{

namespace collision
{

template<class T>
bool RigidDistanceGridDiscreteIntersection::testIntersection(RigidDistanceGridCollisionElement&, geometry::TSphere<T>&)
{
    return true;
}

template<class T>
int RigidDistanceGridDiscreteIntersection::computeIntersection(RigidDistanceGridCollisionElement& e1, geometry::TSphere<T>& e2, OutputVector* contacts)
{
    DistanceGrid* grid1 = e1.getGrid();
    bool useXForm = e1.isTransformed();
    const type::Vec3& t1 = e1.getTranslation();
    const sofa::type::Matrix3& r1 = e1.getRotation();

    const double d0 = e1.getProximity() + e2.getProximity() + intersection->getContactDistance() + e2.r();
    const SReal margin = 0.001f + (SReal)d0;

    type::Vec3 p2 = e2.center();
    DistanceGrid::Coord p1;

    if (useXForm)
    {
        p1 = r1.multTranspose(p2-t1);
    }
    else p1 = p2;

    if (!grid1->inBBox( p1, margin )) return 0;
    if (!grid1->inGrid( p1 ))
    {
        msg_error(intersection) << "Margin less than "<<margin<<" in DistanceGrid "<<e1.getCollisionModel()->getName();
        return 0;
    }

    SReal d = grid1->interp(p1);
    if (d >= margin) return 0;

    type::Vec3 grad = grid1->grad(p1); // note that there are some redundant computations between interp() and grad()
    grad.normalize();

    //p1 -= grad * d; // push p1 back to the surface

    contacts->resize(contacts->size()+1);
    core::collision::DetectionOutput *detection = &*(contacts->end()-1);
    detection->normal = (useXForm) ? r1 * grad : grad; // normal in global space from p1's surface
    detection->value = d - d0;
    detection->elem.first = e1;
    detection->elem.second = e2;
    detection->id = e2.getIndex();
    detection->point[0] = type::Vec3(p1) - grad * d;
    detection->point[1] = e2.getContactPointByNormal( detection->normal );
    return 1;
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
