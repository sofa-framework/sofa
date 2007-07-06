/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_DISCRETEINTERSECTION_H
#define SOFA_COMPONENT_COLLISION_DISCRETEINTERSECTION_H

#include <sofa/core/componentmodel/collision/Intersection.h>
#include <sofa/helper/FnDispatcher.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/collision/LineModel.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/component/collision/RayModel.h>
#include <sofa/component/collision/SphereTreeModel.h>
#include <sofa/component/collision/DistanceGridCollisionModel.h>

namespace sofa
{

namespace component
{

namespace collision
{

class DiscreteIntersection : public core::componentmodel::collision::Intersection
{
public:
    DiscreteIntersection();

    /// Return the intersector class handling the given pair of collision models, or NULL if not supported.
    virtual core::componentmodel::collision::ElementIntersector* findIntersector(core::CollisionModel* object1, core::CollisionModel* object2);

protected:
    core::componentmodel::collision::IntersectorMap intersectors;

public:

    bool testIntersection(Cube&, Cube&);
    bool testIntersection(Sphere&, Sphere&);
    bool testIntersection(Sphere&, Ray&);


    bool testIntersection(SingleSphere&, SingleSphere&);
    bool testIntersection(SingleSphere&, Cube&);
    bool testIntersection(SingleSphere&, Ray&);
    bool testIntersection(SingleSphere&, Triangle&);
    //bool testIntersection(Sphere& , Triangle&);
    //bool testIntersection(Triangle& ,Triangle&);
    bool testIntersection(DistanceGridCollisionElement&, DistanceGridCollisionElement&);

    int computeIntersection(Cube&, Cube&, DetectionOutputVector&);
    int computeIntersection(Sphere&, Sphere&, DetectionOutputVector&);
    int computeIntersection(Sphere&, Ray&, DetectionOutputVector&);
    int computeIntersection(SingleSphere&, SingleSphere&, DetectionOutputVector&);
    int computeIntersection(SingleSphere&, Cube&, DetectionOutputVector&);
    int computeIntersection(SingleSphere&, Ray&, DetectionOutputVector&);
    int computeIntersection(SingleSphere&, Triangle&, DetectionOutputVector&);
    //int computeIntersection(Sphere&, Triangle&, DetectionOutputVector&);
    //int computeIntersection(Triangle&, Triangle&, DetectionOutputVector&);
    int computeIntersection(DistanceGridCollisionElement&, DistanceGridCollisionElement&, DetectionOutputVector&);

};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
