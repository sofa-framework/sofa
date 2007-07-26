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
#include <sofa/component/collision/PointModel.h>
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

    template<class Sphere>
    bool testIntersection(Sphere&, Sphere&);
    template<class Sphere>
    bool testIntersection(Sphere&, Cube&);
    template<class Sphere>
    bool testIntersection(Sphere&, Ray&);
    template<class Sphere>
    bool testIntersection(Sphere& , Triangle&);
    //bool testIntersection(Triangle& ,Triangle&);
    bool testIntersection(RigidDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&);
    bool testIntersection(RigidDistanceGridCollisionElement&, Point&);
    template<class Sphere>
    bool testIntersection(RigidDistanceGridCollisionElement&, Sphere&);
    bool testIntersection(RigidDistanceGridCollisionElement&, Triangle&);
    bool testIntersection(RigidDistanceGridCollisionElement&, Ray&);
    bool testIntersection(FFDDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&);
    bool testIntersection(FFDDistanceGridCollisionElement&, FFDDistanceGridCollisionElement&);
    bool testIntersection(FFDDistanceGridCollisionElement&, Point&);
    template<class Sphere>
    bool testIntersection(FFDDistanceGridCollisionElement&, Sphere&);
    bool testIntersection(FFDDistanceGridCollisionElement&, Triangle&);
    bool testIntersection(FFDDistanceGridCollisionElement&, Ray&);

    int computeIntersection(Cube&, Cube&, DetectionOutputVector&);
    template<class Sphere>
    int computeIntersection(Sphere&, Sphere&, DetectionOutputVector&);
    template<class Sphere>
    int computeIntersection(Sphere&, Cube&, DetectionOutputVector&);
    template<class Sphere>
    int computeIntersection(Sphere&, Ray&, DetectionOutputVector&);
    template<class Sphere>
    int computeIntersection(Sphere&, Triangle&, DetectionOutputVector&);
    //int computeIntersection(Triangle&, Triangle&, DetectionOutputVector&);
    int computeIntersection(RigidDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&, DetectionOutputVector&);
    int computeIntersection(RigidDistanceGridCollisionElement&, Point&, DetectionOutputVector&);
    template<class Sphere>
    int computeIntersection(RigidDistanceGridCollisionElement&, Sphere&, DetectionOutputVector&);
    int computeIntersection(RigidDistanceGridCollisionElement&, Triangle&, DetectionOutputVector&);
    int computeIntersection(RigidDistanceGridCollisionElement&, Ray&, DetectionOutputVector&);
    int computeIntersection(FFDDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&, DetectionOutputVector&);
    int computeIntersection(FFDDistanceGridCollisionElement&, FFDDistanceGridCollisionElement&, DetectionOutputVector&);
    int computeIntersection(FFDDistanceGridCollisionElement&, Point&, DetectionOutputVector&);
    template<class Sphere>
    int computeIntersection(FFDDistanceGridCollisionElement&, Sphere&, DetectionOutputVector&);
    int computeIntersection(FFDDistanceGridCollisionElement&, Triangle&, DetectionOutputVector&);
    int computeIntersection(FFDDistanceGridCollisionElement&, Ray&, DetectionOutputVector&);

};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
