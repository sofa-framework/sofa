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
#ifndef SOFA_COMPONENT_COLLISION_DISCRETEINTERSECTION_H
#define SOFA_COMPONENT_COLLISION_DISCRETEINTERSECTION_H

#include <sofa/core/collision/Intersection.h>
#include <sofa/helper/FnDispatcher.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/LineModel.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/component/collision/RayModel.h>
#include <sofa/component/collision/SphereTreeModel.h>
#include <sofa/component/collision/DistanceGridCollisionModel.h>
#include <sofa/component/collision/TetrahedronModel.h>

namespace sofa
{

namespace component
{

namespace collision
{
class SOFA_BASE_COLLISION_API DiscreteIntersection : public core::collision::Intersection
{
public:
    SOFA_CLASS(DiscreteIntersection,sofa::core::collision::Intersection);

    DiscreteIntersection();

    /// Return the intersector class handling the given pair of collision models, or NULL if not supported.
    /// @param swapModel output value set to true if the collision models must be swapped before calling the intersector.
    virtual core::collision::ElementIntersector* findIntersector(core::CollisionModel* object1, core::CollisionModel* object2, bool& swapModels);

protected:
    core::collision::IntersectorMap intersectors;

public:
    template<class Model1, class Model2>
    sofa::core::collision::TDetectionOutputVector<Model1,Model2>* createOutputVector(Model1*, Model2*)
    {
        return new sofa::core::collision::TDetectionOutputVector<Model1,Model2>;
    }

    template<class Model1, class Model2>
    sofa::core::collision::TDetectionOutputVector<Model1,Model2>* getOutputVector(Model1*, Model2*, sofa::core::collision::DetectionOutputVector* contacts)
    {
        return static_cast<sofa::core::collision::TDetectionOutputVector<Model1,Model2>*>(contacts);
    }

    typedef sofa::helper::vector<sofa::core::collision::DetectionOutput> OutputVector;

    int beginIntersection(sofa::core::CollisionModel* /*model1*/, sofa::core::CollisionModel* /*model2*/, OutputVector* /*contacts*/)
    {
        return 0;
    }

    int endIntersection(sofa::core::CollisionModel* /*model1*/, sofa::core::CollisionModel* /*model2*/, OutputVector* /*contacts*/)
    {
        return 0;
    }

    bool testIntersection(Cube&, Cube&);

    template<class Sphere>
    bool testIntersection(Sphere&, Sphere&);
    template<class Sphere>
    bool testIntersection(Sphere&, Cube&);
    template<class Sphere>
    bool testIntersection(Ray&, Sphere&);
    template<class Sphere>
    bool testIntersection(Sphere&, Triangle&);
    //bool testIntersection(Triangle&, Triangle&);
    bool testIntersection(Triangle&, Line&);
    bool testIntersection(Ray&, Triangle&);
    bool testIntersection(Ray&, Tetrahedron&);
    bool testIntersection(Tetrahedron&, Point&);
    bool testIntersection(RigidDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&);
    bool testIntersection(RigidDistanceGridCollisionElement&, Point&);
    template<class Sphere>
    bool testIntersection(RigidDistanceGridCollisionElement&, Sphere&);
    bool testIntersection(RigidDistanceGridCollisionElement&, Triangle&);
    bool testIntersection(Ray&, RigidDistanceGridCollisionElement&);
    bool testIntersection(FFDDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&);
    bool testIntersection(FFDDistanceGridCollisionElement&, FFDDistanceGridCollisionElement&);
    bool testIntersection(FFDDistanceGridCollisionElement&, Point&);
    template<class Sphere>
    bool testIntersection(FFDDistanceGridCollisionElement&, Sphere&);
    bool testIntersection(FFDDistanceGridCollisionElement&, Triangle&);
    bool testIntersection(Ray&, FFDDistanceGridCollisionElement&);

    int computeIntersection(Cube&, Cube&, OutputVector*);
    template<class Sphere>
    int computeIntersection(Sphere&, Sphere&, OutputVector*);
    template<class Sphere>
    int computeIntersection(Sphere&, Cube&, OutputVector*);
    template<class Sphere>
    int computeIntersection(Ray&, Sphere&, OutputVector*);
    template<class Sphere>
    int computeIntersection(Sphere&, Triangle&, OutputVector*);
    //int computeIntersection(Triangle&, Triangle&, OutputVector*);
    int computeIntersection(Triangle&, Line&, OutputVector*);
    int computeIntersection(Ray&, Triangle&, OutputVector*);
    int computeIntersection(Ray&, Tetrahedron&, OutputVector*);
    int computeIntersection(Tetrahedron&, Point&, OutputVector*);
    int computeIntersection(RigidDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&, OutputVector*);
    int computeIntersection(RigidDistanceGridCollisionElement&, Point&, OutputVector*);
    template<class Sphere>
    int computeIntersection(RigidDistanceGridCollisionElement&, Sphere&, OutputVector*);
    int computeIntersection(RigidDistanceGridCollisionElement&, Triangle&, OutputVector*);
    int computeIntersection(Ray&, RigidDistanceGridCollisionElement&, OutputVector*);
    int computeIntersection(FFDDistanceGridCollisionElement&, RigidDistanceGridCollisionElement&, OutputVector*);
    int computeIntersection(FFDDistanceGridCollisionElement&, FFDDistanceGridCollisionElement&, OutputVector*);
    int computeIntersection(FFDDistanceGridCollisionElement&, Point&, OutputVector*);
    template<class Sphere>
    int computeIntersection(FFDDistanceGridCollisionElement&, Sphere&, OutputVector*);
    int computeIntersection(FFDDistanceGridCollisionElement&, Triangle&, OutputVector*);
    int computeIntersection(Ray&, FFDDistanceGridCollisionElement&, OutputVector*);


};

} // namespace collision

} // namespace component

} // namespace sofa

#endif
