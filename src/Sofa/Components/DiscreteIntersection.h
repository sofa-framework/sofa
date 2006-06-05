#ifndef SOFA_COMPONENTS_DISCRETEINTERSECTION_H
#define SOFA_COMPONENTS_DISCRETEINTERSECTION_H

#include "Collision/Intersection.h"
#include "Common/FnDispatcher.h"
#include "Sphere.h"
#include "Triangle.h"
#include "Cube.h"
#include "Ray.h"

namespace Sofa
{

namespace Components
{

class DiscreteIntersection : public Collision::Intersection
{
public:
    DiscreteIntersection();

    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present)
    virtual bool canIntersect(Abstract::CollisionElement* elem1, Abstract::CollisionElement* elem2);

    /// Compute the intersection between 2 elements.
    virtual Collision::DetectionOutput* intersect(Abstract::CollisionElement* elem1, Abstract::CollisionElement* elem2);

protected:
    FnDispatcher<Abstract::CollisionElement, bool> fnCollisionDetection;
    FnDispatcher<Abstract::CollisionElement, Collision::DetectionOutput*> fnCollisionDetectionOutput;
};

// Jeremie A. : put the methods inside a namespace instead of a class,
// for g++ 3.4 compatibility

namespace DiscreteIntersections
{

bool intersectionCubeCube(Cube& ,Cube&);

bool intersectionSphereSphere(Sphere & ,Sphere &);
//bool intersectionSphereTriangle(Sphere &, Triangle &);
bool intersectionTriangleTriangle(Triangle& ,Triangle&);
bool intersectionSphereRay(Sphere & ,Ray &);

Collision::DetectionOutput* distCorrectionSphereSphere(Sphere & ,Sphere &);
Collision::DetectionOutput* distCorrectionSphereRay(Sphere & ,Ray &);
//Collision::DetectionOutput* distCorrectionSphereTriangle(Sphere &, Triangle &);
Collision::DetectionOutput* distCorrectionTriangleTriangle (Triangle& ,Triangle&);

} // namespace DiscreteIntersections

} // namespace Components

} // namespace Sofa

#endif
