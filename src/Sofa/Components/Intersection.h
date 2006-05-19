#ifndef SOFA_COMPONENTS_INTERSECTION_H
#define SOFA_COMPONENTS_INTERSECTION_H

#include "Sofa/Abstract/CollisionModel.h"
#include "Collision/DetectionOutput.h"
#include "Sphere.h"
#include "Triangle.h"
#include "Cube.h"
#include "Ray.h"

namespace Sofa
{

namespace Components
{

// Jeremie A. : put the methods inside a namespace instead of a class,
// for g++ 3.4 compatibility

namespace Intersections
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

class Intersection
{
protected:
    Intersection();
    static Intersection instance;
};

} // namespace Intersections

} // namespace Components

} // namespace Sofa

#endif
