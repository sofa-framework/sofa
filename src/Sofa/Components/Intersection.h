#ifndef SOFA_COMPONENTS_INTERSECTION_H
#define SOFA_COMPONENTS_INTERSECTION_H

#include "Sofa/Abstract/CollisionModel.h"
#include "Collision/DetectionOutput.h"
#include "Sphere.h"
//#include "Triangle.h"
#include "Cube.h"

namespace Sofa
{

namespace Components
{

class Intersection
{
protected:
    Intersection();
    static Intersection instance;
public:
    static bool intersectionSphereSphere(Sphere & ,Sphere &);
//	static bool intersectionSphereTriangle(Sphere &, Triangle &);
//	static bool intersectionTriangleTriangle(Triangle& ,Triangle&);
    static bool intersectionCubeCube(Cube& ,Cube&);

    static Collision::DetectionOutput* distCorrectionSphereSphere(Sphere & ,Sphere &);
//	static Collision::DetectionOutput* distCorrectionSphereTriangle(Sphere &, Triangle &);
//	static Collision::DetectionOutput* distCorrectionTriangleTriangle (Triangle& ,Triangle&);
};

} // namespace Components

} // namespace Sofa

#endif
