#ifndef SOFA_COMPONENTS_DISCRETEINTERSECTION_H
#define SOFA_COMPONENTS_DISCRETEINTERSECTION_H

#include "Collision/Intersection.h"
#include "Common/FnDispatcher.h"
#include "SphereModel.h"
#include "TriangleModel.h"
#include "CubeModel.h"
#include "RayModel.h"
#include "SphereTreeModel.h"

namespace Sofa
{

namespace Components
{

class DiscreteIntersection : public Collision::Intersection
{
public:
    DiscreteIntersection();

    /// Return the intersector class handling the given pair of collision models, or NULL if not supported.
    virtual Collision::ElementIntersector* findIntersector(Abstract::CollisionModel* object1, Abstract::CollisionModel* object2);

protected:
    Collision::IntersectorMap intersectors;
};

// Jeremie A. : put the methods inside a namespace instead of a class,
// for g++ 3.4 compatibility

namespace DiscreteIntersections
{

bool intersectionCubeCube(Cube&, Cube&);
bool intersectionSphereSphere(Sphere&, Sphere&);
bool intersectionSphereRay(Sphere&, Ray&);


bool intersectionSingleSphereSingleSphere(SingleSphere&, SingleSphere&);

bool intersectionSingleSphereRay(SingleSphere&, Ray&);

//bool intersectionSphereTriangle(Sphere& , Triangle&);
//bool intersectionTriangleTriangle(Triangle& ,Triangle&);

Collision::DetectionOutput* distCorrectionCubeCube(Cube&, Cube&);
Collision::DetectionOutput* distCorrectionSphereSphere(Sphere&, Sphere&);
Collision::DetectionOutput* distCorrectionSingleSphereSingleSphere(SingleSphere&, SingleSphere&);
Collision::DetectionOutput* distCorrectionSphereRay(Sphere&, Ray&);
Collision::DetectionOutput* distCorrectionSingleSphereRay(SingleSphere&, Ray&);
//Collision::DetectionOutput* distCorrectionSphereTriangle(Sphere&, Triangle&);
//Collision::DetectionOutput* distCorrectionTriangleTriangle (Triangle&, Triangle&);

} // namespace DiscreteIntersections

} // namespace Components

} // namespace Sofa

#endif
