#ifndef SOFA_COMPONENTS_MINPROXIMITYINTERSECTION_H
#define SOFA_COMPONENTS_MINPROXIMITYINTERSECTION_H

#include "DiscreteIntersection.h"
#include "Common/FnDispatcher.h"
#include "Sphere.h"
#include "Triangle.h"
#include "Line.h"
#include "Point.h"
#include "Cube.h"
#include "Ray.h"

namespace Sofa
{

namespace Components
{

class MinProximityIntersection : public DiscreteIntersection
{
public:
    MinProximityIntersection(bool useSphereTriangle = true
                            );

    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present)
    virtual bool canIntersect(Abstract::CollisionElement* elem1, Abstract::CollisionElement* elem2);

    /// Compute the intersection between 2 elements.
    virtual Collision::DetectionOutput* intersect(Abstract::CollisionElement* elem1, Abstract::CollisionElement* elem2);

    /// returns true if algorithm uses continous detection
    virtual bool useMinProximity() const { return true; }

    /// Return the alarm distance (must return 0 if useMinProximity() is false)
    double getAlarmDistance() const { return alarmDistance; }

    /// Return the contact distance (must return 0 if useMinProximity() is false)
    double getContactDistance() const { return contactDistance; }

    void setAlarmDistance(double v) { alarmDistance = v; }

    void setContactDistance(double v) { contactDistance = v; }

protected:
    double alarmDistance;
    double contactDistance;
};

// Jeremie A. : put the methods inside a namespace instead of a class,
// for g++ 3.4 compatibility

namespace MinProximityIntersections
{

bool intersectionCubeCube(Cube& ,Cube&);

bool intersectionSphereSphere(Sphere&, Sphere&);
bool intersectionSphereTriangle(Sphere&, Triangle&);
bool intersectionSphereLine(Sphere&, Line&);
bool intersectionSpherePoint(Sphere&, Point&);
bool intersectionSphereRay(Sphere&, Ray&);
bool intersectionPointTriangle(Point& ,Triangle&);
bool intersectionLineLine(Line&, Line&);
bool intersectionPointLine(Point&, Line&);
bool intersectionPointPoint(Point&, Point&);
bool intersectionRayTriangle(Ray&, Triangle&);

Collision::DetectionOutput* distCorrectionSphereSphere(Sphere&, Sphere&);
Collision::DetectionOutput* distCorrectionSphereTriangle(Sphere&, Triangle&);
Collision::DetectionOutput* distCorrectionSphereLine(Sphere&, Line&);
Collision::DetectionOutput* distCorrectionSpherePoint(Sphere&, Point&);
Collision::DetectionOutput* distCorrectionSphereRay(Sphere&, Ray&);
Collision::DetectionOutput* distCorrectionPointTriangle(Point&, Triangle&);
Collision::DetectionOutput* distCorrectionLineLine(Line&, Line&);
Collision::DetectionOutput* distCorrectionPointLine(Point&, Line&);
Collision::DetectionOutput* distCorrectionPointPoint(Point&, Point&);
Collision::DetectionOutput* distCorrectionRayTriangle(Ray&, Triangle&);

} // namespace MinProximityIntersections

} // namespace Components

} // namespace Sofa

#endif
