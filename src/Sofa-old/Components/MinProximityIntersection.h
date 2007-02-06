#ifndef SOFA_COMPONENTS_MINPROXIMITYINTERSECTION_H
#define SOFA_COMPONENTS_MINPROXIMITYINTERSECTION_H

#include "DiscreteIntersection.h"
#include "Common/FnDispatcher.h"
#include "SphereModel.h"
#include "TriangleModel.h"
#include "LineModel.h"
#include "PointModel.h"
#include "CubeModel.h"
#include "RayModel.h"

namespace Sofa
{

namespace Components
{

class MinProximityIntersection : public DiscreteIntersection
{
public:
    MinProximityIntersection(bool useSphereTriangle = true
                            );

    /// Return the intersector class handling the given pair of collision models, or NULL if not supported.
    virtual Collision::ElementIntersector* findIntersector(Abstract::CollisionModel* object1, Abstract::CollisionModel* object2);

    /// returns true if algorithm uses continous detection
    virtual bool useProximity() const { return true; }

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

Collision::DetectionOutput* distCorrectionCubeCube(Cube&, Cube&);
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
