#ifndef SOFA_COMPONENT_COLLISION_MINPROXIMITYINTERSECTION_H
#define SOFA_COMPONENT_COLLISION_MINPROXIMITYINTERSECTION_H

#include <sofa/component/collision/DiscreteIntersection.h>
#include <sofa/helper/FnDispatcher.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/LineModel.h>
#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/component/collision/RayModel.h>

namespace sofa
{

namespace component
{

namespace collision
{

class MinProximityIntersection : public DiscreteIntersection
{
public:
    MinProximityIntersection(bool useSphereTriangle = true
                            );

    /// Return the intersector class handling the given pair of collision models, or NULL if not supported.
    virtual core::componentmodel::collision::ElementIntersector* findIntersector(core::CollisionModel* object1, core::CollisionModel* object2);

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

core::componentmodel::collision::DetectionOutput* distCorrectionCubeCube(Cube&, Cube&);
core::componentmodel::collision::DetectionOutput* distCorrectionSphereSphere(Sphere&, Sphere&);
core::componentmodel::collision::DetectionOutput* distCorrectionSphereTriangle(Sphere&, Triangle&);
core::componentmodel::collision::DetectionOutput* distCorrectionSphereLine(Sphere&, Line&);
core::componentmodel::collision::DetectionOutput* distCorrectionSpherePoint(Sphere&, Point&);
core::componentmodel::collision::DetectionOutput* distCorrectionSphereRay(Sphere&, Ray&);
core::componentmodel::collision::DetectionOutput* distCorrectionPointTriangle(Point&, Triangle&);
core::componentmodel::collision::DetectionOutput* distCorrectionLineLine(Line&, Line&);
core::componentmodel::collision::DetectionOutput* distCorrectionPointLine(Point&, Line&);
core::componentmodel::collision::DetectionOutput* distCorrectionPointPoint(Point&, Point&);
core::componentmodel::collision::DetectionOutput* distCorrectionRayTriangle(Ray&, Triangle&);

} // namespace MinProximityIntersections

} // namespace collision

} // namespace component

} // namespace sofa

#endif
