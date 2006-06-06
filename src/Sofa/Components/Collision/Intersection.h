#ifndef SOFA_COMPONENTS_COLLISION_INTERSECTION_H
#define SOFA_COMPONENTS_COLLISION_INTERSECTION_H

#include "Sofa/Abstract/CollisionModel.h"
#include "DetectionOutput.h"

namespace Sofa
{

namespace Components
{

namespace Collision
{

class Intersection : public virtual Abstract::BaseObject
{
public:
    virtual ~Intersection() { }

    /// Test if 2 elements can collide. Note that this can be conservative (i.e. return true even when no collision is present)
    virtual bool canIntersect(Abstract::CollisionElement* elem1, Abstract::CollisionElement* elem2) = 0;

    /// Compute the intersection between 2 elements.
    virtual Collision::DetectionOutput* intersect(Abstract::CollisionElement* elem1, Abstract::CollisionElement* elem2) = 0;

    /// returns true if algorithm uses proximity detection
    virtual bool useProximity() const { return false; }

    /// returns true if algorithm uses continous detection
    virtual bool useContinuous() const { return false; }

    /// Return the alarm distance (must return 0 if useProximity() is false)
    virtual double getAlarmDistance() const { return 0.0; }

    /// Return the contact distance (must return 0 if useProximity() is false)
    virtual double getContactDistance() const { return 0.0; }

};

} // namespace Collision

} // namespace Components

} // namespace Sofa

#endif
