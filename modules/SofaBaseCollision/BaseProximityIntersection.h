#ifndef SOFA_COMPONENT_COLLISION_BASEPROXIMITYINTERSECTION_H
#define SOFA_COMPONENT_COLLISION_BASEPROXIMITYINTERSECTION_H

#include <SofaBaseCollision/DiscreteIntersection.h>
#include <SofaBaseCollision/BaseIntTool.h>
#include <sofa/helper/FnDispatcher.h>

namespace sofa
{

namespace component
{

namespace collision
{

class SOFA_BASE_COLLISION_API BaseProximityIntersection : public DiscreteIntersection
{
public:
    SOFA_ABSTRACT_CLASS(BaseProximityIntersection,DiscreteIntersection);
    Data<SReal> alarmDistance;
    Data<SReal> contactDistance;
protected:
    BaseProximityIntersection();
    virtual ~BaseProximityIntersection() { }
public:
    /// Returns true if algorithm uses proximity
    virtual bool useProximity() const { return true; }

    /// Returns the alarm distance (must returns 0 if useProximity() is false)
    SReal getAlarmDistance() const { return alarmDistance.getValue(); }

    /// Returns the contact distance (must returns 0 if useProximity() is false)
    SReal getContactDistance() const { return contactDistance.getValue(); }

    /// Sets the alarm distance (if useProximity() is false, the alarm distance is equal to 0)
    void setAlarmDistance(SReal v) { alarmDistance.setValue(v); }

    /// Sets the contact distance (if useProximity() is false, the contact distance is equal to 0)
    void setContactDistance(SReal v) { contactDistance.setValue(v); }
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_COLLISION_BASEPROXIMITYINTERSECTION_H
