#ifndef SOFA_COMPONENT_COLLISION_BASEPROXIMITYINTERSECTION_H
#define SOFA_COMPONENT_COLLISION_BASEPROXIMITYINTERSECTION_H

#include <sofa/component/collision/DiscreteIntersection.h>
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
    Data<double> alarmDistance;
    Data<double> contactDistance;
protected:
    BaseProximityIntersection();
    virtual ~BaseProximityIntersection() { }
public:
    virtual void init()=0;

    /// Returns true if algorithm uses proximity
    virtual bool useProximity() const { return true; }

    /// Returns the alarm distance (must returns 0 if useProximity() is false)
    double getAlarmDistance() const { return alarmDistance.getValue(); }

    /// Returns the contact distance (must returns 0 if useProximity() is false)
    double getContactDistance() const { return contactDistance.getValue(); }

    /// Sets the alarm distance (if useProximity() is false, the alarm distance is equal to 0)
    void setAlarmDistance(double v) { alarmDistance.setValue(v); }

    /// Sets the contact distance (if useProximity() is false, the contact distance is equal to 0)
    void setContactDistance(double v) { contactDistance.setValue(v); }
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_COLLISION_BASEPROXIMITYINTERSECTION_H
