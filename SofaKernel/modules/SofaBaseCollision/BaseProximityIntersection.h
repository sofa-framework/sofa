/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_BASEPROXIMITYINTERSECTION_H
#define SOFA_COMPONENT_COLLISION_BASEPROXIMITYINTERSECTION_H
#include "config.h"

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
