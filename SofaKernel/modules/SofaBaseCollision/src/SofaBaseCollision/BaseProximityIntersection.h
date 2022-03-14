/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <SofaBaseCollision/config.h>

#include <SofaBaseCollision/DiscreteIntersection.h>

namespace sofa::component::collision
{

/**
 * Base class for intersections methods using proximities.
 * It introduces Datas for the alarm distance and contact distance.
 * Cubes intersection is modified to use proximities.
 */
class SOFA_SOFABASECOLLISION_API BaseProximityIntersection : public DiscreteIntersection
{
public:
    SOFA_ABSTRACT_CLASS(BaseProximityIntersection,DiscreteIntersection);
    Data<SReal> alarmDistance; ///< Proximity detection distance
    Data<SReal> contactDistance; ///< Distance below which a contact is created
protected:
    BaseProximityIntersection();
    ~BaseProximityIntersection() override { }
public:
    /// Returns true if algorithm uses proximity
    bool useProximity() const override { return true; }

    /// Returns the alarm distance (must returns 0 if useProximity() is false)
    SReal getAlarmDistance() const override { return alarmDistance.getValue(); }

    /// Returns the contact distance (must returns 0 if useProximity() is false)
    SReal getContactDistance() const override { return contactDistance.getValue(); }

    /// Sets the alarm distance (if useProximity() is false, the alarm distance is equal to 0)
    void setAlarmDistance(SReal v) override { alarmDistance.setValue(v); }

    /// Sets the contact distance (if useProximity() is false, the contact distance is equal to 0)
    void setContactDistance(SReal v) override { contactDistance.setValue(v); }

    /// Intersectors for cubes using proximities
    bool testIntersection(Cube& cube1, Cube& cube2) override;
    int computeIntersection(Cube& cube1, Cube& cube2, OutputVector* contacts) override;

};

} // namespace sofa::component::collision