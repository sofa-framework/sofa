/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "ARTrackEvent.h"

namespace sofa
{

namespace core
{

namespace objectmodel
{
    
SOFA_EVENT_CPP(ARTrackEvent)

ARTrackEvent::ARTrackEvent(const Vector3& position, const Quat& orientation, const sofa::helper::fixed_array<double,3>& angles, const sofa::helper::fixed_array<Vector3,3>& fingersPosition)
    : sofa::core::objectmodel::Event()
    , m_position(position)
    , m_orientation(orientation)
    , m_angles(angles)
    , m_fingersPosition(fingersPosition)
{}

const Vector3 ARTrackEvent::getPosition() const
{
    return m_position;
}

const Quat ARTrackEvent::getOrientation() const
{
    return m_orientation;
}

const sofa::helper::fixed_array<double,3> ARTrackEvent::getAngles() const
{
    return m_angles;
}

const Vector3 ARTrackEvent::getFingerposition(const unsigned int i) const
{
    return m_fingersPosition[i];
}

} // namespace tree

} // namespace simulation

} // namespace sofa
