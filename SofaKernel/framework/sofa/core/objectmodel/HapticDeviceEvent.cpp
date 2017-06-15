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
#include <sofa/core/objectmodel/HapticDeviceEvent.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

SOFA_EVENT_CPP( HapticDeviceEvent )

HapticDeviceEvent::HapticDeviceEvent(const unsigned int id, const sofa::defaulttype::Vector3& position, const sofa::defaulttype::Quat& orientation, const unsigned char button)
    : sofa::core::objectmodel::Event()
    , m_deviceId(id)
    , m_position(position)
    , m_orientation(orientation)
    , m_buttonState(button)
{}

HapticDeviceEvent::~HapticDeviceEvent()
{

}

} // namespace tree

} // namespace simulation

} // namespace sofa
