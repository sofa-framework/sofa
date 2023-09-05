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

#include <sofa/core/objectmodel/Event.h>
#include <sofa/type/Quat.h>
#include <sofa/type/Vec.h>

namespace sofa::core::objectmodel
{

/**
 * @brief This event notifies about haptic device interaction.
 */
class SOFA_CORE_API HapticDeviceEvent : public sofa::core::objectmodel::Event
{
public:

    SOFA_EVENT_H( HapticDeviceEvent )

    /**
     * @brief Define the device state (which button is pressed).
     */
    typedef enum
    {
        Button1StateMask=(1<<0),
        Button2StateMask=(1<<1),
        Button3StateMask=(1<<2),
        Button4StateMask=(1<<3),
        Button5StateMask=(1<<4),
        Button6StateMask=(1<<5),
        Button7StateMask=(1<<6),
        Button8StateMask=(1<<7)
    } ButtonsStateMask;

    /**
     * @brief Constructor.
     */
    HapticDeviceEvent(const unsigned int id, const sofa::type::Vec3& position, const sofa::type::Quat<SReal>& orientation, const unsigned char button);

    /**
     * @brief Destructor.
     */
    ~HapticDeviceEvent() override;

    /**
     * @brief Get the device X coordinate
     */
    SReal getPosX(void) const {return m_position[0];}

    /**
     * @brief Get the device Y coordinate
     */
    SReal getPosY(void) const {return m_position[1];}

    /**
     * @brief Get the device Z coordinate
     */
    SReal getPosZ(void) const {return m_position[2];}

    /**
     * @brief Get the device coordinates.
     */
    sofa::type::Vec3 getPosition(void) const {return m_position;}

    /**
     * @brief Get the device orientation.
     */
    sofa::type::Quat<SReal> getOrientation(void) const {return m_orientation;}

    /**
     * @brief Get the device button state.
     */
    unsigned char getButtonState() const {return m_buttonState;}

    bool getButton(const int id = 0) const {return bool(((m_buttonState >> id) & 1));}

    /**
     * @brief Get the device Id.
     */
    unsigned int getDeviceId() const {return m_deviceId;}

    inline static const char* GetClassName() { return "HapticDeviceEvent"; }
private:

    unsigned int	m_deviceId;
    sofa::type::Vec3 m_position; ///< Haptic device coordinates in 3D space.
    sofa::type::Quat<SReal> m_orientation; ///< Haptic device orientation.
    unsigned char m_buttonState;
};
} // namespace sofa::core::objectmodel
