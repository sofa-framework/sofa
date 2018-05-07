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
#ifndef SOFA_CORE_OBJECTMODEL_HAPTICDEVICEEVENT_H
#define SOFA_CORE_OBJECTMODEL_HAPTICDEVICEEVENT_H

#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/Quat.h>

namespace sofa
{

namespace core
{

namespace objectmodel
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
    HapticDeviceEvent(const unsigned int id, const sofa::defaulttype::Vector3& position, const sofa::defaulttype::Quat& orientation, const unsigned char button);

    /**
     * @brief Destructor.
     */
    virtual ~HapticDeviceEvent();

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
    sofa::defaulttype::Vector3 getPosition(void) const {return m_position;}

    /**
     * @brief Get the device orientation.
     */
    sofa::defaulttype::Quat getOrientation(void) const {return m_orientation;}

    /**
     * @brief Get the device button state.
     */
    unsigned char getButtonState() const {return m_buttonState;}

    bool getButton(const int id = 0) const {return (bool) ((m_buttonState >> id) & 1);}

    /**
     * @brief Get the device Id.
     */
    unsigned int getDeviceId() const {return m_deviceId;}

    virtual const char* getClassName() const { return "HapticDeviceEvent"; }
private:

    unsigned int	m_deviceId;
    sofa::defaulttype::Vector3 m_position; ///< Haptic device coordinates in a Vector3 type.
    sofa::defaulttype::Quat m_orientation; ///< Haptic device orientation.
    unsigned char m_buttonState;
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_OBJECTMODEL_MOUSEEVENT_H
