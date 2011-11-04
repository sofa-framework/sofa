/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
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

using namespace sofa::defaulttype;

/**
 * @brief This event notifies about haptic device interaction.
 */
class SOFA_CORE_API HapticDeviceEvent : public sofa::core::objectmodel::Event
{
public:

    /**
     * @brief Define the device state (which button is pressed).
     */
    typedef enum
    {
        Button1=1,
        Button2=2,
        Button3=3,
        Button4=4,
        Button5=5,
        Button6=6,
        Button7=7,
        Button8=8,
    } State;

    /**
     * @brief Constructor.
     */
    HapticDeviceEvent(const unsigned int id, const Vector3& position, const Quat& orientation, const unsigned char button);

    /**
     * @brief Destructor.
     */
    virtual ~HapticDeviceEvent();

    /**
     * @brief Get the device X coordinate
     */
    double getPosX(void) const {return m_position[0];};

    /**
     * @brief Get the device Y coordinate
     */
    double getPosY(void) const {return m_position[1];};

    /**
     * @brief Get the device Z coordinate
     */
    double getPosZ(void) const {return m_position[2];};

    /**
     * @brief Get the device coordinates.
     */
    Vector3 getPosition(void) const {return m_position;}

    /**
     * @brief Get the device orientation.
     */
    Quat getOrientation(void) const {return m_orientation;}

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
    Vector3 m_position; ///< Haptic device coordinates in a Vec3d type.
    Quat m_orientation; ///< Haptic device orientation.
    unsigned char m_buttonState;
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_OBJECTMODEL_MOUSEEVENT_H
