/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_OBJECTMODEL_OMNIEVENT_H
#define SOFA_CORE_OBJECTMODEL_OMNIEVENT_H

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
 * @brief This event notifies about SensAble PHANTOM® device interaction.
 */
class SOFA_CORE_API OmniEvent : public sofa::core::objectmodel::Event
{
public:

    /**
     * @brief Define the SensAble PHANTOM® state (which button is pressed).
     */
    typedef enum
    {
        Button1=1,
        Button2=2
    } State;

    /**
     * @brief Constructor.
     */
    OmniEvent(const unsigned int id, const Vector3& position, const Quat& orientation, const unsigned char button);

    /**
     * @brief Destructor.
     */
    virtual ~OmniEvent();

    /**
     * @brief Get the PHANTOM® X coordinate
     */
    double getPosX(void) const {return m_position[0];};

    /**
     * @brief Get the PHANTOM® Y coordinate
     */
    double getPosY(void) const {return m_position[1];};

    /**
     * @brief Get the PHANTOM® Z coordinate
     */
    double getPosZ(void) const {return m_position[2];};

    /**
     * @brief Get the PHANTOM® coordinates.
     */
    Vector3 getPosition(void) const {return m_position;}

    /**
     * @brief Get the PHANTOM® orientation.
     */
    Quat getOrientation(void) const {return m_orientation;}

    /**
     * @brief Get the PHANTOM® button state.
     */
    unsigned char getButtonState() const {return m_buttonState;}

    bool getButton(const int id = 0) const {return (bool) ((m_buttonState >> id) & 1);}

    virtual const char* getClassName() const { return "OmniEvent"; }
private:

    unsigned char	m_deviceId;
    Vector3 m_position; ///< SensAble PHANTOM® coordinates in a Vec3d type.
    Quat m_orientation; ///< SensAble PHANTOM® orientation.
    unsigned char m_buttonState;
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_OBJECTMODEL_MOUSEEVENT_H
