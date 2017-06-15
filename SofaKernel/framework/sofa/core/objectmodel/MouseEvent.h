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

// Author: Pierre-Jean Bensoussan, Digital Trainers (2008)

#ifndef SOFA_CORE_OBJECTMODEL_MOUSEEVENT_H
#define SOFA_CORE_OBJECTMODEL_MOUSEEVENT_H

#include <sofa/core/objectmodel/Event.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

/**
 * @brief MouseEvent Class
 *
 * Implements an Event that notifies about a Mouse Interaction.
 */
class SOFA_CORE_API MouseEvent : public sofa::core::objectmodel::Event
{
public:

    SOFA_EVENT_H( MouseEvent )

    /**
     * @brief Defines possible Mouse states.
     */
    typedef enum
    {
        Move=0,
        // The standard mouse button (on a three mouse button)
		LeftPressed,
        LeftReleased,
        RightPressed,
        RightReleased,
        MiddlePressed,
        MiddleReleased,

		// In case the mouse has more than three button
		// the extra button will send this event. Unless the 
		// ExtraButton0Pressed lines is properly implemented throughou Sofa. 
		AnyExtraButtonPressed,   
		AnyExtraButtonReleased,

		// Some mice has extra mouse buttons, 
		// TODO: replace the AnyExtraButton events by Button0, Button1 events.
		// and implement the correct processing throughouh Sofa.
		// ExtraButton0Pressed,
		// ExtraButton0Released,
        // ExtraButton1Pressed,
		// ExtraButton1Released,
		
		Wheel,
        Reset
    } State;

    /**
     * @brief Wheel Mouse Event constructor.
     */
    MouseEvent(State state, int wheelDelta = 0);

    /**
     * @brief Default constructor.
     */
    MouseEvent(State state, int posX, int posY);

    /**
     * @brief Default destructor.
     */
    virtual ~MouseEvent();

    /**
     * @name Accessors
     */
    //@{
    int getPosX(void) const {return m_posX;};
    int getPosY(void) const {return m_posY;};
    int getWheelDelta(void) const {return m_wheelDelta;};
    State getState(void) const {return m_state;};
    //}@

    virtual const char* getClassName() const { return "MouseEvent"; }
private:

    State m_state; ///< Mouse State on the event propagation.
    int m_wheelDelta; ///< Mouse wheel delta.
    int m_posX, m_posY; ///< Mouse coordinates.
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_OBJECTMODEL_MOUSEEVENT_H
