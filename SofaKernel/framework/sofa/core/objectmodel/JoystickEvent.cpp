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

// Author: Pierre-Jean Bensoussan, Digtal Trainers, (C) 2008

#include <sofa/core/objectmodel/JoystickEvent.h>
#include <cassert>
#include <cstring> // for NULL

namespace sofa
{

namespace core
{

namespace objectmodel
{


SOFA_EVENT_CPP( JoystickEvent )

/*************
 * AxisEvent
 *************/

JoystickEvent::AxisEvent::AxisEvent( const int index, const float value )
    : m_index( index )
    , m_value( value )
{
    assert( (value >= -1.f) && (value <= 1.f) && "Value must be in range [-1, 1]" );
}



int JoystickEvent::AxisEvent::getIndex() const
{
    return m_index;
}



float JoystickEvent::AxisEvent::getValue() const
{
    return m_value;
}



/**************
 * ButtonEvent
 **************/

JoystickEvent::ButtonEvent::ButtonEvent( const int buttons )
{
    setButtons(buttons);
}



void JoystickEvent::ButtonEvent::setButtons(const int buttons)
{
    int mask = 0x1;
    for(int i = 0; i < 32; ++i)
    {
        m_buttons[i] = ((buttons & mask) != 0);
        mask *= 2;
    }
}



bool JoystickEvent::ButtonEvent::getButton(const int i) const
{
    assert((i < 32) && "Number of joystick button out of range");
    return m_buttons[i];
}



/***********
 * HatEvent
 ***********/

JoystickEvent::HatEvent::HatEvent(const int index, const State state )
    : m_index( index )
    , m_state( state )
{
}



int JoystickEvent::HatEvent::getIndex() const
{
    return m_index;
}



JoystickEvent::HatEvent::State JoystickEvent::HatEvent::getState() const
{
    return m_state;
}



/****************
 * JoystickEvent
 ***************/

JoystickEvent::JoystickEvent()
{
    buttonEvent = 0;
}



JoystickEvent::~JoystickEvent()
{
    if (buttonEvent) delete buttonEvent;

    for (unsigned int i=0; i < axisEvents.size(); i++)
        delete axisEvents[i];

    for (unsigned int i=0; i < hatEvents.size(); i++)
        delete hatEvents[i];
}

const JoystickEvent::AxisEvent *JoystickEvent::getAxisEvent(const int index) const
{
    assert( (index < (int)axisEvents.size()) && "AxisEvents index out of range" );

    for (unsigned int i = 0; i < axisEvents.size(); i++)
    {
        if (axisEvents[i]->getIndex() == index)
            return axisEvents[i];
    }

    return NULL;
}



const std::vector<JoystickEvent::AxisEvent*> &JoystickEvent::getAxisEvents(void) const
{
    return axisEvents;
}



unsigned int JoystickEvent::getAxisEventsSize(void) const
{
    return (unsigned int) axisEvents.size();
}



void JoystickEvent::addAxisEvent( JoystickEvent::AxisEvent * aEvent)
{
    axisEvents.push_back(aEvent);
}



JoystickEvent::ButtonEvent *JoystickEvent::getButtonEvent(void) const
{
    return buttonEvent;
}



bool JoystickEvent::getButton(unsigned int buttonIndex) const
{
    if (getButtonEvent())
        return getButtonEvent()->getButton(buttonIndex);

    return false;
}



void JoystickEvent::setButtonEvent( JoystickEvent::ButtonEvent * bEvent)
{
    buttonEvent = bEvent;
}



const JoystickEvent::HatEvent *JoystickEvent::getHatEvent(const int index) const
{
    assert( (index < (int)hatEvents.size()) && "HatEvents index out of range" );

    for (unsigned int i = 0; i < hatEvents.size(); i++)
    {
        if (hatEvents[i]->getIndex() == index)
            return hatEvents[i];
    }

    return 0l;
}



const std::vector<JoystickEvent::HatEvent *> &JoystickEvent::getHatEvents(void) const
{
    return hatEvents;
}



unsigned int JoystickEvent::getHatEventsSize(void) const
{
    return (unsigned int) hatEvents.size();
}



void JoystickEvent::addHatEvent( JoystickEvent::HatEvent * hEvent)
{
    hatEvents.push_back(hEvent);
}

} // namespace tree

} // namespace simulation

} // namespace sofa
