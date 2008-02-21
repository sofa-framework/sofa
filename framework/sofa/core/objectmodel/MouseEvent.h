/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
//
// C++ Interface: MouseEvent
//
// Description:
//
//
// Author: Pierre-Jean Bensoussan, Digital Trainers (2008)
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_CORE_OBJECTMODEL_MOUSEEVENT_H
#define SOFA_CORE_OBJECTMODEL_MOUSEEVENT_H

#include <sofa/core/objectmodel/Event.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

class MouseEvent : public sofa::core::objectmodel::Event
{
public:
    enum EventType { Move=0, LeftPressed, LeftReleased, RightPressed, RightReleased, Wheel, Reset };

    MouseEvent(EventType mouseEvent, int mouseWheelDelta=0);
    MouseEvent(EventType mouseEvent, int posX, int posY);

    virtual ~MouseEvent();

    int getPosX(void) const {return _posX;};
    int getPosY(void) const {return _posY;};
    int getWheelDelta(void) const {return _mouseWheelDelta;};
    EventType getEventType(void) const {return _eventType;};

private:

    EventType _eventType;
    int _mouseWheelDelta;
    int _posX, _posY;
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_OBJECTMODEL_MOUSEEVENT_H
