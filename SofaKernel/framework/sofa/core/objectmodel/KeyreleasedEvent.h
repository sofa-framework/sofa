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
#ifndef SOFA_COMPONENT_HAND_KEYRELEASEDEVENT_H
#define SOFA_COMPONENT_HAND_KEYRELEASEDEVENT_H

#include <sofa/core/objectmodel/Event.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

/**
	@author Juan Pablo de la Plata
	@brief This event is propagated along the objects when a key on the keyboard is released.
*/
class SOFA_CORE_API KeyreleasedEvent : public core::objectmodel::Event
{
public:

    SOFA_EVENT_H( KeyreleasedEvent )

    /// Constructor
    KeyreleasedEvent( char );
    /// Destructor
    ~KeyreleasedEvent();
    /// Return the key released
    char getKey() const;

    virtual const char* getClassName() const { return "KeyreleasedEvent"; }
protected:
    /// Store the key
    char m_char;
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
