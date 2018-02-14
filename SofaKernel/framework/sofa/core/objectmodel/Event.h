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
#ifndef SOFA_CORE_OBJECTMODEL_EVENT_H
#define SOFA_CORE_OBJECTMODEL_EVENT_H

#include <sofa/core/core.h>
#include <stdlib.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

/// this has to be added in the Event class definition (as public)
#define SOFA_EVENT_H(T) \
    protected:\
    static const size_t s_eventTypeIndex; \
    public:\
    virtual size_t getEventTypeIndex() const { return T::s_eventTypeIndex; } \
    static bool checkEventType( const Event* event ) { return event->getEventTypeIndex() == T::s_eventTypeIndex; }


/// this has to be added in the Event implementation file
#define SOFA_EVENT_CPP(T) \
    const size_t T::s_eventTypeIndex = ++sofa::core::objectmodel::Event::s_lastEventTypeIndex;


/**
 *  \brief Base class for all events received by the objects.
 *
 * When created, the status is initialized as not handled. It is then propagated along the objects until it is handled.
 *
 * @author Francois Faure
 */
class SOFA_CORE_API Event
{
public:
    Event();

    virtual ~Event();

    /// Tag the event as handled, i.e. the event needs not be propagated further
    void setHandled();

    /// Returns true of the event has been handled
    bool isHandled() const;

    virtual const char* getClassName() const { return "Event"; }


    /// \returns unique type index
    /// for fast Event type comparison with unique indices (see function 'checkEventType')
    /// @warning this mechanism will only work for the last derivated type (and not for eventual intermediaries)
    /// e.g. for C derivated from B derivated from A, checkEventType will returns true only for C* but false for B* or A*
    /// Should be implemented by using macros SOFA_EVENT_H / SOFA_EVENT_CPP
    virtual size_t getEventTypeIndex() const = 0;

protected:
    bool m_handled;


    static size_t s_lastEventTypeIndex; ///< storing the last given id
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
