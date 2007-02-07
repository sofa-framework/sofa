//
// C++ Interface: Event
//
// Description:
//
//
// Author: Francois Faure, INRIA/UJF-Grenoble, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_CORE_OBJECTMODEL_EVENT_H
#define SOFA_CORE_OBJECTMODEL_EVENT_H

namespace sofa
{

namespace core
{

namespace objectmodel
{

/**
Base class for all events received by the objects.
When created, the status is initialized as not handled. It is then propagated along the objects until it is handled.

	@author Francois Faure
*/
class Event
{
public:
    Event();

    virtual ~Event();

    /// Tag the event as handled, i.e. the event needs not be propagated further
    void setHandled();

    ///
    bool isHandled() const;

protected:
    bool m_handled;
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
