//
// C++ Interface: KeypressedEvent
//
// Description:
//
//
// Author: Francois Faure, INRIA/UJF-Grenoble, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_CORE_OBJECTMODEL_KEYPRESSEDEVENT_H
#define SOFA_CORE_OBJECTMODEL_KEYPRESSEDEVENT_H

#include <sofa/core/objectmodel/Event.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

/**
	@author Francois Faure
*/
class KeypressedEvent : public Event
{
public:
    KeypressedEvent( char );

    ~KeypressedEvent();

    char getKey() const;

protected:
    char m_char;

};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
