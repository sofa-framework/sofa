//
// C++ Implementation: KeypressedEvent
//
// Description:
//
//
// Author: Francois Faure, UJF-Grenoble/INRIA, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include <sofa/core/objectmodel/KeypressedEvent.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

KeypressedEvent::KeypressedEvent(char c)
    : Event()
    , m_char(c)
{
}


KeypressedEvent::~KeypressedEvent()
{
}

char KeypressedEvent::getKey() const
{
    return m_char;
}


} // namespace objectmodel

} // namespace core

} // namespace sofa
