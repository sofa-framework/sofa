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
#include "KeypressedEvent.h"

namespace Sofa
{

namespace Components
{

KeypressedEvent::KeypressedEvent(char c)
    : Sofa::Abstract::Event()
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


}

}
