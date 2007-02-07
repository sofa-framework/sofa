//
// C++ Implementation: Event
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include "Event.h"

namespace Sofa
{

namespace Abstract
{

Event::Event()
{
    m_handled = false;
}


Event::~Event()
{
}

void Event::setHandled()
{
    m_handled = true;
}

bool Event::isHandled() const
{
    return m_handled;
}


}

}
