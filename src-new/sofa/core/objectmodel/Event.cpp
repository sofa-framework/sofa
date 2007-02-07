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
#include <sofa/core/objectmodel/Event.h>

namespace sofa
{

namespace core
{

namespace objectmodel
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


} // namespace objectmodel

} // namespace core

} // namespace sofa
