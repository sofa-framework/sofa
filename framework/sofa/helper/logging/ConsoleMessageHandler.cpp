/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* This component is open-source                                               *
*                                                                             *
* Authors: Damien Marchal                                                     *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*****************************************************************************
* User of this library should read the documentation
* in the messaging.h file.
******************************************************************************/

#include <sstream>
using std::ostringstream ;

#include <iostream>

#include "Message.h"
#include "MessageFormatter.h"
#include "ConsoleMessageHandler.h"
#include "DefaultStyleMessageFormatter.h"


namespace sofa
{

namespace helper
{

namespace logging
{

ConsoleMessageHandler::ConsoleMessageHandler(MessageFormatter* formatter)
{
    m_formatter = (formatter==0?DefaultStyleMessageFormatter::getInstance():formatter);
}

void ConsoleMessageHandler::process(Message &m) {
    ostringstream out;
    m_formatter->formatMessage(m, out) ;
    std::cout << out.str() << std::endl ;
}


} // logging
} // helper
} // sofa

