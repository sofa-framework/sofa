/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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

#include <sofa/helper/logging/DefaultStyleMessageFormatter.h>
#include <sofa/helper/logging/TracyMessageHandler.h>
#include <sofa/helper/logging/MessageFormatter.h>
#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#endif


namespace sofa::helper::logging
{

TracyMessageHandler::TracyMessageHandler(MessageFormatter* formatter)
    : m_formatter(formatter)
{
    if (m_formatter == nullptr)
    {
        m_formatter = &DefaultStyleMessageFormatter::getInstance();
    }
}

void TracyMessageHandler::process(Message& m)
{
#ifdef TRACY_ENABLE
    std::stringstream ss;
    m_formatter->formatMessage(m, ss) ;
    TracyMessage(ss.str().c_str(), ss.str().size());
#endif
}

void TracyMessageHandler::setMessageFormatter(MessageFormatter* formatter)
{
    m_formatter = formatter;
}

TracyMessageHandler& MainTracyMessageHandler::getInstance()
{
    static TracyMessageHandler s_instance;
    return s_instance;
}

}
