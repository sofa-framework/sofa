/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
/*****************************************************************************
* User of this library should read the documentation
* in the messaging.h file.
******************************************************************************/
#include <sofa/helper/logging/LoggingMessageHandler.h>

namespace sofa
{
namespace helper
{
namespace logging
{
namespace loggingmessagehandler
{

void LoggingMessageHandler::process(Message& m)
{
    if(m_activationCount>0){
        m_messages.push_back(m) ;
    }
}

void LoggingMessageHandler::reset()
{
    m_messages.clear() ;
}

int LoggingMessageHandler::activate()
{
    assert(m_activationCount>=0) ;
    m_activationCount++;
    return m_messages.size() ;
}

int LoggingMessageHandler::deactivate()
{
    assert(m_activationCount>0) ;
    m_activationCount--;

    int size = m_messages.size();

    if(m_activationCount<=0)
        m_messages.clear() ;

    return size;
}

LoggingMessageHandler::LoggingMessageHandler()
{
}

const vector<Message>& LoggingMessageHandler::getMessages() const
{
    return m_messages ;
}

LoggingMessageHandler& MainLoggingMessageHandler::getInstance()
{
    static LoggingMessageHandler s_instance;
    return s_instance;
}

int MainLoggingMessageHandler::activate()
{
    return getInstance().activate() ;
}

int MainLoggingMessageHandler::deactivate()
{
    return getInstance().deactivate() ;
}

const vector<Message>& MainLoggingMessageHandler::getMessages()
{
    return getInstance().getMessages() ;
}

} // loggingmessagehandler
} // logging
} // helper
} // sofa

