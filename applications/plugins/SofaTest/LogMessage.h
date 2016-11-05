/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-20ll6 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Contributors:                                                               *
*       - damien.marchal@univ-lille1.fr                                       *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
/*****************************************************************************
* User of this library should read the documentation
* in the messaging.h file.
******************************************************************************/
#ifndef LOGMESSAGE_H
#define LOGMESSAGE_H

#include <sofa/helper/vector.h>
#include <sofa/helper/logging/MessageHandler.h>
#include <sofa/helper/logging/Message.h>
#include "InitPlugin_test.h"
#include <gtest/gtest.h>

#include <sofa/helper/logging/RichConsoleStyleMessageFormatter.h>

namespace sofa
{

namespace helper
{

namespace logging
{

namespace logmessage
{
    using sofa::helper::vector ;
    using sofa::helper::logging::MainRichConsoleStyleMessageFormatter ;

class LoggingMessageHandler : public MessageHandler
{
public:
    virtual void process(Message& m)
    {
        if(m_activationCount>0){
            m_messages.push_back(m) ;
        }
    }

    void reset()
    {
        m_messages.clear() ;
    }

    int activate()
    {
        assert(m_activationCount>=0) ;
        m_activationCount++;
        return m_messages.size() ;
    }

    int deactivate()
    {
        assert(m_activationCount>0) ;
        m_activationCount--;

        int size = m_messages.size();

        if(m_activationCount<=0)
            m_messages.clear() ;

        return size;
    }

    LoggingMessageHandler() {}

    const vector<Message>& getMessages() const
    {
        return m_messages ;
    }

private:
    int m_activationCount {0};
    vector<Message> m_messages ;
} ;

class MainLogginMessageHandler
{
public:
    // singleton
    static LoggingMessageHandler& getInstance()
    {
        static LoggingMessageHandler s_instance;
        return s_instance;
    }

    static int activate()
    {
        return getInstance().activate() ;
    }

    static int deactivate()
    {
        return getInstance().deactivate() ;
    }

    static vector<Message> getMessages()
    {
        return getInstance().getMessages() ;
    }
};

class SOFA_TestPlugin_API LogMessage
{
public:
    LogMessage() {
        m_firstMessage = MainLogginMessageHandler::activate() ;
    }

    ~LogMessage() {}

    std::vector<Message>::const_iterator begin()
    {
        const std::vector<Message>& messages = MainLogginMessageHandler::getMessages() ;

        assert(m_firstMessage<messages.size()) ;
        return messages.begin()+m_firstMessage ;
    }

    std::vector<Message>::const_iterator end()
    {
        const std::vector<Message>& messages = MainLogginMessageHandler::getMessages() ;
        return messages.end() ;
    }

private:
    unsigned int m_firstMessage      {0} ;
};

struct SOFA_TestPlugin_API MessageAsTestFailure
{
    int m_lastCount      {0} ;
    Message::Type m_type {Message::TEmpty} ;
    ScopedDeactivatedTestMessageHandler m_scopeddeac ;
    LogMessage m_log;

    MessageAsTestFailure(const Message::Type t)
    {
        m_type = t ;
        m_lastCount = MainCountingMessageHandler::getMessageCountFor(m_type) ;
    }

    ~MessageAsTestFailure()
    {
        if(m_lastCount != MainCountingMessageHandler::getMessageCountFor(m_type) )
        {
            ADD_FAILURE() << "A message of type '" << m_type << "' was not expected but it was received. " << std::endl ;
            std::cout << "====================== Messages Backlog =======================" << std::endl ;
            for(auto& message : m_log)
            {
                std::cout << message << std::endl ;
            }
            std::cout << "===============================================================" << std::endl ;
        }
    }
};



} // logmessage

using logmessage::MessageAsTestFailure ;
using logmessage::MainLogginMessageHandler ;
using logmessage::LogMessage ;

} // logging
} // helper
} // sofa

#endif // TESTMESSAGEHANDLER_H

