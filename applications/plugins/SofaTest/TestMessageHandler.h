/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef TESTMESSAGEHANDLER_H
#define TESTMESSAGEHANDLER_H

#include <sofa/helper/vector.h>
#include <sofa/helper/logging/CountingMessageHandler.h>
#include <sofa/helper/logging/LoggingMessageHandler.h>
#include <sofa/helper/logging/MessageHandler.h>
#include <sofa/helper/logging/Message.h>
#include "InitPlugin_test.h"
#include <gtest/gtest.h>

namespace sofa
{

namespace helper
{

namespace logging
{

const std::string toString(const Message::Type type) ;

struct SOFA_TestPlugin_API ExpectMessage
{
    std::string m_filename;
    uint64_t  m_lineno;

    int m_lastCount      {0} ;
    Message::Type m_type {Message::TEmpty} ;

    ExpectMessage(const Message::Type t, const std::string& filename, const uint64_t lineno) {

        m_type = t ;
        m_lastCount = MainCountingMessageHandler::getMessageCountFor(m_type) ;
    }

    ~ExpectMessage() {
        if(m_lastCount == MainCountingMessageHandler::getMessageCountFor(m_type) )
        {
            ADD_FAILURE() << "A message of type '" << toString(m_type) << "' was expected. None was received." << std::endl ;
        }
    }
};

struct SOFA_TestPlugin_API MessageAsTestFailure
{
    std::string m_filename;
    uint64_t  m_lineno;

    int m_lastCount      {0} ;
    Message::Type m_type {Message::TEmpty} ;
    LogMessage m_log;

    MessageAsTestFailure(const Message::Type t,
                         const std::string& filename="", const uint64_t lineno=0)
    {
        m_filename = filename ;
        m_lineno = lineno ;
        m_type = t ;
        m_lastCount = MainCountingMessageHandler::getMessageCountFor(m_type) ;
    }

    ~MessageAsTestFailure()
    {
        if(m_lastCount != MainCountingMessageHandler::getMessageCountFor(m_type) )
        {
            std::stringstream backlog;
            backlog << "====================== Messages Backlog =======================" << std::endl ;
            for(auto& message : m_log)
            {
                backlog << message << std::endl ;
            }
            backlog << "===============================================================" << std::endl ;

            ADD_FAILURE_AT(m_filename, m_lineno) << "A message of type '" << toString(m_type) << "' was not expected but it was received. " << std::endl
                          << backlog.str() ;
        }
    }
};

class SOFA_TestPlugin_API WarningAndErrorAsTestFailure
{
    MessageAsTestFailure m_error ;
    MessageAsTestFailure m_warning ;
public:
    WarningAndErrorAsTestFailure() :
        m_error(Message::Error),
        m_warning(Message::Warning) {
    }

    virtual ~WarningAndErrorAsTestFailure(){
    }
};

} // logging
} // helper

namespace test
{
    using sofa::helper::logging::WarningAndErrorAsTestFailure ;
}

} // sofa

#endif // TESTMESSAGEHANDLER_H

