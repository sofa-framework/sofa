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
#include <queue>

#define SOURCE_LOCATION __FILE__, __LINE__

namespace sofa
{

namespace helper
{

namespace logging
{

class GtestMessageFrame
{
public:
    virtual ~GtestMessageFrame() {}

    const char* m_filename;
    int   m_lineno ;

    virtual void process(Message& m) ;
    virtual void finalize() ;
};

class GtestMessageFrameFailure : public GtestMessageFrame
{
public:
    Message::Type m_type;

    GtestMessageFrameFailure(Message::Type type,
                             const char* filename, int lineno) ;
    virtual void process(Message& message) ;
};

class GtestMessageFrameFailureWhenMissing  : public GtestMessageFrame
{
public:
    Message::Type m_type;
    bool  m_gotMessage {false} ;

    GtestMessageFrameFailureWhenMissing(Message::Type type,
                                        const char* filename,  int lineno) ;

    virtual void process(Message& message) ;
    void finalize() ;
};


class SOFA_TestPlugin_API GtestMessageHandler : public MessageHandler
{
    Message::Class m_class ;
    std::vector<std::vector<GtestMessageFrame*> > m_gtestframes;

public:
    GtestMessageHandler(Message::Class mclass) ;
    virtual ~ GtestMessageHandler();

    /// Inherited from MessageHandler
    virtual void process(Message& m) ;
    void pushFrame(Message::Type type, GtestMessageFrame* frame)  ;
    void popFrame(Message::Type type) ;
};

class SOFA_TestPlugin_API MainGtestMessageHandler
{
public:
    static GtestMessageHandler& getInstance() ;
    static void pushFrame(Message::Type type, GtestMessageFrame* frame) ;
    static void popFrame(Message::Type type) ;
};

struct SOFA_TestPlugin_API MesssageAsTestFailure2
{
    GtestMessageFrameFailure frame ;

    MesssageAsTestFailure2(Message::Type t,
                           const char* filename="unknown", int lineno=0) ;
    virtual ~MesssageAsTestFailure2() ;
};

struct SOFA_TestPlugin_API ExpectMessage2
{
    GtestMessageFrameFailureWhenMissing frame ;

    ExpectMessage2(Message::Type t,
                   const char* filename="unknown", int lineno=0) ;
    virtual ~ExpectMessage2() ;
};

//From http://en.cppreference.com/w/cpp/preprocessor/replace
#define EXPECT_MSG_PASTER(x,y) x ## _ ## y
#define EXPECT_MSG_EVALUATOR(x,y)  EXPECT_MSG_PASTER(x,y)

#define EXPECT_MSG_EMIT_V2(t) sofa::helper::logging::ExpectMessage2 EXPECT_MSG_EVALUATOR(__hiddenscopevar_, __LINE__) { sofa::helper::logging::Message::t, __FILE__, __LINE__ }
#define EXPECT_MSG_NOEMIT_V2(t) sofa::helper::logging::MesssageAsTestFailure2 EXPECT_MSG_EVALUATOR(__hiddenscopevar_, __LINE__){ sofa::helper::logging::Message::t, __FILE__, __LINE__ }


struct SOFA_TestPlugin_API ExpectMessage
{
    const char* m_filename;
    int  m_lineno;

    helper::vector<int>           m_lastCounts ;
    helper::vector<Message::Type> m_types ;

    ExpectMessage(const Message::Type t,
                  const char* filename="unknown", const int lineno=0) {
        m_filename=filename;
        m_lineno = lineno;
        m_types.push_back(t) ;
        m_lastCounts.push_back(MainCountingMessageHandler::getMessageCountFor(t)) ;
    }

    ExpectMessage(std::initializer_list<Message::Type> types,
                  const char* filename="unknown", const int lineno=0) {
        m_filename=filename;
        m_lineno = lineno;

        for(auto type : types){
            m_types.push_back(type) ;
            m_lastCounts.push_back( MainCountingMessageHandler::getMessageCountFor(type) ) ;
        }
    }

    ~ExpectMessage() {
        for(unsigned int i=0;i<m_types.size();++i){
            if(m_lastCounts[i] == MainCountingMessageHandler::getMessageCountFor(m_types[i]) )
            {
                ADD_FAILURE_AT(m_filename, m_lineno) << "A message of type '" << toString(m_types[i]) << "' was expected. None was received." << std::endl ;
            }
        }
    }
};

struct SOFA_TestPlugin_API MessageAsTestFailure
{
    const  char* m_filename;
    int  m_lineno;

    helper::vector<int>           m_lastCounts ;
    helper::vector<Message::Type> m_types ;
    LogMessage m_log;

    MessageAsTestFailure(std::initializer_list<Message::Type> types,
                         const char* filename="unknown", const int lineno=0)
    {
        m_filename = filename ;
        m_lineno = lineno ;

        for(auto type : types){
            m_types.push_back(type) ;
            m_lastCounts.push_back( MainCountingMessageHandler::getMessageCountFor(type) ) ;
        }
    }

    MessageAsTestFailure(const Message::Type type,
                         const char* filename="unknown", const int lineno=0)
    {
        m_filename = filename ;
        m_lineno = lineno ;
        m_types.push_back(type) ;
        m_lastCounts.push_back(MainCountingMessageHandler::getMessageCountFor(type)) ;
    }

    ~MessageAsTestFailure()
    {
        for(unsigned int i=0;i<m_types.size();++i){
            if(m_lastCounts[i] != MainCountingMessageHandler::getMessageCountFor(m_types[i]) )
            {
                std::stringstream backlog;
                backlog << "====================== Messages Backlog =======================" << std::endl ;
                for(auto& message : m_log)
                {
                    backlog << message << std::endl ;
                }
                backlog << "===============================================================" << std::endl ;

                ADD_FAILURE_AT(m_filename, m_lineno) << "A message of type '" << toString(m_types[i]) << "' was not expected but it was received. " << std::endl
                              << backlog.str() ;
            }
        }
    }
};

///TAKE FROM http://stackoverflow.com/questions/3046889/optional-parameters-with-c-macros
#define FUNC_CHOOSER(_f1, _f2, _f3, ...) _f3
#define FUNC_RECOMPOSER(argsWithParentheses) FUNC_CHOOSER argsWithParentheses

#define EXPECT_MSG_EMIT2( code, code2 ) sofa::helper::logging::ExpectMessage({sofa::helper::logging::Message::code, sofa::helper::logging::Message::code2}, SOURCE_LOCATION)
#define EXPECT_MSG_EMIT1( code ) sofa::helper::logging::ExpectMessage(sofa::helper::logging::Message::code, SOURCE_LOCATION)
#define EXPECT_MSG_EMIT0

#define EXPECT_MSG_EMIT_CHOOSE_FROM_ARG_COUNT(...) FUNC_RECOMPOSER((__VA_ARGS__, EXPECT_MSG_EMIT2, EXPECT_MSG_EMIT1, ))
#define EXPECT_MSG_EMIT_NO_ARG_EXPANDER() ,,EXPECT_MSG_EMIT0
#define EXPECT_MSG_EMIT_CHOOSER(...) EXPECT_MSG_EMIT_CHOOSE_FROM_ARG_COUNT(EXPECT_MSG_EMIT_NO_ARG_EXPANDER __VA_ARGS__ ())

#define EXPECT_MSG_EMIT(...) EXPECT_MSG_EMIT_CHOOSER(__VA_ARGS__)(__VA_ARGS__)


#define EXPECT_MSG_NOEMIT1( code ) sofa::helper::logging::MessageAsTestFailure(sofa::helper::logging::Message::code, SOURCE_LOCATION)
#define EXPECT_MSG_NOEMIT2( code, code2 ) sofa::helper::logging::MessageAsTestFailure({sofa::helper::logging::Message::code, sofa::helper::logging::Message::code2}, SOURCE_LOCATION)
#define EXPECT_MSG_NOEMIT( code, ... )


} // logging
} // helper

} // sofa

#endif // TESTMESSAGEHANDLER_H

