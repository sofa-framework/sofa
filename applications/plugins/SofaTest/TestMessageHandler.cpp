/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <SofaTest/TestMessageHandler.h>


////////////////////////////////////////////////////////////////////////////////////////////////////
/// This file organization:
///     - PRIVATE DECLARATION  (the class that are only used internally)
///     - PRIVATE DEFINITION   (the implementation of the private classes)
///     - PUBLIC  DEFINITION   (the implementation of the public classes)
////////////////////////////////////////////////////////////////////////////////////////////////////
namespace sofa
{

namespace helper
{

namespace logging
{

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// PRIVATE DECLARATION /////////////////////////////////////////////////
/// Here are declared the classes that are only used for internal use.
/// In case someone want to use them it is easy to move that in .h file.
/// Until that happens, keeps these here to hide the implementation details from the users of the .h
/// And accelerate compilation of Sofa :)
////////////////////////////////////////////////////////////////////////////////////////////////////
class GtestMessageFrame
{
public:
    virtual ~GtestMessageFrame() {}

    virtual void process(Message& /*m*/) {}
    virtual void finalize() {}

public:
    Message::Type m_type;
    const char* m_filename;
    int   m_lineno ;
};

class GtestMessageFrameFailure : public GtestMessageFrame
{
public:

    GtestMessageFrameFailure(Message::Type type,
                             const char* filename, int lineno) ;
    virtual void process(Message& message) ;
};

class GtestMessageFrameFailureWhenMissing  : public GtestMessageFrame
{
public:
    bool  m_gotMessage {false} ;

    GtestMessageFrameFailureWhenMissing(Message::Type type,
                                        const char* filename,  int lineno) ;

    virtual void process(Message& message) ;
    virtual void finalize() ;
};

class GtestMessageFrameIgnore  : public GtestMessageFrame
{
public:
    GtestMessageFrameIgnore(Message::Type type) ;
};



class SOFA_TestPlugin_API GtestMessageHandler : public MessageHandler
{
    std::vector<std::vector<GtestMessageFrame*> > m_gtestframes;

public:
    GtestMessageHandler(Message::Class mclass) ;
    virtual ~ GtestMessageHandler();

    /// Inherited from MessageHandler
    virtual void process(Message& m) ;
    void pushFrame(Message::Type type, GtestMessageFrame* frame)  ;
    void popFrame(Message::Type type) ;
};

class SOFA_TestPlugin_API MainGtestMessageHandlerPrivate
{
public:
    static GtestMessageHandler& getInstance() ;
    static void pushFrame(Message::Type type, GtestMessageFrame* frame) ;
    static void popFrame(Message::Type type) ;
};



////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// DEFINITION OF PRIVATE CLASSES //////////////////////////////////
///
///
////////////////////////////////////////////////////////////////////////////////////////////////////
GtestMessageFrameFailure::GtestMessageFrameFailure(Message::Type type,
                                                   const char* filename,
                                                   int lineno)
{
    m_type = type;
    m_filename = filename;
    m_lineno = lineno;
}


void GtestMessageFrameFailure::process(Message& message) {
    std::stringstream backlog;
    backlog << "====================== Messages =======================" << std::endl ;
    backlog << message << std::endl ;
    backlog << "===============================================================" << std::endl ;

    ADD_FAILURE_AT(m_filename, m_lineno) << "A message of type '" << toString(message.type())
                                         << "' was not expected but it was received. " << std::endl
                                         << backlog.str() ;
}

GtestMessageFrameFailureWhenMissing::GtestMessageFrameFailureWhenMissing(Message::Type type,
                                                                         const char* filename,
                                                                         int lineno)
{
    m_type = type;
    m_filename = filename;
    m_lineno = lineno;
}

void  GtestMessageFrameFailureWhenMissing::process(Message& message) {
    SOFA_UNUSED(message);
    m_gotMessage = true ;
}

void GtestMessageFrameFailureWhenMissing::finalize(){
    if(!m_gotMessage)
        ADD_FAILURE_AT(m_filename, m_lineno) << "A message of type '" << toString(m_type)
                                             << "' was expected but none was received. " << std::endl ;
}


GtestMessageFrameIgnore::GtestMessageFrameIgnore(Message::Type type)
{
    m_type = type;
    m_filename = "";
    m_lineno = -1;
}



GtestMessageHandler::GtestMessageHandler(Message::Class mclass)
{
    for(unsigned int i=0; i < Message::TypeCount ; ++i)
    {
        m_gtestframes.push_back( std::vector<GtestMessageFrame*>({new GtestMessageFrame()}) ) ;
    }
}

void GtestMessageHandler::process(Message& m)
{
    m_gtestframes[m.type()].back()->process(m) ;
}

GtestMessageHandler::~GtestMessageHandler()
{
    for(unsigned int i=0; i < Message::TypeCount ; ++i)
    {
        assert(m_gtestframes.size() >= 1) ;
        delete m_gtestframes[i][0] ;
    }
}

void GtestMessageHandler::pushFrame(Message::Type type,
                                    GtestMessageFrame* frame){
    m_gtestframes[type].push_back(frame) ;
}

void GtestMessageHandler::popFrame(Message::Type type){
    m_gtestframes[type].pop_back() ;
}

MessageHandler* MainGtestMessageHandler::getInstance(){
    return &MainGtestMessageHandlerPrivate::getInstance() ;
}

GtestMessageHandler& MainGtestMessageHandlerPrivate::getInstance(){
    static GtestMessageHandler instance(Message::Runtime) ;
    return instance ;
}

void MainGtestMessageHandlerPrivate::pushFrame(Message::Type type, GtestMessageFrame *frame){
    getInstance().pushFrame(type, frame) ;
}

void MainGtestMessageHandlerPrivate::popFrame(Message::Type type){
    getInstance().popFrame(type) ;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// DEFINITION OF PUBLIC CLASSES ///////////////////////////////////
///
///
////////////////////////////////////////////////////////////////////////////////////////////////////
MessageAsTestFailure::MessageAsTestFailure(Message::Type type,
                                               const char* filename, int lineno)
{
    m_frame = new GtestMessageFrameFailure(type, filename, lineno) ;
    MainGtestMessageHandlerPrivate::pushFrame(type, m_frame) ;
}

MessageAsTestFailure::~MessageAsTestFailure(){
    MainGtestMessageHandlerPrivate::popFrame( m_frame->m_type ) ;
    m_frame->finalize() ;
    delete m_frame ;
}

ExpectMessage::ExpectMessage(Message::Type type,
                               const char* filename, int lineno)
{
    m_frame = new GtestMessageFrameFailureWhenMissing(type, filename, lineno);
    MainGtestMessageHandlerPrivate::pushFrame(type, m_frame ) ;
}

ExpectMessage::~ExpectMessage(){
    MainGtestMessageHandlerPrivate::popFrame(m_frame->m_type) ;
    m_frame->finalize() ;
    delete m_frame ;
}


IgnoreMessage::IgnoreMessage(Message::Type type)
{
    m_frame = new GtestMessageFrameIgnore(type);
    MainGtestMessageHandlerPrivate::pushFrame(type, m_frame ) ;
}

IgnoreMessage::~IgnoreMessage(){
    MainGtestMessageHandlerPrivate::popFrame(m_frame->m_type) ;
    m_frame->finalize() ;
    delete m_frame ;
}


} // logging
} // helper

} // sofa


