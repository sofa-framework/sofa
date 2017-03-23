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
#include <SofaTest/TestMessageHandler.h>

namespace sofa
{

namespace helper
{

namespace logging
{

namespace {
    static struct raii {
      raii() {
            //std::cout << "INSTALLING THE HANDLER " << std::endl ;
            helper::logging::MessageDispatcher::addHandler( &MainGtestMessageHandler::getInstance() ) ;
      }
    } sin ;
}

void GtestMessageFrame::process(Message& m) {
    SOFA_UNUSED(m);
}

void GtestMessageFrame::finalize() {
}

GtestMessageFrameFailure::GtestMessageFrameFailure(Message::Type type, const char* filename, int lineno)
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


GtestMessageFrameFailureWhenMissing::GtestMessageFrameFailureWhenMissing( Message::Type type,  const char* filename,  int lineno)
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


GtestMessageHandler::GtestMessageHandler(Message::Class mclass) : m_class(mclass)
{
    for(unsigned int i=0; i < Message::TypeCount ; ++i)
    {
        m_gtestframes.push_back( std::vector<GtestMessageFrame*>({new GtestMessageFrame()}) ) ;
    }
}

/// Inherited from MessageHandler
void GtestMessageHandler::process(Message& m)
{
    //if(m.context()==m_class){
    //std::cout << "PROCESSING MESSAGE" << toString(m.type()) << ": " << m_gtestframes[m.type()].size() << std::endl;
    m_gtestframes[m.type()].back()->process(m) ;
    //}
}

GtestMessageHandler::~GtestMessageHandler()
{
    for(unsigned int i=0; i < Message::TypeCount ; ++i)
    {
        assert(m_gtestframes.size() >= 1) ;
        delete m_gtestframes[i][0] ;
    }
}

void GtestMessageHandler::pushFrame(Message::Type type, GtestMessageFrame* frame){
    m_gtestframes[type].push_back(frame) ;
}

void GtestMessageHandler::popFrame(Message::Type type){
    m_gtestframes[type].pop_back() ;
}

GtestMessageHandler& MainGtestMessageHandler::getInstance(){
    static GtestMessageHandler instance(Message::Runtime) ;
    return instance ;
}

void MainGtestMessageHandler::pushFrame(Message::Type type, GtestMessageFrame *frame){
    getInstance().pushFrame(type, frame) ;
}

void MainGtestMessageHandler::popFrame(Message::Type type){
    getInstance().popFrame(type) ;
}

MesssageAsTestFailure2::MesssageAsTestFailure2(Message::Type type,
                                               const char* filename, int lineno)
{
    //std::cout << "INSTALL HANDLER FOR" << toString(type) << std::endl;
    auto frame = new GtestMessageFrameFailure(type, filename, lineno) ;
    m_frames.push_back(frame);
    MainGtestMessageHandler::pushFrame(type, frame) ;
}

MesssageAsTestFailure2::MesssageAsTestFailure2(std::initializer_list<Message::Type> types,
                                               const char* filename, int lineno)
{
    for(Message::Type type : types)
    {
        //std::cout << "INSTALL HANDLER FOR" << toString(type) << std::endl;
        auto frame = new GtestMessageFrameFailure(type, filename, lineno) ;
        m_frames.push_back(frame);
        MainGtestMessageHandler::pushFrame(type, frame) ;
    }
}

MesssageAsTestFailure2::~MesssageAsTestFailure2(){
    for(auto frame : m_frames)
    {
        //std::cout << "REMOVE HANDLER FOR" << toString(frame->m_type) << std::endl;
        MainGtestMessageHandler::popFrame(frame->m_type) ;
        frame->finalize() ;
        delete frame;
    }
    m_frames.clear();
}


ExpectMessage2::ExpectMessage2(Message::Type type,
                               const char* filename, int lineno)
{
    auto frame = new GtestMessageFrameFailureWhenMissing(type, filename, lineno);
    m_frames.push_back( frame ) ;
    MainGtestMessageHandler::pushFrame(type, frame ) ;
}

ExpectMessage2::ExpectMessage2(std::initializer_list<Message::Type> types,
                               const char* filename, int lineno)
{
    for(Message::Type type : types)
    {
        auto frame = new GtestMessageFrameFailureWhenMissing(type, filename, lineno);
        m_frames.push_back(frame);
        MainGtestMessageHandler::pushFrame(type, frame) ;
    }
}

ExpectMessage2::~ExpectMessage2(){
    for(auto frame : m_frames)
    {
        MainGtestMessageHandler::popFrame(frame->m_type) ;
        frame->finalize() ;
        delete frame;
    }
    m_frames.clear();
}

} // logging
} // helper

} // sofa


