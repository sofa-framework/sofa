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
* Authors: Matthieu Nesme                                                     *
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


/// each ERROR and FATAL message raises a gtest error
class SOFA_TestPlugin_API TestMessageHandler : public MessageHandler
{
public:

    /// raises a gtest error as soon as message is an error
    /// iff the handler is active (see setActive)
    virtual void process(Message &m)
    {
        assert(m.type()<m_failsOn.size() && "If this happens this means that the code initializing m_failsOn is broken.") ;

        if( active && m_failsOn[m.type()] ){
            ADD_FAILURE() << "An error message was emitted and is interpreted as a test failure. "
                          <<  "src: " << std::string(m.fileInfo().filename) << ":" << m.fileInfo().line
                          << "message: " << m.message().str() << std::endl;

        }
    }

    // singleton
    static TestMessageHandler& getInstance()
    {
        static TestMessageHandler s_instance;
        return s_instance;
    }

    /// raising a gtest error can be temporarily deactivated
    /// indeed, sometimes, testing that a error message is raised is mandatory
    /// and should not raise a gtest error
    static void setActive( bool a ) { getInstance().active = a; }

private:
    sofa::helper::vector<bool> m_failsOn ;

    /// true by default
    bool active;

    // private default constructor for singleton
    TestMessageHandler() : active(true) {
        for(unsigned int i=Message::Info ; i<Message::TypeCount;i++){
            m_failsOn.push_back(false) ;
        }
        m_failsOn[Message::Error] = true ;
        m_failsOn[Message::Fatal] = true ;
    }

    void setFailureOn(const Message::Type m, bool state){
        m_failsOn[m] = state ;
    }
};


/// the TestMessageHandler is deactivated in the scope of a ScopedDeactivatedTestMessageHandler variable
struct SOFA_TestPlugin_API ScopedDeactivatedTestMessageHandler
{
    ScopedDeactivatedTestMessageHandler() { TestMessageHandler::setActive(false); }
    ~ScopedDeactivatedTestMessageHandler() { TestMessageHandler::setActive(true); }
};

class CountingMessageHandler : public MessageHandler
{
public:
    virtual void process(Message& m)
    {
        assert(m.type()<m_countMatching.size() && "If this happens this means that the code initializing m_countMatching is broken.") ;

        m_countMatching[m.type()]++ ;
    }

    void reset(){
        for(unsigned int i=0;i<m_countMatching.size();i++){
            m_countMatching[i] = 0 ;
        }
    }

    CountingMessageHandler() {
        for(unsigned int i=Message::Info;i<Message::TypeCount;i++){
            m_countMatching.push_back(0) ;
        }
    }

    int getMessageCountFor(const Message::Type& type) const {
        assert(type < m_countMatching.size() && "If this happens this means that the code initializing m_countMatching is broken.") ;
        return m_countMatching[type] ;
    }

private:
    sofa::helper::vector<int> m_countMatching ;
} ;

class MainCountingMessageHandler
{
public:
    // singleton
    static sofa::helper::logging::CountingMessageHandler& getInstance()
    {
        static sofa::helper::logging::CountingMessageHandler s_instance;
        return s_instance;
    }

    static void reset(){
        getInstance().reset() ;
    }

    static int getMessageCountFor(const Message::Type &type)
    {
        return getInstance().getMessageCountFor(type) ;
    }
};


struct SOFA_TestPlugin_API ExpectMessage
{
    int m_lastCount      {0} ;
    Message::Type m_type {Message::TEmpty} ;
    ScopedDeactivatedTestMessageHandler m_scopeddeac ;

    ExpectMessage(const Message::Type t) {
        m_type = t ;
        m_lastCount = MainCountingMessageHandler::getMessageCountFor(m_type) ;
    }

    ~ExpectMessage() {
        if(m_lastCount == MainCountingMessageHandler::getMessageCountFor(m_type) )
        {
            ADD_FAILURE() << "A message of type '" << m_type << "' was expected. None was received." << std::endl ;
        }
    }
};



} // logging
} // helper
} // sofa

#endif // TESTMESSAGEHANDLER_H

