/******************************************************************************
*       SOFA, Simulation-Framework Architecture, development version     *
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
#include <gtest/gtest.h>
#include <exception>
#include <algorithm>
#include <thread>

#include <iostream>
using std::endl ;

#include <vector>
using std::vector ;

#include <sofa/helper/logging/Messaging.h>
using sofa::helper::logging::MessageDispatcher ;

#include <sofa/helper/logging/MessageHandler.h>
using sofa::helper::logging::MessageHandler ;

#include <sofa/helper/logging/Message.h>
using sofa::helper::logging::Message ;

#include <sofa/helper/logging/LoggingMessageHandler.h>
using sofa::helper::logging::MainLoggingMessageHandler ;
using sofa::helper::logging::LogMessage ;

#include <sofa/helper/logging/CountingMessageHandler.h>
using sofa::helper::logging::MainCountingMessageHandler ;
using sofa::helper::logging::CountingMessageHandler ;

#include <sofa/helper/logging/RoutingMessageHandler.h>
using sofa::helper::logging::RoutingMessageHandler ;
using sofa::helper::logging::MainRoutingMessageHandler ;

#include <sofa/helper/logging/ConsoleMessageHandler.h>
using sofa::helper::logging::ConsoleMessageHandler ;

#include <sofa/core/logging/PerComponentLoggingMessageHandler.h>
using sofa::helper::logging::PerComponentLoggingMessageHandler ;
using sofa::helper::logging::MainPerComponentLoggingMessageHandler ;

#include <sofa/core/logging/RichConsoleStyleMessageFormatter.h>
using sofa::helper::logging::RichConsoleStyleMessageFormatter ;

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/ObjectFactory.h>

//TODO(dmarchal): replace that with the LoggingMessageHandler
class MyMessageHandler : public MessageHandler
{
    vector<Message> m_messages ;
public:
    virtual void process(Message& m){
        m_messages.push_back(m);
    }

    size_t numMessages(){
        return m_messages.size() ;
    }

    const vector<Message>& messages() const {
        return m_messages;
    }
    const Message& lastMessage() const {
        return m_messages.back();
    }
} ;

TEST(LoggingTest, noHandler)
{
    // This test does not test anything, except the absence of crash
    MessageDispatcher::clearHandlers() ;

    msg_info("") << " info message with conversion" << 1.5 << "\n" ;
    msg_deprecated("") << " deprecated message with conversion" << 1.5 << "\n" ;
    msg_warning("") << " warning message with conversion "<< 1.5 << "\n" ;
    msg_error("") << " error message with conversion" << 1.5 << "\n" ;
}


TEST(LoggingTest, oneHandler)
{
    MessageDispatcher::clearHandlers() ;

    MyMessageHandler h;

    // add is expected to return the handler ID. Here is it the o'th
    EXPECT_TRUE(MessageDispatcher::addHandler(&h) == 0 ) ;

    msg_info("") << " info message with conversion" << 1.5 << "\n" ;
    msg_deprecated("") << " deprecated message with conversion" << 1.5 << "\n" ;
    msg_warning("") << " warning message with conversion "<< 1.5 << "\n" ;
    msg_error("") << " error message with conversion" << 1.5 << "\n" ;

    EXPECT_EQ( h.numMessages() , 4u ) ;
}

TEST(LoggingTest, duplicatedHandler)
{
    MessageDispatcher::clearHandlers() ;

    MyMessageHandler h;

    // First add is expected to return the handler ID.
    EXPECT_TRUE(MessageDispatcher::addHandler(&h) == 0) ;

    // Second is supposed to fail to add and thus return -1.
    EXPECT_TRUE(MessageDispatcher::addHandler(&h) == -1) ;

    msg_info("") << " info message with conversion" << 1.5 << "\n" ;
    msg_deprecated("") << " deprecated message with conversion" << 1.5 << "\n" ;
    msg_warning("") << " warning message with conversion "<< 1.5 << "\n" ;
    msg_error("") << " error message with conversion" << 1.5 << "\n" ;

    EXPECT_TRUE( h.numMessages() == 4u) ;
}

void f1()
{
    for(unsigned int i=0;i<100000;i++){
        msg_info("Thread1") << "Hello world" ;
        msg_warning("Thread1") << "Hello world" ;
        msg_error("Thread1") << "Hello world" ;
    }
}

void f2()
{
    for(unsigned int i=0;i<100000;i++){
        msg_info("Thread2") << "Hello world" ;
        msg_warning("Thread2") << "Hello world" ;
        msg_error("Thread2") << "Hello world" ;
    }
}

void f3()
{
    for(unsigned int i=0;i<100000;i++){
        msg_info("Thread3") << "Hello world" ;
        msg_warning("Thread3") << "Hello world" ;
        msg_error("Thread3") << "Hello world" ;
    }
}



TEST(LoggingTest, threadingTests)
{
    if(!SOFA_WITH_THREADING){
        /// This cout shouldn't be using the msg_* API.
        std::cout << "Test canceled because sofa is not compiled with SOFA_WITH_THREADING option." << std::endl ;
        return ;
    }

    MessageDispatcher::clearHandlers() ;

    CountingMessageHandler& mh = MainCountingMessageHandler::getInstance();
    mh.reset();

    // First add is expected to return the handler ID.
    EXPECT_TRUE(MessageDispatcher::addHandler(&mh) == 0) ;

    std::thread t1(f1);
    std::thread t1bis(f1);
    std::thread t2(f2);
    std::thread t3(f3);

    t1.join();
    t1bis.join();
    t2.join();
    t3.join();

    EXPECT_EQ( mh.getMessageCountFor(Message::Info), 400000) ;
    EXPECT_EQ( mh.getMessageCountFor(Message::Warning), 400000) ;
    EXPECT_EQ( mh.getMessageCountFor(Message::Error), 400000) ;
}


TEST(LoggingTest, withoutDevMode)
{
    MessageDispatcher::clearHandlers() ;

    MyMessageHandler h;
    MessageDispatcher::addHandler(&h) ;

    msg_info("") << " info message with conversion" << 1.5 << "\n" ;
    msg_deprecated("") << " deprecated message with conversion" << 1.5 << "\n" ;
    msg_warning("") << " warning message with conversion "<< 1.5 << "\n" ;
    msg_error("") << " error message with conversion" << 1.5 << "\n" ;

    nmsg_info("") << " null info message with conversion" << 1.5 << "\n" ;
    nmsg_deprecated("") << " null deprecated message with conversion" << 1.5 << "\n" ;
    nmsg_warning("") << " null warning message with conversion "<< 1.5 << "\n" ;
    nmsg_error("") << " null error message with conversion" << 1.5 << "\n" ;

    EXPECT_TRUE( h.numMessages() == 4u ) ;
}

TEST(LoggingTest, speedTest)
{
    MessageDispatcher::clearHandlers() ;

    MyMessageHandler h;
    MessageDispatcher::addHandler(&h) ;

    for(unsigned int i=0;i<100000;i++){
        msg_info("") << " info message with conversion" << 1.5 << "\n" ;
        msg_warning("") << " warning message with conversion "<< 1.5 << "\n" ;
        msg_error("") << " error message with conversion" << 1.5 << "\n" ;
    }
}



TEST(LoggingTest, emptyMessage)
{
    MessageDispatcher::clearHandlers() ;
    MyMessageHandler h;
    MessageDispatcher::addHandler(&h) ;

    // an empty message should not be processed

    msg_info("");
    EXPECT_EQ( h.numMessages(), 0u );

    msg_info("")<<"ok";
    msg_info("");
    EXPECT_EQ( h.numMessages(), 1u );
}
#include <string>
#include <sofa/core/objectmodel/Data.h>
using sofa::core::objectmodel::TData ;



class MyComponent : public sofa::core::objectmodel::BaseObject
{
public:

    MyComponent()
    {

    }

    void emitSerrSoutMessages(){
        f_printLog.setValue(true); // to print sout
        serr<<"regular serr"<<sendl;
        sout<<"regular sout"<<sendl;
        serr<<SOFA_FILE_INFO<<"serr with fileinfo"<<sendl;
        sout<<SOFA_FILE_INFO<<"sout with fileinfo"<<sendl;
    }

    void emitMessages(){
        msg_info(this) << "an info message" ;
        msg_warning(this) << "a warning message" ;
        msg_error(this) << "an error message" ;
    }
};




TEST(LoggingTest, checkBaseObjectSerr)
{
    MessageDispatcher::clearHandlers() ;
    MyMessageHandler h;
    MessageDispatcher::addHandler(&h) ;


    MyComponent c;

    c.emitSerrSoutMessages();
    /// the constructor of MyComponent is sending 4 messages
    EXPECT_EQ( h.numMessages(), 4u ) ;

    c.serr<<"regular external serr"<<c.sendl;
    EXPECT_EQ( h.lastMessage().fileInfo()->line, 0 );
    EXPECT_TRUE( !strcmp( h.lastMessage().fileInfo()->filename, sofa::helper::logging::s_unknownFile ) );
    EXPECT_EQ( h.lastMessage().type(), sofa::helper::logging::Message::Warning );
    EXPECT_EQ( h.lastMessage().context(), sofa::helper::logging::Message::Runtime );

    c.serr<<sofa::helper::logging::Message::Error<<"external serr as Error"<<c.sendl;
    EXPECT_EQ( h.lastMessage().fileInfo()->line, 0 );
    EXPECT_TRUE( !strcmp( h.lastMessage().fileInfo()->filename, sofa::helper::logging::s_unknownFile ) );
    EXPECT_EQ( h.lastMessage().type(), sofa::helper::logging::Message::Error );
    EXPECT_EQ( h.lastMessage().context(), sofa::helper::logging::Message::Runtime );

    c.sout<<"regular external sout"<<c.sendl;
    EXPECT_EQ( h.lastMessage().fileInfo()->line, 0 );
    EXPECT_TRUE( !strcmp( h.lastMessage().fileInfo()->filename, sofa::helper::logging::s_unknownFile ) );
    EXPECT_EQ( h.lastMessage().type(), sofa::helper::logging::Message::Info );
    EXPECT_EQ( h.lastMessage().context(), sofa::helper::logging::Message::Runtime );

    c.sout<<sofa::helper::logging::Message::Error<<"external sout as Error"<<c.sendl;
    EXPECT_EQ( h.lastMessage().fileInfo()->line, 0 );
    EXPECT_TRUE( !strcmp( h.lastMessage().fileInfo()->filename, sofa::helper::logging::s_unknownFile ) );
    EXPECT_EQ( h.lastMessage().type(), sofa::helper::logging::Message::Error );
    EXPECT_EQ( h.lastMessage().context(), sofa::helper::logging::Message::Runtime );


    c.serr<<SOFA_FILE_INFO<<"external serr with fileinfo"<<c.sendl;
    EXPECT_EQ( h.lastMessage().fileInfo()->line, __LINE__-1 );
    EXPECT_TRUE( !strcmp( h.lastMessage().fileInfo()->filename, __FILE__ ) );

    c.sout<<SOFA_FILE_INFO<<"external sout with fileinfo"<<c.sendl;
    EXPECT_EQ( h.lastMessage().fileInfo()->line, __LINE__-1 );
    EXPECT_TRUE( !strcmp( h.lastMessage().fileInfo()->filename, __FILE__ ) );

    c.serr<<SOFA_FILE_INFO<<sofa::helper::logging::Message::Error<<"external serr as Error with fileinfo"<<c.sendl;
    EXPECT_EQ( h.lastMessage().fileInfo()->line, __LINE__-1 );
    EXPECT_TRUE( !strcmp( h.lastMessage().fileInfo()->filename, __FILE__ ) );
    EXPECT_EQ( h.lastMessage().type(), sofa::helper::logging::Message::Error );

    c.sout<<sofa::helper::logging::Message::Error<<SOFA_FILE_INFO<<"external sout as Error with fileinfo"<<c.sendl;
    EXPECT_EQ( h.lastMessage().fileInfo()->line, __LINE__-1 );
    EXPECT_TRUE( !strcmp( h.lastMessage().fileInfo()->filename, __FILE__ ) );
    EXPECT_EQ( h.lastMessage().type(), sofa::helper::logging::Message::Error );
    EXPECT_EQ( h.lastMessage().context(), sofa::helper::logging::Message::Runtime );

    c.serr<<"serr with sendl that comes in a second time";
    c.serr<<c.sendl;
    EXPECT_EQ( h.lastMessage().fileInfo()->line, 0 );
    EXPECT_TRUE( !strcmp( h.lastMessage().fileInfo()->filename, sofa::helper::logging::s_unknownFile ) );
    EXPECT_EQ( h.lastMessage().type(), sofa::helper::logging::Message::Warning );

    c.serr<<"\n serr with \n end of "<<std::endl<<" lines"<<c.sendl;
    EXPECT_EQ( h.lastMessage().fileInfo()->line, 0 );
    EXPECT_TRUE( !strcmp( h.lastMessage().fileInfo()->filename, sofa::helper::logging::s_unknownFile ) );
    EXPECT_EQ( h.lastMessage().type(), sofa::helper::logging::Message::Warning );

    c.serr<<sofa::helper::logging::Message::Dev<<sofa::helper::logging::Message::Error<<"external Dev serr"<<c.sendl;
    EXPECT_EQ( h.lastMessage().type(), sofa::helper::logging::Message::Error );
    EXPECT_EQ( h.lastMessage().context(), sofa::helper::logging::Message::Dev );


    EXPECT_EQ( h.numMessages(), 15u ) ;

    // an empty message should not be processed
    c.serr<<c.sendl;

    EXPECT_EQ( h.numMessages(), 15u ) ;

}

TEST(LoggingTest, checkBaseObjectMsgAPI)
{
    MessageDispatcher::clearHandlers() ;
    MyMessageHandler h;
    MessageDispatcher::addHandler(&h) ;


    MyComponent c;

    c.f_printLog.setValue(true);

    c.emitMessages() ;
    EXPECT_EQ(h.numMessages(), 3u);
    EXPECT_EQ(c.getLoggedMessages().size(), 0u) ;

    /// We install the handler that copy the message into the component.
    MessageDispatcher::addHandler(&MainPerComponentLoggingMessageHandler::getInstance()) ;

    c.emitMessages() ;
    EXPECT_EQ(h.numMessages(), 6u);

    std::stringstream s;
    s << "============= Back log ==============" << std::endl ;
    for(auto& message : c.getLoggedMessages()){
        s << message << std::endl ;
    }
    s << "=====================================" << std::endl ; ;
    EXPECT_EQ(c.getLoggedMessages().size(), 3u) << s.str();

    msg_info(&c) << "A fourth message ";

    EXPECT_EQ(c.getLoggedMessages().size(), 4u) << s.str();
}

TEST(LoggingTest, checkBaseObjectMsgAPInoPrintLog)
{
    MessageDispatcher::clearHandlers() ;
    MyMessageHandler h;
    MessageDispatcher::addHandler(&h) ;


    MyComponent c;

    c.f_printLog.setValue(false);

    c.emitMessages() ;
    EXPECT_EQ(h.numMessages(), 2u);
    EXPECT_EQ(c.getLoggedMessages().size(), 0u) ;

    /// We install the handler that copy the message into the component.
    MessageDispatcher::addHandler(&MainPerComponentLoggingMessageHandler::getInstance()) ;

    c.emitMessages() ;
    EXPECT_EQ(h.numMessages(), 4u);

    std::stringstream s;
    s << "============= Back log ==============" << std::endl ;
    for(auto& message : c.getLoggedMessages()){
        s << message << std::endl ;
    }
    s << "=====================================" << std::endl ; ;
    EXPECT_EQ(c.getLoggedMessages().size(), 2u) << s.str();

    msg_info(&c) << "A fourth message ";

    EXPECT_EQ(c.getLoggedMessages().size(), 2u) << s.str();
}


TEST(LoggingTest, checkBaseObjectQueueSize)
{
    /// We install the handler that copy the message into the component.
    MessageDispatcher::clearHandlers() ;
    MessageDispatcher::addHandler(&MainPerComponentLoggingMessageHandler::getInstance()) ;

    MyComponent c;
    c.f_printLog.setValue(true);

    /// Filling the internal message queue.
    for(unsigned int i=0;i<100;i++){
        c.emitMessages();
    }
    EXPECT_EQ(c.getLoggedMessages().size(), 100u);
}

TEST(LoggingTest, checkBaseObjectSoutSerr)
{
    /// We install the handler that copy the message into the component.
    MessageDispatcher::clearHandlers() ;
    MessageDispatcher::addHandler(&MainPerComponentLoggingMessageHandler::getInstance()) ;

    MyComponent c;
    c.emitSerrSoutMessages();

    /// Well serr message are routed through warning while
    /// Sout are routed to the Info ones.
    EXPECT_TRUE(c.getLoggedMessagesAsString({Message::Error}).empty());
    EXPECT_TRUE(c.getLoggedMessagesAsString({Message::Fatal}).empty());
    EXPECT_FALSE(c.getLoggedMessagesAsString({Message::Warning}).empty());
    EXPECT_FALSE(c.getLoggedMessagesAsString({Message::Error,
                                              Message::Warning,
                                              Message::Fatal}).empty());

    EXPECT_TRUE(c.getLoggedMessagesAsString({Message::Deprecated}).empty());
    EXPECT_TRUE(c.getLoggedMessagesAsString({Message::Advice}).empty());
    EXPECT_FALSE(c.getLoggedMessagesAsString({Message::Info}).empty());
    EXPECT_FALSE(c.getLoggedMessagesAsString({Message::Info,
                                              Message::Deprecated,
                                              Message::Advice}).empty());

}


#undef MESSAGING_H
#ifndef WITH_SOFA_DEVTOOLS
   #define WITH_SOFA_DEVTOOLS
#endif
#undef dmsg_info
#undef dmsg_deprecated
#undef dmsg_error
#undef dmsg_warning
#undef dmsg_fatal
#undef dmsg_advice
#include <sofa/helper/logging/Messaging.h>

TEST(LoggingTest, withDevMode)
{
    MessageDispatcher::clearHandlers() ;

    MyMessageHandler h;
    MessageDispatcher::addHandler(&h) ;

    msg_info("") << " info message with conversion" << 1.5 << "\n" ;
    msg_warning("") << " warning message with conversion "<< 1.5 << "\n" ;
    msg_error("") << " error message with conversion" << 1.5 << "\n" ;

    nmsg_info("") << " null info message with conversion" << 1.5 << "\n" ;
    nmsg_warning("") << " null warning message with conversion "<< 1.5 << "\n" ;
    nmsg_error("") << " null error message with conversion" << 1.5 << "\n" ;

    dmsg_info("") << " debug info message with conversion" << 1.5 << "\n" ;
    dmsg_warning("") << " debug warning message with conversion "<< 1.5 << "\n" ;
    dmsg_error("") << " debug error message with conversion" << 1.5 << "\n" ;

    EXPECT_TRUE( h.numMessages() == 6u ) ;
}

TEST(LoggingTest, checkLoggingMessageHandler)
{
    CountingMessageHandler& m = MainCountingMessageHandler::getInstance() ;
    m.reset();

    MessageDispatcher::clearHandlers() ;
    MessageDispatcher::addHandler( &MainLoggingMessageHandler::getInstance() );
    LogMessage loggingFrame;

    msg_info("") << " info message with conversion" << 1.5 << "\n" ;
    msg_warning("") << " warning message with conversion "<< 1.5 << "\n" ;
    msg_error("") << " error message with conversion" << 1.5 << "\n" ;

    nmsg_info("") << " null info message with conversion" << 1.5 << "\n" ;
    nmsg_warning("") << " null warning message with conversion "<< 1.5 << "\n" ;
    nmsg_error("") << " null error message with conversion" << 1.5 << "\n" ;

    if( loggingFrame.size() != 3 )
    {
        EXPECT_EQ( loggingFrame.size(), 3) ;
        for(auto& message : loggingFrame)
        {
            std::cout << message << std::endl ;
        }
    }
}


TEST(LoggingTest, checkCountingMessageHandler)
{
    CountingMessageHandler& m = MainCountingMessageHandler::getInstance() ;
    m.reset();

    MessageDispatcher::clearHandlers() ;
    MessageDispatcher::addHandler( &m );

    std::vector<Message::Type> errortypes = {Message::Error, Message::Warning, Message::Info,
                                             Message::Advice, Message::Deprecated, Message::Fatal} ;

    for(auto& type : errortypes)
    {
        EXPECT_EQ(m.getMessageCountFor(type), 0) ;
    }

    int i = 0 ;
    for(auto& type : errortypes)
    {
        i++;
        msg_info("") << " info message with conversion" << 1.5 << "\n" ;
        msg_warning("") << " warning message with conversion "<< 1.5 << "\n" ;
        msg_error("") << " error message with conversion" << 1.5 << "\n" ;
        msg_fatal("") << " fatal message with conversion" << 1.5 << "\n" ;
        msg_deprecated("") << " deprecated message with conversion "<< 1.5 << "\n" ;
        msg_advice("") << " advice message with conversion" << 1.5 << "\n" ;

        nmsg_info("") << " null info message with conversion" << 1.5 << "\n" ;
        nmsg_warning("") << " null warning message with conversion "<< 1.5 << "\n" ;
        nmsg_error("") << " null error message with conversion" << 1.5 << "\n" ;
        nmsg_fatal("") << " fatal message with conversion" << 1.5 << "\n" ;
        nmsg_deprecated("") << " deprecated message with conversion "<< 1.5 << "\n" ;
        nmsg_advice("") << " advice message with conversion" << 1.5 << "\n" ;

        EXPECT_EQ(m.getMessageCountFor(type), i) ;
    }
}

TEST(LoggingTest, checkRoutingMessageHandler)
{
    RoutingMessageHandler& m = MainRoutingMessageHandler::getInstance() ;

    MessageDispatcher::clearHandlers() ;
    MessageDispatcher::addHandler( &m );

    std::vector<Message::Type> errortypes = {Message::Error, Message::Warning, Message::Info,
                                             Message::Advice, Message::Deprecated, Message::Fatal} ;

    RichConsoleStyleMessageFormatter* fmt = new RichConsoleStyleMessageFormatter();
    ConsoleMessageHandler* consolehandler = new ConsoleMessageHandler(fmt) ;

    /// Install a simple message filter that always call the ConsoleMessageHandler.
    m.setAFilter( [](Message&) { return true ; }, consolehandler );


    CountingMessageHandler* countinghandler = new CountingMessageHandler() ;
    /// Install a new message filter that cal the counting message handler if the message
    /// are Runtime & Info
    m.setAFilter( [](Message& m)
        {
            if(m.context() == Message::Runtime && m.type() == Message::Warning)
                return true ;
            return false ;
        }, countinghandler );

    msg_info("test") << "An info message " ;
    dmsg_info("test") << "An info message " ;
    logmsg_info("test") << "An info message " ;

    for(auto& type : errortypes)
        EXPECT_EQ( countinghandler->getMessageCountFor(type), 0) ;

    msg_warning("test") << "An second message " ;
    dmsg_warning("test") << "An second message " ;
    logmsg_warning("test") << "An second message " ;

    EXPECT_EQ( countinghandler->getMessageCountFor(Message::Warning), 1) ;
    EXPECT_EQ( countinghandler->getMessageCountFor(Message::Info), 0) ;


    countinghandler->reset() ;
    m.removeAllFilters();

    msg_warning("test") << "An second message " ;
    dmsg_warning("test") << "An second message " ;
    logmsg_warning("test") << "An second message " ;

    EXPECT_EQ( countinghandler->getMessageCountFor(Message::Warning), 0) ;
    EXPECT_EQ( countinghandler->getMessageCountFor(Message::Info), 0) ;

    delete consolehandler ;
}
