#include <gtest/gtest.h>
#include <exception>
#include <algorithm>

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

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/ObjectFactory.h>


class MyMessageHandler : public MessageHandler
{
    vector<Message> m_messages ;
public:
    virtual void process(Message& m){
        m_messages.push_back(m);

        if( !m.sender().empty() )
            //std::cerr<<m.message().rdbuf()<<" "<<m.fileInfo().filename<<" "<<m.fileInfo().line<<std::endl;
            std::cerr<<m<<std::endl;
    }

    int numMessages(){
        return m_messages.size() ;
    }
} ;


class MyComponent : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS( MyComponent, sofa::core::objectmodel::BaseObject );
    MyComponent()
    {
        f_printLog.setValue(true); // to print sout
        serr<<"regular serr"<<sendl;
        sout<<"regular sout"<<sendl;
        serr<<SOFA_FILE_INFO<<"serr with fileinfo"<<sendl;
        sout<<SOFA_FILE_INFO<<"sout with fileinfo"<<sendl;
    }
};

SOFA_DECL_CLASS(MyComponent)
int MyComponentClass = sofa::core::RegisterObject("MyComponent")
        .add< MyComponent >();

TEST(LoggingTest, noHandler)
{
    MessageDispatcher::clearHandlers() ;

    msg_info("") << " info message with conversion" << 1.5 << "\n" ;
    msg_warning("") << " warning message with conversion "<< 1.5 << "\n" ;
    msg_error("") << " error message with conversion" << 1.5 << "\n" ;
}


TEST(LoggingTest, oneHandler)
{
    MessageDispatcher::clearHandlers() ;

    MyMessageHandler *h=new MyMessageHandler() ;

    // add is expected to return the handler ID. Here is it the o'th
    EXPECT_TRUE(MessageDispatcher::addHandler(h) == 0 ) ;

    msg_info("") << " info message with conversion" << 1.5 << "\n" ;
    msg_warning("") << " warning message with conversion "<< 1.5 << "\n" ;
    msg_error("") << " error message with conversion" << 1.5 << "\n" ;

    EXPECT_TRUE( h->numMessages() == 3 ) ;
}

TEST(LoggingTest, duplicatedHandler)
{
    MessageDispatcher::clearHandlers() ;

    MyMessageHandler *h=new MyMessageHandler() ;

    // First add is expected to return the handler ID.
    EXPECT_TRUE(MessageDispatcher::addHandler(h) == 0) ;

    // Second is supposed to fail to add and thus return -1.
    EXPECT_TRUE(MessageDispatcher::addHandler(h) == -1) ;

    msg_info("") << " info message with conversion" << 1.5 << "\n" ;
    msg_warning("") << " warning message with conversion "<< 1.5 << "\n" ;
    msg_error("") << " error message with conversion" << 1.5 << "\n" ;

    EXPECT_TRUE( h->numMessages() == 3) ;
}


TEST(LoggingTest, withoutDevMode)
{
    MessageDispatcher::clearHandlers() ;

    MyMessageHandler *h=new MyMessageHandler() ;
    MessageDispatcher::addHandler(h) ;

    msg_info("") << " info message with conversion" << 1.5 << "\n" ;
    msg_warning("") << " warning message with conversion "<< 1.5 << "\n" ;
    msg_error("") << " error message with conversion" << 1.5 << "\n" ;

    nmsg_info("") << " null info message with conversion" << 1.5 << "\n" ;
    nmsg_warning("") << " null warning message with conversion "<< 1.5 << "\n" ;
    nmsg_error("") << " null error message with conversion" << 1.5 << "\n" ;

    EXPECT_TRUE( h->numMessages() == 3 ) ;
}

//TEST(LoggingTest, speedTest)
//{
//    MessageDispatcher::clearHandlers() ;

//    MyMessageHandler *h=new MyMessageHandler() ;
//    MessageDispatcher::addHandler(h) ;

//    for(unsigned int i=0;i<10000;i++){
//        msg_info("") << " info message with conversion" << 1.5 << "\n" ;
//        msg_warning("") << " warning message with conversion "<< 1.5 << "\n" ;
//        msg_error("") << " error message with conversion" << 1.5 << "\n" ;
//    }
//}


TEST(LoggingTest, BaseObject)
{
    MessageDispatcher::clearHandlers() ;
    MyMessageHandler *h=new MyMessageHandler() ;
    MessageDispatcher::addHandler(h) ;

    MyComponent c;
}

#undef MESSAGING_H
#define WITH_SOFA_DEVTOOLS
#undef dmsg_info
#undef dmsg_error
#undef dmsg_warning
#undef dmsg_fatal
#include <sofa/helper/logging/Messaging.h>

TEST(LoggingTest, withDevMode)
{
    MessageDispatcher::clearHandlers() ;

    MyMessageHandler *h=new MyMessageHandler() ;
    MessageDispatcher::addHandler(h) ;

    msg_info("") << " info message with conversion" << 1.5 << "\n" ;
    msg_warning("") << " warning message with conversion "<< 1.5 << "\n" ;
    msg_error("") << " error message with conversion" << 1.5 << "\n" ;

    nmsg_info("") << " null info message with conversion" << 1.5 << "\n" ;
    nmsg_warning("") << " null warning message with conversion "<< 1.5 << "\n" ;
    nmsg_error("") << " null error message with conversion" << 1.5 << "\n" ;

    dmsg_info("") << " debug info message with conversion" << 1.5 << "\n" ;
    dmsg_warning("") << " debug warning message with conversion "<< 1.5 << "\n" ;
    dmsg_error("") << " debug error message with conversion" << 1.5 << "\n" ;

    EXPECT_TRUE( h->numMessages() == 6 ) ;
}
