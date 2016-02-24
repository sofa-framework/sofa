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

//        if( !m.sender().empty() )
//            std::cerr<<m<<std::endl;
    }

    int numMessages(){
        return m_messages.size() ;
    }

    const vector<Message>& messages() const {
        return m_messages;
    }
    const Message& lastMessage() const {
        return m_messages.back();
    }
} ;


class MyComponent : public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS( MyComponent, sofa::core::objectmodel::BaseObject );
protected:
    MyComponent()
    {        
//        std::cerr<<SOFA_CLASS_METHOD<<"constructor"<<std::endl;
        f_printLog.setValue(true); // to print sout
//        std::cerr<<SOFA_CLASS_METHOD<<"regular serr"<<std::endl;
        serr<<"regular serr"<<sendl;
//        std::cerr<<SOFA_CLASS_METHOD<<"regular sout"<<std::endl;
        sout<<"regular sout"<<sendl;
//        std::cerr<<SOFA_CLASS_METHOD<<"serr with fileinfo"<<std::endl;
        serr<<SOFA_FILE_INFO<<"serr with fileinfo"<<sendl;
//        std::cerr<<SOFA_CLASS_METHOD<<"sout with fileinfo"<<std::endl;
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

    MyMessageHandler h;

    // add is expected to return the handler ID. Here is it the o'th
    EXPECT_TRUE(MessageDispatcher::addHandler(&h) == 0 ) ;

    msg_info("") << " info message with conversion" << 1.5 << "\n" ;
    msg_warning("") << " warning message with conversion "<< 1.5 << "\n" ;
    msg_error("") << " error message with conversion" << 1.5 << "\n" ;

    EXPECT_TRUE( h.numMessages() == 3 ) ;
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
    msg_warning("") << " warning message with conversion "<< 1.5 << "\n" ;
    msg_error("") << " error message with conversion" << 1.5 << "\n" ;

    EXPECT_TRUE( h.numMessages() == 3) ;
}


TEST(LoggingTest, withoutDevMode)
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

    EXPECT_TRUE( h.numMessages() == 3 ) ;
}

//TEST(LoggingTest, speedTest)
//{
//    MessageDispatcher::clearHandlers() ;

//    MyMessageHandler h;
//    MessageDispatcher::addHandler(&h) ;

//    for(unsigned int i=0;i<10000;i++){
//        msg_info("") << " info message with conversion" << 1.5 << "\n" ;
//        msg_warning("") << " warning message with conversion "<< 1.5 << "\n" ;
//        msg_error("") << " error message with conversion" << 1.5 << "\n" ;
//    }
//}



TEST(LoggingTest, emptyMessage)
{
    MessageDispatcher::clearHandlers() ;
    MyMessageHandler h;
    MessageDispatcher::addHandler(&h) ;

    // an empty message should not be processed

    msg_info("");
    EXPECT_EQ( h.numMessages(), 0 );

    msg_info("")<<"ok";
    msg_info("");
    EXPECT_EQ( h.numMessages(), 1 );
}

TEST(LoggingTest, BaseObject)
{
    MessageDispatcher::clearHandlers() ;
    MyMessageHandler h;
    MessageDispatcher::addHandler(&h) ;


    typename MyComponent::SPtr c = sofa::core::objectmodel::New<MyComponent>();


    /// the constructor of MyComponent is sending 4 messages
    EXPECT_EQ( h.numMessages(), 4 ) ;

    c->serr<<"regular external serr"<<c->sendl;
    EXPECT_EQ( h.lastMessage().fileInfo().line, 0 );
    EXPECT_TRUE( !strcmp( h.lastMessage().fileInfo().filename, sofa::helper::logging::s_unknownFile ) );
    EXPECT_EQ( h.lastMessage().type(), sofa::helper::logging::Message::Warning );

    c->serr<<sofa::helper::logging::Message::Error<<"external serr as Error"<<c->sendl;
    EXPECT_EQ( h.lastMessage().fileInfo().line, 0 );
    EXPECT_TRUE( !strcmp( h.lastMessage().fileInfo().filename, sofa::helper::logging::s_unknownFile ) );
    EXPECT_EQ( h.lastMessage().type(), sofa::helper::logging::Message::Error );

    c->sout<<"regular external sout"<<c->sendl;
    EXPECT_EQ( h.lastMessage().fileInfo().line, 0 );
    EXPECT_TRUE( !strcmp( h.lastMessage().fileInfo().filename, sofa::helper::logging::s_unknownFile ) );
    EXPECT_EQ( h.lastMessage().type(), sofa::helper::logging::Message::Info );

    c->sout<<sofa::helper::logging::Message::Error<<"external sout as Error"<<c->sendl;
    EXPECT_EQ( h.lastMessage().fileInfo().line, 0 );
    EXPECT_TRUE( !strcmp( h.lastMessage().fileInfo().filename, sofa::helper::logging::s_unknownFile ) );
    EXPECT_EQ( h.lastMessage().type(), sofa::helper::logging::Message::Error );


    c->serr<<SOFA_FILE_INFO<<"external serr with fileinfo"<<c->sendl;
    EXPECT_EQ( h.lastMessage().fileInfo().line, __LINE__-1 );
    EXPECT_TRUE( !strcmp( h.lastMessage().fileInfo().filename, __FILE__ ) );

    c->sout<<SOFA_FILE_INFO<<"external sout with fileinfo"<<c->sendl;
    EXPECT_EQ( h.lastMessage().fileInfo().line, __LINE__-1 );
    EXPECT_TRUE( !strcmp( h.lastMessage().fileInfo().filename, __FILE__ ) );

    c->serr<<SOFA_FILE_INFO<<sofa::helper::logging::Message::Error<<"external serr as Error with fileinfo"<<c->sendl;
    EXPECT_EQ( h.lastMessage().fileInfo().line, __LINE__-1 );
    EXPECT_TRUE( !strcmp( h.lastMessage().fileInfo().filename, __FILE__ ) );
    EXPECT_EQ( h.lastMessage().type(), sofa::helper::logging::Message::Error );

    c->sout<<sofa::helper::logging::Message::Error<<SOFA_FILE_INFO<<"external sout as Error with fileinfo"<<c->sendl;
    EXPECT_EQ( h.lastMessage().fileInfo().line, __LINE__-1 );
    EXPECT_TRUE( !strcmp( h.lastMessage().fileInfo().filename, __FILE__ ) );
    EXPECT_EQ( h.lastMessage().type(), sofa::helper::logging::Message::Error );

    c->serr<<"serr with sendl that comes in a second time";
    c->serr<<c->sendl;
    EXPECT_EQ( h.lastMessage().fileInfo().line, 0 );
    EXPECT_TRUE( !strcmp( h.lastMessage().fileInfo().filename, sofa::helper::logging::s_unknownFile ) );
    EXPECT_EQ( h.lastMessage().type(), sofa::helper::logging::Message::Warning );

    c->serr<<"\n serr with \n end of "<<std::endl<<" lines"<<c->sendl;
    EXPECT_EQ( h.lastMessage().fileInfo().line, 0 );
    EXPECT_TRUE( !strcmp( h.lastMessage().fileInfo().filename, sofa::helper::logging::s_unknownFile ) );
    EXPECT_EQ( h.lastMessage().type(), sofa::helper::logging::Message::Warning );

    EXPECT_EQ( h.numMessages(), 14 ) ;


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

    EXPECT_TRUE( h.numMessages() == 6 ) ;
}
