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
#include <SofaTest/Sofa_test.h>
using sofa::Sofa_test ;

#include <sofa/defaulttype/Vec.h>
using sofa::defaulttype::Vec3f;

#include <sofa/simulation/Node.h>
using sofa::simulation::Node ;
using sofa::core::ExecParams;

#include <sofa/core/objectmodel/Base.h>
using sofa::core::objectmodel::Base;

#include <sofa/core/objectmodel/Data.h>
using sofa::core::objectmodel::Data;
using sofa::core::objectmodel::BaseData;

#include <sofa/helper/vectorData.h>
using sofa::helper::vectorData;
using sofa::helper::WriteAccessorVector;
using sofa::helper::WriteAccessor;
using sofa::helper::ReadAccessor;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

#include <sofa/simulation/Simulation.h>
using sofa::simulation::Simulation ;

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory ;

// COMMUNICATION PART
#include <Communication/components/serverCommunication.h>
#include <Communication/components/serverCommunicationOSC.h>
#include <Communication/components/serverCommunicationZMQ.h>
#include <Communication/components/CommunicationSubscriber.h>
using sofa::component::communication::ServerCommunication;
using sofa::component::communication::ServerCommunicationOSC;
using sofa::component::communication::ServerCommunicationZMQ;
using sofa::component::communication::CommunicationSubscriber;

// OSC TEST PART
#include <oscpack/osc/OscReceivedElements.h>
#include <oscpack/osc/OscPrintReceivedElements.h>
#include <oscpack/osc/OscPacketListener.h>
#include <oscpack/osc/OscOutboundPacketStream.h>

#include <oscpack/ip/UdpSocket.h>
#include <oscpack/ip/PacketListener.h>

// ZMQ TEST PART
#include <zmq.hpp>

// TIMEOUT
#include <iostream>
#include <future>
#include <thread>
#include <chrono>

namespace sofa
{
namespace component
{
namespace communication
{

using sofa::defaulttype::Vector3 ;

class MyComponent : public BaseObject
{
public:
    MyComponent() :
        d_vectorIn(initData (&d_vectorIn, "vectorIn", ""))
      , d_vectorOut(initData (&d_vectorOut, "vectorOut", ""))
    {
        f_listening = true ;
    }

    virtual void init() override
    {
        std::stringstream ss;
        for (int i =0; i <100; i++)
            ss << std::to_string(static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1.0))) << " ";
        d_vectorOut.read(ss.str());
    }

    virtual void handleEvent(sofa::core::objectmodel::Event *event) override
    {
        //        std::cout << "event " << std::endl;
        //        for(Data<Vec3f>* t : d_positionsOut)
        //        {
        //            Vec3f a;
        //            a.at(0) = a.at(0)+1.0f;
        //            a.at(1) = a.at(1)+1.0f;
        //            a.at(2) = a.at(2)+1.0f;
        //            t->setValue(a);
        //        }
    }

    //    vectorData<Vec3f> d_pos;
    Data<vector<float>> d_vectorIn;
    Data<vector<float>> d_vectorOut;
} ;

int mclass = sofa::core::RegisterObject("").add<MyComponent>();

class Communication_test : public Sofa_test<>
{
private:

    class OscDumpPacketListener : public osc::OscPacketListener{
    public:
        std::vector<std::string> m_data;
        UdpListeningReceiveSocket* m_socket;

        virtual void ProcessMessage( const osc::ReceivedMessage& m, const IpEndpointName& remoteEndpoint )
        {
            (void) remoteEndpoint;

            osc::ReceivedMessageArgumentIterator it = m.ArgumentsBegin();
            for ( it ; it != m.ArgumentsEnd(); it++)
            {
                std::stringstream stream;
                stream << *it;
                m_data.push_back(stream.str());
            }
            m_socket->Break();
        }

        void setSocket(UdpListeningReceiveSocket* s) { m_socket = s;}
    };

public:

    /// BASIC STUFF TEST
    /// testing subscriber + argument creation
    /// it uses OSC but those tested functions are the same for OSC/ZMQ

    void checkAddSubscriber()
    {
        std::stringstream scene1 ;
        scene1 <<
                  "<?xml version='1.0' ?>                                                       \n"
                  "<Node name='root'>                                                           \n"
                  "   <RequiredPlugin name='Communication' />                                   \n"
                  "   <ServerCommunicationOSC name='oscSender' job='sender' port='6000'  refreshRate='1000'/> \n"
                  "   <CommunicationSubscriber name='sub1' communication='@oscSender' subject='/test' source='@oscSender' arguments='x'/>"
                  "</Node>                                                                      \n";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
        root->init(ExecParams::defaultInstance()) ;

        ServerCommunication* aServerCommunicationOSC = dynamic_cast<ServerCommunication*>(root->getObject("oscSender"));
        std::map<std::string, CommunicationSubscriber*> map = aServerCommunicationOSC->getSubscribers();
        EXPECT_EQ(map.size(), 1);
    }

    void checkGetSubscriber()
    {
        std::stringstream scene1 ;
        scene1 <<
                  "<?xml version='1.0' ?>                                                       \n"
                  "<Node name='root'>                                                           \n"
                  "   <RequiredPlugin name='Communication' />                                   \n"
                  "   <ServerCommunicationOSC name='oscSender' job='sender' port='6000'  refreshRate='1000'/> \n"
                  "   <CommunicationSubscriber name='sub1' communication='@oscSender' subject='/test' source='@oscSender' arguments='x'/>"
                  "</Node>                                                                      \n";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
        root->init(ExecParams::defaultInstance()) ;

        ServerCommunication* aServerCommunicationOSC = dynamic_cast<ServerCommunication*>(root->getObject("oscSender"));
        CommunicationSubscriber* subscriber = aServerCommunicationOSC->getSubscriberFor("/test");
        EXPECT_NE(subscriber, nullptr) ;
    }

    void checkArgumentCreation()
    {
        std::stringstream scene1 ;
        scene1 <<
                  "<?xml version='1.0' ?>                                                       \n"
                  "<Node name='root'>                                                           \n"
                  "   <RequiredPlugin name='Communication' />                                   \n"
                  "   <ServerCommunicationOSC name='oscSender' job='sender' port='6000' refreshRate='1000'/> \n"
                  "   <CommunicationSubscriber name='sub1' communication='@oscSender' subject='/test' source='@oscSender' arguments='x'/>"
                  "</Node>                                                                      \n";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
        root->init(ExecParams::defaultInstance());
        ServerCommunication* aServerCommunicationOSC = dynamic_cast<ServerCommunication*>(root->getObject("oscSender"));
        EXPECT_NE(aServerCommunicationOSC, nullptr);

        usleep(10000);

        Base::MapData dataMap = aServerCommunicationOSC->getDataAliases();
        Base::MapData::const_iterator itData;
        BaseData* data;

        itData = dataMap.find("port");
        EXPECT_TRUE(itData != dataMap.end());
        if (itData != dataMap.end())
        {
            data = itData->second;
            EXPECT_NE(data, nullptr) ;
        }

        itData = dataMap.find("x");
        EXPECT_TRUE(itData != dataMap.end());
        if (itData != dataMap.end())
        {
            data = itData->second;
            EXPECT_NE(data, nullptr) ;
        }
    }

    /// OSC TEST PART
    /// testing basic use + specific functions such as parsing

    void checkCreationDestruction()
    {
        std::stringstream scene1 ;
        scene1 <<
                  "<?xml version='1.0' ?>                                                       \n"
                  "<Node name='root'>                                                           \n"
                  "   <RequiredPlugin name='Communication' />                                   \n"
                  "   <ServerCommunicationOSC name='oscSender' job='sender' port='6000'  refreshRate='1000'/> \n"
                  "</Node>                                                                      \n";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
        root->init(ExecParams::defaultInstance()) ;
    }

    void checkSendOSC()
    {
        std::stringstream scene1 ;
        scene1 <<
                  "<?xml version='1.0' ?>                                                       \n"
                  "<Node name='root'>                                                           \n"
                  "   <RequiredPlugin name='Communication' />                                   \n"
                  "   <ServerCommunicationOSC name='oscSender' job='sender' port='6000'  refreshRate='1000'/> \n"
                  "   <CommunicationSubscriber name='sub1' communication='@oscSender' subject='/test' source='@oscSender' arguments='x'/>"
                  "</Node>                                                                      \n";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
        root->init(ExecParams::defaultInstance()) ;

        OscDumpPacketListener listener;
        UdpListeningReceiveSocket s(
                    IpEndpointName( IpEndpointName::ANY_ADDRESS, 6000 ),
                    &listener );
        listener.setSocket(&s);
        s.Run();
        EXPECT_EQ(listener.m_data.size(), 1);
    }

    void checkReceiveOSC()
    {
        std::stringstream scene1 ;
        scene1 <<
                  "<?xml version='1.0' ?>                                                       \n"
                  "<Node name='root'>                                                           \n"
                  "   <RequiredPlugin name='Communication' />                                   \n"
                  "   <ServerCommunicationOSC name='oscReceiver' job='receiver' port='6000' /> \n"
                  "   <CommunicationSubscriber name='sub1' communication='@oscReceiver' subject='/test' source='@oscReceiver' arguments='x'/>"
                  "</Node>                                                                      \n";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
        root->init(ExecParams::defaultInstance());
        ServerCommunication* aServerCommunicationOSC = dynamic_cast<ServerCommunication*>(root->getObject("oscReceiver"));
        aServerCommunicationOSC->setRunning(false);
        usleep(10000);
        UdpTransmitSocket transmitSocket( IpEndpointName( "127.0.0.1", 6000 ) );
        char buffer[1024];
        osc::OutboundPacketStream p(buffer, 1024 );
        p << osc::BeginBundleImmediate;
        p << osc::BeginMessage("/test");
        p << "";
        p << osc::EndMessage;
        p << osc::EndBundle;
        transmitSocket.Send( p.Data(), p.Size() );
        usleep(10000);


        Base::MapData dataMap = aServerCommunicationOSC->getDataAliases();
        Base::MapData::const_iterator itData = dataMap.find("x");
        BaseData* data;

        EXPECT_TRUE(itData != dataMap.end());
        if (itData != dataMap.end())
        {
            data = itData->second;
            EXPECT_NE(data, nullptr) ;
        }
    }

    void checkSendReceiveOSC()
    {

        std::stringstream scene1 ;
        scene1 <<
                  "<?xml version='1.0' ?>                                                       \n"
                  "<Node name='root'>                                                           \n"
                  "   <RequiredPlugin name='Communication' />                                   \n"
                  "   <ServerCommunicationOSC name='oscSender' job='sender' port='6000'  refreshRate='1000'/> \n"
                  "   <CommunicationSubscriber name='subSender' communication='@oscSender' subject='/test' source='@oscSender' arguments='x'/>"
                  "   <ServerCommunicationOSC name='oscReceiver' job='receiver' port='6000' /> \n"
                  "   <CommunicationSubscriber name='subReceiver' communication='@oscReceiver' subject='/test' source='@oscReceiver' arguments='x'/>"
                  "</Node>                                                                      \n";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
        root->init(ExecParams::defaultInstance()) ;

        ServerCommunication* aServerCommunicationOSCSender = dynamic_cast<ServerCommunication*>(root->getObject("oscSender"));
        ServerCommunication* aServerCommunicationOSCReceiver = dynamic_cast<ServerCommunication*>(root->getObject("oscReceiver"));
        EXPECT_NE(aServerCommunicationOSCSender, nullptr);
        EXPECT_NE(aServerCommunicationOSCReceiver, nullptr);

        aServerCommunicationOSCReceiver->setRunning(false);

        usleep(10000);

        Base::MapData dataMap = aServerCommunicationOSCSender->getDataAliases();
        Base::MapData::const_iterator itData;
        BaseData* data;

        itData = dataMap.find("x");
        EXPECT_TRUE(itData != dataMap.end());
        if (itData != dataMap.end())
        {
            data = itData->second;
            EXPECT_NE(data, nullptr) ;
        }

        dataMap = aServerCommunicationOSCSender->getDataAliases();
        itData = dataMap.find("x");
        EXPECT_TRUE(itData != dataMap.end());
        if (itData != dataMap.end())
        {
            data = itData->second;
            EXPECT_NE(data, nullptr) ;
        }
    }


    /// ZMQ TEST PART
    /// testing basic use + specific functions such as parsing

    void checkSendZMQ()
    {
        std::stringstream scene1 ;
        scene1 <<
                  "<?xml version='1.0' ?>                                                       \n"
                  "<Node name='root'>                                                           \n"
                  "   <RequiredPlugin name='Communication' />                                   \n"
                  "   <ServerCommunicationZMQ name='sender' job='sender' port='6000' pattern='publish/subscribe' refreshRate='100'/> \n"
                  "   <CommunicationSubscriber name='subSender' communication='@sender' subject='/test' source='@sender' arguments='port'/>"

                  "</Node>                                                                      \n";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
        root->init(ExecParams::defaultInstance()) ;


        std::future<std::string> future = std::async(std::launch::async, [](){
            zmq::context_t context (1);
            zmq::socket_t socket (context, ZMQ_SUB);
            socket.connect ("tcp://localhost:6000");
            socket.setsockopt(ZMQ_SUBSCRIBE, "", 0);
            int timeout = 3000; // timeout
            socket.setsockopt(ZMQ_RCVTIMEO, &timeout, sizeof (int));

            zmq::message_t reply;
            socket.recv (&reply);
            std::string rpl = std::string(static_cast<char*>(reply.data()), reply.size());

            return rpl;
        });

        std::future_status status;
        status = future.wait_for(std::chrono::seconds(3));
        EXPECT_EQ(status, std::future_status::ready);
    }

    void checkReceiveZMQ()
    {
        std::stringstream scene1 ;
        scene1 <<
                  "<?xml version='1.0' ?>                                                       \n"
                  "<Node name='root'>                                                           \n"
                  "   <RequiredPlugin name='Communication' />                                   \n"
                  "   <ServerCommunicationZMQ name='receiver' job='receiver' port='6000' pattern='publish/subscribe'/> \n"
                  "   <CommunicationSubscriber name='subSender' communication='@receiver' subject='/test' source='@receiver' arguments='x'/>"
                  "</Node>                                                                      \n";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
        root->init(ExecParams::defaultInstance());
        ServerCommunication* aServerCommunication = dynamic_cast<ServerCommunication*>(root->getObject("receiver"));
        aServerCommunication->setRunning(false);

        usleep(10000);
        zmq::context_t context (1);
        zmq::socket_t socket (context, ZMQ_PUB);
        socket.bind ("tcp://*:6000");
        for(int i = 0; i <3; i++) // ensure the receiver, receive it at least once
        {
            std::string mesg = "/test ";
            mesg += "int:" + std::to_string(i);
            zmq::message_t reply (mesg.size());
            memcpy (reply.data (), mesg.c_str(), mesg.size());
            socket.send (reply);
            usleep(100000);
        }
        usleep(10000);

        Base::MapData dataMap = aServerCommunication->getDataAliases();
        Base::MapData::const_iterator itData = dataMap.find("x");
        BaseData* data;

        EXPECT_TRUE(itData != dataMap.end());
        if (itData != dataMap.end())
        {
            data = itData->second;
            EXPECT_NE(data, nullptr) ;
        }
    }

    void checkSendReceiveZMQ()
    {

        std::stringstream scene1 ;
        scene1 <<
                  "<?xml version='1.0' ?>                                                       \n"
                  "<Node name='root'>                                                           \n"
                  "   <RequiredPlugin name='Communication' />                                   \n"
                  "   <ServerCommunicationZMQ name='sender' job='sender' port='6000'  refreshRate='1000'/> \n"
                  "   <CommunicationSubscriber name='subSender' communication='@sender' subject='/test' source='@sender' arguments='x'/>"

                  "   <ServerCommunicationZMQ name='receiver' job='receiver' port='6000' /> \n"
                  "   <CommunicationSubscriber name='subReceiver' communication='@receiver' subject='/test' source='@receiver' arguments='x'/>"
                  "</Node>                                                                      \n";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
        root->init(ExecParams::defaultInstance()) ;

        ServerCommunication* aServerCommunicationSender = dynamic_cast<ServerCommunication*>(root->getObject("sender"));
        ServerCommunication* aServerCommunicationReceiver = dynamic_cast<ServerCommunication*>(root->getObject("receiver"));
        EXPECT_NE(aServerCommunicationSender, nullptr);
        EXPECT_NE(aServerCommunicationReceiver, nullptr);

        aServerCommunicationReceiver->setRunning(false);

        usleep(100000);

        Base::MapData dataMap = aServerCommunicationSender->getDataAliases();
        Base::MapData::const_iterator itData;
        BaseData* data;

        itData = dataMap.find("x");
        EXPECT_TRUE(itData != dataMap.end());
        if (itData != dataMap.end())
        {
            data = itData->second;
            EXPECT_NE(data, nullptr) ;
        }

        dataMap = aServerCommunicationSender->getDataAliases();
        itData = dataMap.find("x");
        EXPECT_TRUE(itData != dataMap.end());
        if (itData != dataMap.end())
        {
            data = itData->second;
            EXPECT_NE(data, nullptr) ;
        }
    }

    void checkZMQParsingFunctions()
    {
        ServerCommunicationZMQ * zmqServer = new ServerCommunicationZMQ();
        EXPECT_STREQ(zmqServer->getArgumentType("string:''").c_str(), "string");
        EXPECT_STREQ(zmqServer->getArgumentType("string:'toto'").c_str(), "string");
        EXPECT_STREQ(zmqServer->getArgumentType("string:'toto blop'").c_str(), "string");
        EXPECT_STREQ(zmqServer->getArgumentType("test").c_str(), "string");

        EXPECT_STREQ(zmqServer->getArgumentValue("string:''").c_str(), "");
        EXPECT_STREQ(zmqServer->getArgumentValue("string:'toto'").c_str(), "toto");
        EXPECT_STREQ(zmqServer->getArgumentValue("string:'toto blop'").c_str(), "toto blop");
        EXPECT_STREQ(zmqServer->getArgumentValue("test").c_str(), "test");

        std::vector<std::string> argumentList;

        argumentList = zmqServer->stringToArgumentList("/test string:'toto' int:26");
        EXPECT_EQ(argumentList.size(), 3);

        argumentList = zmqServer->stringToArgumentList("/test string:'toto tata' string:'toto' int:26");
        EXPECT_EQ(argumentList.size(), 4);

        argumentList = zmqServer->stringToArgumentList("/test string:'toto tata titi' string:'toto' int:26");
        EXPECT_EQ(argumentList.size(), 4);

        argumentList = zmqServer->stringToArgumentList("/test string:'toto tata string:'toto' int:26");
        EXPECT_EQ(argumentList.size(), 4);

        argumentList = zmqServer->stringToArgumentList("/test string:'toto tata int:26");
        EXPECT_EQ(argumentList.size(), 2);
    }

    void checkThreadSafeZMQ()
    {
        std::stringstream scene1 ;
        scene1 <<
                  "<?xml version='1.0' ?>                                                       \n"
                  "<Node  name='Root' gravity='0 0 0' time='0' animate='0'   >                  \n"
                  "   <RequiredPlugin name='Communication' />                                   \n"
                  "   <MyComponent name='aComponent' />                                         \n"
                  "   <ServerCommunicationZMQ name='Sender' job='sender' port='6000' pattern='publish/subscribe' refreshRate='100000'/> \n"
                  "   <CommunicationSubscriber name='subSender' communication='@Sender' subject='/test' source='@aComponent' arguments='vectorOut'/>"
                  "   <ServerCommunicationZMQ name='Receiver' job='receiver' port='6000' /> \n"
                  "   <CommunicationSubscriber name='subReceiver' communication='@Receiver' subject='/test' source='@aComponent' arguments='vectorIn'/>"
                  "</Node>                                                                      \n";


        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
        root->init(ExecParams::defaultInstance()) ;
        root->setName("root");

        ServerCommunication* aServerCommunicationSender = dynamic_cast<ServerCommunication*>(root->getObject("Sender"));
        ServerCommunication* aServerCommunicationReceiver = dynamic_cast<ServerCommunication*>(root->getObject("Receiver"));
        MyComponent* aComponent = dynamic_cast<MyComponent*>(root->getObject("aComponent"));

        EXPECT_NE(aServerCommunicationSender, nullptr);
        EXPECT_NE(aServerCommunicationReceiver, nullptr);
        EXPECT_NE(aComponent, nullptr);


        for(unsigned int i=0; i<10000; i++)
        {
            //            for(Data<Vec3f>* a : aComponent->d_positionsIn)
            //            {
            //                Vec3f value = a->getValue();
            //                std::cout << value << std::endl;
            //                EXPECT_FLOAT_EQ(value.at(0), value.at(1));
            //                EXPECT_FLOAT_EQ(value.at(1), value.at(2));
            //            }
            //            sofa::simulation::getSimulation()->animate(root.get(), 0.01);
        }


        aServerCommunicationReceiver->setRunning(false);
        usleep(10000);
        aServerCommunicationSender->setRunning(false);
        usleep(10000);

        EXPECT_STREQ(aComponent->d_vectorIn.getValueString().c_str(), aComponent->d_vectorOut.getValueString().c_str()); // crappy but it's not the final test. The test have to be inside the animation loop and test this easch step

    }
};

TEST_F(Communication_test, checkCreationDestruction) {
    ASSERT_NO_THROW(this->checkCreationDestruction()) ;
}

TEST_F(Communication_test, checkAddSubscriber) {
    ASSERT_NO_THROW(this->checkAddSubscriber()) ;
}

TEST_F(Communication_test, checkGetSubscriber) {
    ASSERT_NO_THROW(this->checkGetSubscriber()) ;
}

TEST_F(Communication_test, checkSendOSC) {
    ASSERT_NO_THROW(this->checkSendOSC()) ;
}

TEST_F(Communication_test, checkReceiveOSC) {
    ASSERT_NO_THROW(this->checkReceiveOSC()) ;
}

TEST_F(Communication_test, checkSendReceiveOSC) {
    ASSERT_NO_THROW(this->checkSendReceiveOSC()) ;
}

TEST_F(Communication_test, checkArgumentCreation) {
    ASSERT_NO_THROW(this->checkArgumentCreation()) ;
}

TEST_F(Communication_test, checkSendZMQ) {
    ASSERT_NO_THROW(this->checkSendZMQ()) ;
}

TEST_F(Communication_test, checkReceiveZMQ) {
    ASSERT_NO_THROW(this->checkReceiveZMQ()) ;
}

TEST_F(Communication_test, checkSendReceiveZMQ) {
    ASSERT_NO_THROW(this->checkSendReceiveZMQ()) ;
}

TEST_F(Communication_test, checkZMQParsingFunctions) {
    ASSERT_NO_THROW(this->checkZMQParsingFunctions()) ;
}

TEST_F(Communication_test, checkThreadSafeZMQ) {
    ASSERT_NO_THROW(this->checkThreadSafeZMQ()) ;
}

} // communication
} // component
} // sofa
