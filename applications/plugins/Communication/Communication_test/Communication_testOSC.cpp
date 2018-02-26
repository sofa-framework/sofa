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
#include <Communication/components/CommunicationSubscriber.h>
using sofa::component::communication::ServerCommunication;
using sofa::component::communication::ServerCommunicationOSC;
using sofa::component::communication::CommunicationSubscriber;

// OSC TEST PART
#include <oscpack/osc/OscReceivedElements.h>
#include <oscpack/osc/OscPrintReceivedElements.h>
#include <oscpack/osc/OscPacketListener.h>
#include <oscpack/osc/OscOutboundPacketStream.h>

#include <oscpack/ip/UdpSocket.h>
#include <oscpack/ip/PacketListener.h>

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

class Communication_testOSC : public Sofa_test<>
{
private:

    class OscDumpPacketListener : public osc::OscPacketListener{
    public:
        std::vector<std::string> m_data;
        UdpListeningReceiveSocket* m_socket;

        virtual void ProcessMessage( const osc::ReceivedMessage& m, const IpEndpointName& remoteEndpoint )
        {
            (void) remoteEndpoint;

            for (osc::ReceivedMessageArgumentIterator it = m.ArgumentsBegin() ; it != m.ArgumentsEnd(); it++)
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
    /// it uses OSC but those tested functions are the same for OSC

    void checkAddSubscriber()
    {
        std::stringstream scene1 ;
        scene1 <<
                  "<?xml version='1.0' ?>                                                       \n"
                  "<Node name='root'>                                                           \n"
                  "   <DefaultAnimationLoop/>                                                   \n"
                  "   <RequiredPlugin name='Communication' />                                   \n"
                  "   <ServerCommunicationOSC name='oscSender' job='sender' port='6000'  refreshRate='1000'/> \n"
                  "   <CommunicationSubscriber name='sub1' communication='@oscSender' subject='/test' target='@oscSender' datas='x'/>"
                  "</Node>                                                                      \n";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
        root->init(ExecParams::defaultInstance()) ;

        ServerCommunication* aServerCommunicationOSC = dynamic_cast<ServerCommunication*>(root->getObject("oscSender"));
        std::map<std::string, CommunicationSubscriber*> map = aServerCommunicationOSC->getSubscribers();
        EXPECT_EQ((int)map.size(), 1);
    }

    void checkGetSubscriber()
    {
        std::stringstream scene1 ;
        scene1 <<
                  "<?xml version='1.0' ?>                                                       \n"
                  "<Node name='root'>                                                           \n"
                  "   <DefaultAnimationLoop/>                                                   \n"
                  "   <RequiredPlugin name='Communication' />                                   \n"
                  "   <ServerCommunicationOSC name='oscSender' job='sender' port='6000'  refreshRate='1000'/> \n"
                  "   <CommunicationSubscriber name='sub1' communication='@oscSender' subject='/test' target='@oscSender' datas='x'/>"
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
                  "   <DefaultAnimationLoop/>                                                   \n"
                  "   <RequiredPlugin name='Communication' />                                   \n"
                  "   <ServerCommunicationOSC name='oscSender' job='sender' port='6000' refreshRate='1000'/> \n"
                  "   <CommunicationSubscriber name='sub1' communication='@oscSender' subject='/test' target='@oscSender' datas='x'/>"
                  "</Node>                                                                      \n";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
        root->init(ExecParams::defaultInstance());

        ServerCommunication* aServerCommunicationOSC = dynamic_cast<ServerCommunication*>(root->getObject("oscSender"));
        EXPECT_NE(aServerCommunicationOSC, nullptr);

        for(unsigned int i=0; i<10; i++)
            sofa::simulation::getSimulation()->animate(root.get(), 0.01);

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

    void checkCreationDestruction()
    {
        std::stringstream scene1 ;
        scene1 <<
                  "<?xml version='1.0' ?>                                                       \n"
                  "<Node name='root'>                                                           \n"
                  "   <DefaultAnimationLoop/>                                                   \n"
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
                  "   <DefaultAnimationLoop/>                                                   \n"
                  "   <RequiredPlugin name='Communication' />                                   \n"
                  "   <ServerCommunicationOSC name='oscSender' job='sender' port='6000'  refreshRate='1000'/> \n"
                  "   <CommunicationSubscriber name='sub1' communication='@oscSender' subject='/test' target='@oscSender' datas='x'/>"
                  "</Node>                                                                      \n";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
        root->init(ExecParams::defaultInstance()) ;

        std::future<void> future = std::async(std::launch::async, [](){
            OscDumpPacketListener listener;
            UdpListeningReceiveSocket s(
                        IpEndpointName( IpEndpointName::ANY_ADDRESS, 6000 ),
                        &listener );
            listener.setSocket(&s);
            s.Run();
            EXPECT_EQ(listener.m_data.size(), 1);
        });

        for( int i = 0; i < 10; i++ )
            sofa::simulation::getSimulation()->animate(root.get(),0.01);

        std::future_status status;
        status = future.wait_for(std::chrono::seconds(3));
        EXPECT_EQ(status, std::future_status::ready);
    }

    void checkReceiveOSC()
    {


        std::stringstream scene1 ;
        scene1 <<
                  "<?xml version='1.0' ?>                                                       \n"
                  "<Node name='root'>                                                           \n"
                  "   <DefaultAnimationLoop/>                                                   \n"
                  "   <RequiredPlugin name='Communication' />                                   \n"
                  "   <ServerCommunicationOSC name='oscReceiver' job='receiver' port='6000' /> \n"
                  "   <CommunicationSubscriber name='sub1' communication='@oscReceiver' subject='/test' target='@oscReceiver' datas='x'/>"
                  "</Node>                                                                      \n";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
        root->init(ExecParams::defaultInstance());
        ServerCommunication* aServerCommunicationOSC = dynamic_cast<ServerCommunication*>(root->getObject("oscReceiver"));
        aServerCommunicationOSC->setRunning(false);
        UdpTransmitSocket transmitSocket( IpEndpointName( "127.0.0.1", 6000 ) );
        sofa::simulation::getSimulation()->animate(root.get(), 0.01);
        for (int i = 0; i < 100; i++)
        {
            char buffer[1024];
            osc::OutboundPacketStream p(buffer, 1024 );
            p << osc::BeginBundleImmediate;
            p << osc::BeginMessage("/test");
            p << "";
            p << osc::EndMessage;
            p << osc::EndBundle;
            transmitSocket.Send( p.Data(), p.Size() );
        }

        for(unsigned int i=0; i<10; i++)
            sofa::simulation::getSimulation()->animate(root.get(), 0.01);

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
                  "   <DefaultAnimationLoop/>                                                   \n"
                  "   <RequiredPlugin name='Communication' />                                   \n"
                  "   <ServerCommunicationOSC name='oscSender' job='sender' port='6000'  refreshRate='1000'/> \n"
                  "   <CommunicationSubscriber name='subSender' communication='@oscSender' subject='/test' target='@oscSender' datas='port'/>"
                  "   <ServerCommunicationOSC name='oscReceiver' job='receiver' port='6000' /> \n"
                  "   <CommunicationSubscriber name='subReceiver' communication='@oscReceiver' subject='/test' target='@oscReceiver' datas='x'/>"
                  "</Node>                                                                      \n";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
        root->init(ExecParams::defaultInstance()) ;

        ServerCommunication* aServerCommunicationOSCSender = dynamic_cast<ServerCommunication*>(root->getObject("oscSender"));
        ServerCommunication* aServerCommunicationOSCReceiver = dynamic_cast<ServerCommunication*>(root->getObject("oscReceiver"));
        EXPECT_NE(aServerCommunicationOSCSender, nullptr);
        EXPECT_NE(aServerCommunicationOSCReceiver, nullptr);

        for( int i = 0; i < 100; i++ )
            sofa::simulation::getSimulation()->animate(root.get(),0.01);

        aServerCommunicationOSCReceiver->setRunning(false);

        Base::MapData dataMap = aServerCommunicationOSCReceiver->getDataAliases();
        Base::MapData::const_iterator itData;
        BaseData* data;

        itData = dataMap.find("x");
        EXPECT_TRUE(itData != dataMap.end());
        if (itData != dataMap.end())
        {
            data = itData->second;
            EXPECT_NE(data, nullptr) ;
            EXPECT_STRCASEEQ(data->getValueString().c_str(), "6000") ;
        }
    }

};

TEST_F(Communication_testOSC, checkAddSubscriber) {
    ASSERT_NO_THROW(this->checkAddSubscriber()) ;
}

TEST_F(Communication_testOSC, checkGetSubscriber) {
    ASSERT_NO_THROW(this->checkGetSubscriber()) ;
}

TEST_F(Communication_testOSC, checkArgumentCreation) {
    ASSERT_NO_THROW(this->checkArgumentCreation()) ;
}

TEST_F(Communication_testOSC, checkCreationDestruction) {
    ASSERT_NO_THROW(this->checkCreationDestruction()) ;
}

TEST_F(Communication_testOSC, checkSendOSC) {
    ASSERT_NO_THROW(this->checkSendOSC()) ;
}

TEST_F(Communication_testOSC, checkReceiveOSC) {
    ASSERT_NO_THROW(this->checkReceiveOSC()) ;
}

TEST_F(Communication_testOSC, checkSendReceiveOSC) {
    ASSERT_NO_THROW(this->checkSendReceiveOSC()) ;
}

} // communication
} // component
} // sofa
