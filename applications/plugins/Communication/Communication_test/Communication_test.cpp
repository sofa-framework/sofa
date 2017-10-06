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

#include <sofa/simulation/Node.h>
using sofa::simulation::Node ;
using sofa::core::ExecParams;
using sofa::core::objectmodel::BaseData ;
using sofa::core::objectmodel::BaseObject;

#include <sofa/helper/vectorData.h>
using sofa::helper::vectorData;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

#include <sofa/simulation/Simulation.h>
using sofa::simulation::Simulation ;

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory ;

#include <condition_variable>

// COMMUNICATION PART
#include <Communication/components/serverCommunication.h>
using sofa::component::communication::ServerCommunication;
#include <Communication/components/CommunicationSubscriber.h>
using sofa::component::communication::CommunicationSubscriber;

// OSC TEST PART
#include <oscpack/osc/OscReceivedElements.h>
#include <oscpack/osc/OscPrintReceivedElements.h>
#include <oscpack/ip/UdpSocket.h>
#include <oscpack/ip/PacketListener.h>
#include <oscpack/osc/OscReceivedElements.h>
#include <oscpack/osc/OscPrintReceivedElements.h>
#include <oscpack/osc/OscPacketListener.h>
#include <oscpack/osc/OscOutboundPacketStream.h>

#include <oscpack/ip/UdpSocket.h>

using sofa::defaulttype::Vec3f;


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
        d_positionsOut(this, "positionOut", "")
      , d_positionsIn(this, "positionIn", "")
    {
        f_listening = true ;
    }

    virtual void init() override
    {
        d_positionsOut.resize(1);
        d_positionsIn.resize(1);

        for(Data<Vec3f>* t : d_positionsIn)
        {
            Vec3f a;
            a.at(0) = 1.0f;
            a.at(1) = 1.0f;
            a.at(2) = 1.0f;
            t->setValue(a);
        }

    }

    virtual void handleEvent(sofa::core::objectmodel::Event *event) override
    {
    }

    vectorData<Vec3f>  d_positionsOut ;
    vectorData<Vec3f> d_positionsIn ;
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

    //    void server(){
    //        //        std::stringstream scene1 ;
    //        //        scene1 <<
    //        //                  "<?xml version='1.0' ?>                                                       \n"
    //        //                  "<Node name='root'>                                                           \n"
    //        //                  "   <RequiredPlugin name='Communication' />                                   \n"
    //        //                  "   <ServerCommunicationOSC name='oscSender' job='sender' port='6000'  refreshRate='1000'/> \n"
    //        //                  "   <CommunicationSubscriber name='sub1' communication='@oscSender' subject='/test' source='@oscSender' arguments='port'/>"
    //        //                  "</Node>                                                                      \n";

    //        //        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
    //        //        root->init(ExecParams::defaultInstance()) ;
    //        root->doSimu();
    //    }


    void checkSendOSC()
    {
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
        OscDumpPacketListener listener;
        UdpListeningReceiveSocket s(
                    IpEndpointName( IpEndpointName::ANY_ADDRESS, 6000 ),
                    &listener );
        listener.setSocket(&s);
        s.Run();
        EXPECT_EQ(listener.m_data.size(), 1);
    }


    //    void checkSendTODOOSC()
    //    {
    //        port ===,

    //        std::thread(server);
    //        std::thread(client);

    //        std::join(server);
    //        std::join(client);

    //        //        bool received = false;

    //        //        std::condition_variable cv;
    //        //        std::mutex mtx;


    //        //        std::stringstream scene1 ;
    //        //        scene1 <<
    //        //                  "<?xml version='1.0' ?>                                                       \n"
    //        //                  "<Node name='root'>                                                           \n"
    //        //                  "   <RequiredPlugin name='Communication' />                                   \n"
    //        //                  "   <ServerCommunicationOSC name='oscSender' job='sender' port='6000'  refreshRate='1000'/> \n"
    //        //                  "   <CommunicationSubscriber name='sub1' communication='@oscSender' subject='/test' source='@oscSender' arguments='port'/>"
    //        //                  "</Node>                                                                      \n";

    //        //        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
    //        //        root->init(ExecParams::defaultInstance()) ;

    //        //        std::unique_lock<std::mutex> lk(mtx);
    //        //        cv.wait(mtx, listener.();{ return; });
    //    }

//    void checkSendOSC(int numstep)
//    {
//        std::stringstream scene1 ;
//        scene1 <<
//                  "<?xml version='1.0' ?>                                                                                                               \n"
//                  "<Node name='root'>                                                                                                                   \n"
//                  "     <RequiredPlugin name='Communication' />                                                                                         \n"
//                  "     <MyComponent name='aName' positionIn='1.0 1.0 1.0' positionOut='2.0 2.0 2.0'/>                                                                                                     \n"
//                  "     <ServerCommunicationOSC name='oscReceiver' job='receiver' port='6000'/>                                                         \n"
//                  "     <CommunicationSubscriber name='sub1' communication='@oscReceiver' subject='/test' source='@oscReceiver' arguments='positionOut'/>      \n"
//                  "     <ServerCommunicationOSC name='oscSender' job='sender' port='6000' refreshRate='1000'/>                                          \n"
//                  "     <CommunicationSubscriber name='sub2' communication='@oscSender' subject='/test' source='@oscSender' arguments='positionIn'/>          \n"

//                  "</Node>                                                                      \n";

//        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
//        root->init(ExecParams::defaultInstance()) ;

//        for(unsigned int i=0;i<numstep;i++)
//            sofa::simulation::getSimulation()->animate(root.get(), 0.001);

//        MyComponent* component = dynamic_cast<MyComponent*>(root->getObject("aName"));
//        vectorData<Vec3f>  d_positionsOut = component->d_positionsOut;

//        for(unsigned int i=0; i<d_positionsOut.size(); i++)
//        {
//            ReadAccessor<Data<Vec3f>> data = d_positionsOut[i];
//            std::cout << data << std::endl;
//        }
//        vectorData<Vec3f>  d_positionsIn = component->d_positionsIn;

//        for(unsigned int i=0; i<d_positionsIn.size(); i++)
//        {
//            ReadAccessor<Data<Vec3f>> data = d_positionsIn[i];
//            std::cout << data << std::endl;
//        }


//    }

};

TEST_F(Communication_test, checkPerformancs) {
    ASSERT_NO_THROW(this->checkCreationDestruction()) ;
    ASSERT_NO_THROW(this->checkAddSubscriber()) ;
    ASSERT_NO_THROW(this->checkGetSubscriber()) ;
    ASSERT_NO_THROW(this->checkSendOSC()) ;
    ASSERT_NO_THROW(this->checkReceiveOSC()) ;
}

} // communication
} // component
} // sofa
