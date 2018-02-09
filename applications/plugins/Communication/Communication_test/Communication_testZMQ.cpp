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
#include <SofaSimulationGraph/init.h>
#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::simulation::Simulation ;

#include <sofa/core/ObjectFactory.h>
using sofa::core::ObjectFactory ;

// COMMUNICATION PART
#include <Communication/components/serverCommunication.h>
#include <Communication/components/serverCommunicationZMQ.h>
#include <Communication/components/CommunicationSubscriber.h>
using sofa::component::communication::ServerCommunication;
using sofa::component::communication::ServerCommunicationZMQ;
using sofa::component::communication::CommunicationSubscriber;

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

class MyComponentZMQ : public BaseObject
{
public:
    MyComponentZMQ() :
        d_vectorIn(initData (&d_vectorIn, "vectorIn", ""))
      , d_vectorOut(initData (&d_vectorOut, "vectorOut", ""))
    {
        f_listening = true ;
    }

    virtual void init() override
    {
        void* a = d_vectorIn.beginEditVoidPtr();
        FullMatrix<SReal> * b = static_cast<FullMatrix<SReal>*>(a);
        b->resize(10, 10);
        for(int i = 0; i < b->rows(); i++)
            for(int j = 0; j < b->cols(); j++)
                b->set(i, j, 1.0);

        a = d_vectorOut.beginEditVoidPtr();
        b = static_cast<FullMatrix<SReal>*>(a);
        b->resize(10, 10);
        for(int i = 0; i < b->rows(); i++)
            for(int j = 0; j < b->cols(); j++)
                b->set(i, j, 2.0);

    }

    virtual void handleEvent(sofa::core::objectmodel::Event *event) override
    {
        std::cout << "Test thread safe" << std::endl;
        void* voidInput = d_vectorIn.beginEditVoidPtr();
        FullMatrix<SReal> * input = static_cast<FullMatrix<SReal>*>(voidInput);
        void* voidOutput = d_vectorOut.beginEditVoidPtr();
        FullMatrix<SReal> * output = static_cast<FullMatrix<SReal>*>(voidOutput);

        EXPECT_EQ(input->rows(), output->rows());
        EXPECT_EQ(input->cols(), output->cols());


        // for the next step we increase the value of the ouput
        SReal firstValue = input->element(0,0);
        for(int i = 0; i < input->rows(); i++)
        {
            for(int j = 0; j < input->cols(); j++)
            {
                if(input->element(i,j) != firstValue)
                {
                    EXPECT_EQ(input->element(i,j), firstValue);
                    std::cout << "Input: " << d_vectorIn.getValueString() << "\nOutput: " << d_vectorOut.getValueString() << std::endl;
                    break;
                }
            }
        }



        // for the next step we increase the value of the ouput
        for(int i = 0; i < output->rows(); i++)
        {
            for(int j = 0; j < output->cols(); j++)
            {
                output->set(i, j, output->element(i,j)+1.0);
            }
        }
    }


    FullMatrix<SReal>* getInput()
    {
        void* a = d_vectorIn.beginEditVoidPtr();
        FullMatrix<SReal> * b = static_cast<FullMatrix<SReal>*>(a);
        return b;
    }

    FullMatrix<SReal>* getOutput()
    {
        void* a = d_vectorOut.beginEditVoidPtr();
        FullMatrix<SReal> * b = static_cast<FullMatrix<SReal>*>(a);
        return b;
    }

    //    vectorData<Vec3f> d_pos;
    Data<FullMatrix<SReal>> d_vectorIn;
    Data<FullMatrix<SReal>> d_vectorOut;
} ;

int mZMQclass = sofa::core::RegisterObject("").add<MyComponentZMQ>();

class Communication_testZMQ : public Sofa_test<>
{

public:

    /// BASIC STUFF TEST
    /// testing subscriber + argument creation ZMQ

    void checkAddSubscriber()
    {
        std::stringstream scene1 ;
        scene1 <<
                  "<?xml version='1.0' ?>                                                       \n"
                  "<Node name='root'>                                                           \n"
                  "   <RequiredPlugin name='Communication' />                                   \n"
                  "   <ServerCommunicationZMQ name='zmqSender' job='sender' port='6000'  refreshRate='1000'/> \n"
                  "   <CommunicationSubscriber name='sub1' communication='@zmqSender' subject='/test' source='@zmqSender' arguments='x'/>"
                  "</Node>                                                                      \n";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
        root->init(ExecParams::defaultInstance()) ;

        ServerCommunication* aServerCommunicationZMQ = dynamic_cast<ServerCommunication*>(root->getObject("zmqSender"));
        std::map<std::string, CommunicationSubscriber*> map = aServerCommunicationZMQ->getSubscribers();
        EXPECT_EQ(map.size(), 1);
    }

    void checkGetSubscriber()
    {
        std::stringstream scene1 ;
        scene1 <<
                  "<?xml version='1.0' ?>                                                       \n"
                  "<Node name='root'>                                                           \n"
                  "   <RequiredPlugin name='Communication' />                                   \n"
                  "   <ServerCommunicationZMQ name='zmqSender' job='sender' port='6000'  refreshRate='1000'/> \n"
                  "   <CommunicationSubscriber name='sub1' communication='@zmqSender' subject='/test' source='@zmqSender' arguments='x'/>"
                  "</Node>                                                                      \n";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
        root->init(ExecParams::defaultInstance()) ;

        ServerCommunication* aServerCommunicationZMQ = dynamic_cast<ServerCommunication*>(root->getObject("zmqSender"));
        CommunicationSubscriber* subscriber = aServerCommunicationZMQ->getSubscriberFor("/test");
        EXPECT_NE(subscriber, nullptr) ;
    }

    void checkArgumentCreation()
    {
        std::stringstream scene1 ;
        scene1 <<
                  "<?xml version='1.0' ?>                                                       \n"
                  "<Node name='root'>                                                           \n"
                  "   <RequiredPlugin name='Communication' />                                   \n"
                  "   <ServerCommunicationZMQ name='zmqSender' job='sender' port='6000' refreshRate='1000'/> \n"
                  "   <CommunicationSubscriber name='sub1' communication='@zmqSender' subject='/test' source='@zmqSender' arguments='x'/>"
                  "</Node>                                                                      \n";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
        root->init(ExecParams::defaultInstance());

        std::cout << "a" << std::endl;
        sofa::simulation::graph::getSimulation()->init(root.get());
        root->setAnimate(true);
        std::cout << "a" << std::endl;


        ServerCommunication* aServerCommunicationZMQ = dynamic_cast<ServerCommunication*>(root->getObject("zmqSender"));
        EXPECT_NE(aServerCommunicationZMQ, nullptr);

        usleep(1000000);

        Base::MapData dataMap = aServerCommunicationZMQ->getDataAliases();
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
                  "   <RequiredPlugin name='Communication' />                                   \n"
                  "   <ServerCommunicationZMQ name='zmqSender' job='sender' port='6000'  refreshRate='1000'/> \n"
                  "</Node>                                                                      \n";

        Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene", scene1.str().c_str(), scene1.str().size()) ;
        root->init(ExecParams::defaultInstance()) ;
    }

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
                  "<Node  name='root' gravity='0 0 0' time='0' animate='0'   >                  \n"
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
        MyComponentZMQ* aComponent = dynamic_cast<MyComponentZMQ*>(root->getObject("aComponent"));

        EXPECT_NE(aServerCommunicationSender, nullptr);
        EXPECT_NE(aServerCommunicationReceiver, nullptr);
        EXPECT_NE(aComponent, nullptr);


        for(unsigned int i=0; i<100; i++)
        {
            usleep(1000);
            aComponent->handleEvent(NULL);

        }

        aServerCommunicationReceiver->setRunning(false);
        std::cout << "Wait for end of communication" << std::endl;
        usleep(10000); // wait 5 seconds for the changes
        aServerCommunicationSender->setRunning(false);

    }
};

TEST_F(Communication_testZMQ, checkAddSubscriber) {
    ASSERT_NO_THROW(this->checkAddSubscriber()) ;
}

TEST_F(Communication_testZMQ, checkGetSubscriber) {
    ASSERT_NO_THROW(this->checkGetSubscriber()) ;
}

TEST_F(Communication_testZMQ, checkArgumentCreation) {
    ASSERT_NO_THROW(this->checkArgumentCreation()) ;
}

TEST_F(Communication_testZMQ, checkCreationDestruction) {
    ASSERT_NO_THROW(this->checkCreationDestruction()) ;
}

TEST_F(Communication_testZMQ, checkSendZMQ) {
    ASSERT_NO_THROW(this->checkSendZMQ()) ;
}

TEST_F(Communication_testZMQ, checkReceiveZMQ) {
    ASSERT_NO_THROW(this->checkReceiveZMQ()) ;
}

TEST_F(Communication_testZMQ, checkSendReceiveZMQ) {
    ASSERT_NO_THROW(this->checkSendReceiveZMQ()) ;
}

TEST_F(Communication_testZMQ, checkZMQParsingFunctions) {
    ASSERT_NO_THROW(this->checkZMQParsingFunctions()) ;
}

TEST_F(Communication_testZMQ, checkThreadSafeZMQ) {
    ASSERT_NO_THROW(this->checkThreadSafeZMQ()) ;
}

} // communication
} // component
} // sofa
