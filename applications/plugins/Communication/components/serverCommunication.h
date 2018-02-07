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
#ifndef SOFA_SERVERCOMMUNICATION_H
#define SOFA_SERVERCOMMUNICATION_H

#include <Communication/config.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/objectmodel/BaseObject.h>
using sofa::helper::vector;
using sofa::core::objectmodel::Event;
using sofa::core::objectmodel::Data;
using sofa::core::objectmodel::BaseData;
using sofa::core::objectmodel::BaseObject ;

#include <sofa/helper/vectorData.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/Factory.inl>
#include <sofa/helper/OptionsGroup.h>
using sofa::helper::OptionsGroup;
using sofa::helper::Factory;
using sofa::helper::vectorData;
using sofa::helper::WriteAccessorVector;
using sofa::helper::WriteAccessor;
using sofa::helper::ReadAccessor;

#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
using sofa::simulation::AnimateBeginEvent;
using sofa::simulation::AnimateEndEvent;

#include <sofa/defaulttype/DataTypeInfo.h>
using sofa::defaulttype::AbstractTypeInfo ;

#include <SofaBaseLinearSolver/FullMatrix.h>
using sofa::component::linearsolver::FullMatrix;


#include <Communication/components/communicationCircularBuffer.h>

#include <iostream>
#include <chrono>
#include <thread>
#include <stdio.h>
#include <mutex>
#include <cmath>

namespace sofa
{

namespace component
{

namespace communication
{

//forward declaration
class CommunicationSubscriber;


template<typename DataType>
class DataCreator : public sofa::helper::BaseCreator<BaseData>
{
public:
    virtual BaseData* createInstance(sofa::helper::NoArgument) override { return new sofa::core::objectmodel::Data<DataType>(); }
    virtual const std::type_info& type() override { return typeid(BaseData);}
};

class SOFA_COMMUNICATION_API ServerCommunication : public BaseObject
{

public:

    typedef BaseObject Inherited;
    SOFA_ABSTRACT_CLASS(ServerCommunication, Inherited);

    ServerCommunication() ;
    virtual ~ServerCommunication() ;

    bool isSubscribedTo(std::string, unsigned int);
    void addSubscriber(CommunicationSubscriber*);
    std::map<std::string, CommunicationSubscriber*> getSubscribers();
    CommunicationSubscriber* getSubscriberFor(std::string);

    bool isRunning() { return m_running;}
    void setRunning(bool value) {m_running = value;}

    virtual std::string getArgumentType(std::string argument) =0;
    virtual std::string getArgumentValue(std::string argument) =0;

    //////////////////////////////// Factory type /////////////////////////////////
    typedef sofa::helper::Factory< std::string, BaseData> CommunicationDataFactory;
    virtual CommunicationDataFactory* getFactoryInstance() =0;
    virtual void initTypeFactory() =0;
    /////////////////////////////////////////////////////////////////////////////////

    ////////////////////////// Inherited from BaseObject ////////////////////
    virtual void init() override;
    virtual void handleEvent(Event *) override;
    /////////////////////////////////////////////////////////////////////////

    Data<helper::OptionsGroup>  d_job;
    Data<std::string>           d_address;
    Data<int>                   d_port;
    Data<double>                d_refreshRate;

protected:

    CircularBufferReceiver* receiveDataBuffer = new CircularBufferReceiver(3);
    std::map<std::string, CircularBufferSender*> senderDataMap;
    std::map<std::string, CommunicationSubscriber*> m_subscriberMap;
    std::thread                                     m_thread;
    bool                                            m_running = true;

    virtual void openCommunication();
    virtual void closeCommunication();
    static void* thread_launcher(void*);
    virtual void sendData() =0;
    virtual void receiveData() =0;
    virtual std::string defaultDataType() =0;

    ////////////////////////// Buffer ////////////////////
    bool saveDataToSenderBuffer();
    BaseData* fetchDataFromSenderBuffer(CommunicationSubscriber* subscriber, std::string argument);
    bool saveArgumentsToReceivedBuffer(std::string subject, ArgumentList argumentList, int rows, int cols);
    BufferData* fetchArgumentsFromReceivedBuffer();
    /////////////////////////////////////////////////////////////////////////

    BaseData* fetchData(SingleLink<CommunicationSubscriber,  BaseObject, BaseLink::FLAG_DOUBLELINK> source, std::string keyTypeMessage, std::string argumentName);
    bool writeData(BufferData* data);
    bool writeDataToContainer(BufferData* data);
    bool writeDataToFullMatrix(BufferData* data);

};

} /// namespace communication
} /// namespace component
} /// namespace sofa

#endif // SOFA_SERVERCOMMUNICATION_H
