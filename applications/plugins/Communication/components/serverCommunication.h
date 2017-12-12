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

#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;

#include <sofa/core/objectmodel/Data.h>
using sofa::core::objectmodel::Data;
using sofa::core::objectmodel::BaseData;

#include <sofa/helper/vectorData.h>
using sofa::helper::vectorData;
using sofa::helper::WriteAccessorVector;
using sofa::helper::WriteAccessor;
using sofa::helper::ReadAccessor;

#include <sofa/helper/OptionsGroup.h>
using sofa::helper::OptionsGroup;

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
using sofa::core::objectmodel::Event;

#include <sofa/defaulttype/DataTypeInfo.h>
using sofa::defaulttype::AbstractTypeInfo ;

using sofa::helper::vector;

#include <SofaBaseLinearSolver/FullMatrix.h>
using sofa::component::linearsolver::FullMatrix;

#include <sofa/helper/Factory.h>
#include <sofa/helper/Factory.inl>
using sofa::helper::Factory;

#include <pthread.h>
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

template<typename DataType>
class DataCreator : public sofa::helper::BaseCreator<BaseData>
{
public:
    virtual BaseData* createInstance(sofa::helper::NoArgument) override { return new sofa::core::objectmodel::Data<DataType>(); }
    virtual const std::type_info& type() override { return typeid(BaseData);}
};

//forward declaration
class CommunicationSubscriber;

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

    std::map<std::string, CommunicationSubscriber*> m_subscriberMap;
    pthread_mutex_t                                 mutex;
    pthread_t                                       m_thread;
    bool                                            m_running = true;

    virtual void openCommunication();
    virtual void closeCommunication();
    static void* thread_launcher(void*);
    virtual void sendData() =0;
    virtual void receiveData() =0;

    BaseData* fetchData(SingleLink<CommunicationSubscriber,  BaseObject, BaseLink::FLAG_DOUBLELINK> source, std::string keyTypeMessage, std::string argumentName);

    bool writeData(SingleLink<CommunicationSubscriber, BaseObject, BaseLink::FLAG_DOUBLELINK> source, CommunicationSubscriber * subscriber, std::string subject, std::vector<std::string> argumentList);
    bool writeDataToContainer(SingleLink<CommunicationSubscriber, BaseObject, BaseLink::FLAG_DOUBLELINK> source, CommunicationSubscriber * subscriber, std::string subject, std::vector<std::string> argumentList);
    bool writeDataToFullMatrix(SingleLink<CommunicationSubscriber, BaseObject, BaseLink::FLAG_DOUBLELINK> source, CommunicationSubscriber * subscriber, std::string subject, std::vector<std::string> argumentList, int rows, int cols);

    virtual std::string getArgumentType(std::string argument) =0;
    virtual std::string getArgumentValue(std::string argument) =0;

};

} /// namespace communication
} /// namespace component
} /// namespace sofa

#endif // SOFA_SERVERCOMMUNICATION_H
