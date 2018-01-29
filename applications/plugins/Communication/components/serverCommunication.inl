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
#include <Communication/components/serverCommunication.h>
#include <Communication/components/CommunicationSubscriber.h>

using sofa::core::RegisterObject ;

namespace sofa
{

namespace component
{

namespace communication
{

ServerCommunication::ServerCommunication()
    : d_job(initData(&d_job, OptionsGroup(2,"receiver","sender"), "job", "If unspecified, the default value is receiver"))
    , d_address(initData(&d_address, (std::string)"127.0.0.1", "address", "Scale for object display. (default=localhost)"))
    , d_port(initData(&d_port, (int)(6000), "port", "Port to listen (default=6000)"))
    , d_refreshRate(initData(&d_refreshRate, (double)(30.0), "refreshRate", "Refres rate aka frequency (default=30), only used by sender"))
{
    pthread_mutex_init(&mutex, NULL);
}

ServerCommunication::~ServerCommunication()
{
    closeCommunication();
}

void ServerCommunication::init()
{
    f_listening = true;
    initTypeFactory();
    pthread_create(&m_thread, NULL, &ServerCommunication::thread_launcher, this);
}

void ServerCommunication::handleEvent(Event * event)
{
    if (AnimateBeginEvent::checkEventType(event))
    {
        BufferData* data = fetchArgumentsFromBuffer();
        if (data->source == NULL) // simply check if the data is not null
        {
            msg_error() << "something went wrong with received datas, fetched datas from buffer is null";
            return;
        }
        if(data->rows == -1 && data->cols == -1)
        {
            writeData(data);
            return;
        }

        if (!writeDataToFullMatrix(data))
            if (!writeDataToContainer(data))
                msg_error() << "something went wrong while converting network data into sofa matrix";
    }
}

bool ServerCommunication::saveArgumentsToBuffer(
        SingleLink<CommunicationSubscriber, BaseObject, BaseLink::FLAG_DOUBLELINK> source,
        CommunicationSubscriber * subscriber,
        std::string subject,
        ArgumentList argumentList,
        int rows ,
        int cols )
{
    try
    {
        receiveDataBuffer->add(source, subscriber, subject, argumentList, rows, cols);
    } catch (std::exception &exception) {
        msg_info("ServerCommunication") << exception.what();
        return false;
    }
    return true;
}

BufferData* ServerCommunication::fetchArgumentsFromBuffer()
{
    try
    {
        return receiveDataBuffer->get();
    } catch (std::exception &exception) {
        msg_info("ServerCommunication") << exception.what();
        return new BufferData();
    }
    return new BufferData();
}

void ServerCommunication::openCommunication()
{
    if (d_job.getValueString().compare("receiver") == 0)
    {
        receiveData();
    }
    else if (d_job.getValueString().compare("sender") == 0)
    {
        sendData();
    }
}

void ServerCommunication::closeCommunication()
{
    m_running = false;
    pthread_join(m_thread, NULL);
}

void * ServerCommunication::thread_launcher(void *voidArgs)
{
    ServerCommunication *args = (ServerCommunication*)voidArgs;
    args->openCommunication();
    return NULL;
}

bool ServerCommunication::isSubscribedTo(std::string subject, unsigned int argumentSize)
{
    try
    {
        CommunicationSubscriber* subscriber = m_subscriberMap.at(subject);
        if (subscriber->getArgumentSize() == argumentSize)
            return true;
        else
        {
            msg_warning(this->getName()) << " is subscrided to " << subject << " but arguments should be size of " << subscriber->getArgumentSize() << ", received " << argumentSize << '\n';
        }
    } catch (const std::out_of_range& oor) {
        msg_warning(this->getName()) << " is not subscrided to " << subject << '\n';
    }
    return false;
}

CommunicationSubscriber * ServerCommunication::getSubscriberFor(std::string subject)
{
    try
    {
        return m_subscriberMap.at(subject);
    } catch (const std::out_of_range& oor) {
        msg_warning(this->getClassName()) << " is not subscrided to " << subject << '\n';
    }
    return nullptr;
}

void ServerCommunication::addSubscriber(CommunicationSubscriber * subscriber)
{
    m_subscriberMap.insert(std::pair<std::string, CommunicationSubscriber*>(subscriber->getSubject(), subscriber));
}

std::map<std::string, CommunicationSubscriber*> ServerCommunication::getSubscribers()
{
    return m_subscriberMap;
}

BaseData* ServerCommunication::fetchData(SingleLink<CommunicationSubscriber, BaseObject, BaseLink::FLAG_DOUBLELINK> source, std::string keyTypeMessage, std::string argumentName)
{
    MapData dataMap = source->getDataAliases();
    MapData::const_iterator itData = dataMap.find(argumentName);
    BaseData* data;

    if (itData == dataMap.end())
    {
        data = getFactoryInstance()->createObject(keyTypeMessage, sofa::helper::NoArgument());
        if (data == nullptr)
            msg_warning() << keyTypeMessage << " is not a known type";
        else
        {
            data->setName(argumentName);
            data->setHelp("Auto generated help from communication");
            source->addData(data, argumentName);
            msg_info(source->getName()) << " data field named : " << argumentName << " of type " << keyTypeMessage << " has been created";
        }
    } else
        data = itData->second;
    return data;
}

bool ServerCommunication::writeData(BufferData* data)
{
    int i = 0;
    if (!isSubscribedTo(data->subject, data->argumentList.size()))
        return false;
    for (std::vector<std::string>::iterator it = data->argumentList.begin(); it != data->argumentList.end(); it++)
    {
        BaseData* baseData = fetchData(data->source, getArgumentType(*it), data->subscriber->getArgumentName(i));
        if (!baseData)
            continue;
        baseData->read(getArgumentValue(*it));
        i++;
    }
    return true;
}

bool ServerCommunication::writeDataToFullMatrix(BufferData* data)
{
    //    std::string type = std::string("matrix") + getArgumentType(data.argumentList.at(0));
    //    BaseData* baseData = fetchData(data.source, type, data.subscriber->getArgumentName(0));
    //    std::string dataType = baseData->getValueTypeString();

    //    if(dataType.compare("FullMatrix<double>") == 0|| dataType.compare("FullMatrix<float>") == 0)
    //    {
    //        void* a = baseData->beginEditVoidPtr();
    //        FullMatrix<SReal> * b = static_cast<FullMatrix<SReal>*>(a);
    //        std::vector<std::string>::iterator it = data.argumentList.begin();
    //        b->resize(data.rows, data.cols);
    //        for(int i = 0; i < b->rows(); i++)
    //        {
    //            for(int j = 0; j < b->cols(); j++)
    //            {
    //                b->set(i, j, stod(getArgumentValue(*it)));
    //                ++it;
    //            }
    //        }
    //        return true;
    //    }
    //    return false;
}

bool ServerCommunication::writeDataToContainer(BufferData* data)
{
    //    std::string type = std::string("matrix") + getArgumentType(data.argumentList.at(0));
    //    BaseData* baseData = fetchData(data.source, type, data.subscriber->getArgumentName(0));
    //    const AbstractTypeInfo *typeinfo = baseData->getValueTypeInfo();

    //    if (!typeinfo->Container())
    //        return false;
    //    std::string value = "";
    //    for (std::vector<std::string>::iterator it = data.argumentList.begin(); it != data.argumentList.end(); it++)
    //    {
    //        value.append(" " + getArgumentValue(*it));
    //    }
    //    baseData->read(value);
    //    return true;
}

} /// communication

} /// component

} /// sofa
