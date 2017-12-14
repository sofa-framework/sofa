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
    //    pthread_mutex_init(&mutex, NULL);
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

bool ServerCommunication::writeData(SingleLink<CommunicationSubscriber, BaseObject, BaseLink::FLAG_DOUBLELINK> source, CommunicationSubscriber * subscriber, std::string subject, std::vector<std::string> argumentList)
{
    int i = 0;
    if (!isSubscribedTo(subject, argumentList.size()))
        return false;
    pthread_mutex_lock(&mutex);
    for (std::vector<std::string>::iterator it = argumentList.begin(); it != argumentList.end(); it++)
    {
        BaseData* data = fetchData(source, getArgumentType(*it), subscriber->getArgumentName(i));
        if (!data)
            continue;
        data->read(getArgumentValue(*it));
        i++;
    }
    pthread_mutex_unlock(&mutex);
    return true;
}

bool ServerCommunication::writeDataToFullMatrix(SingleLink<CommunicationSubscriber, BaseObject, BaseLink::FLAG_DOUBLELINK> source, CommunicationSubscriber * subscriber, std::string subject, std::vector<std::string> argumentList, int rows, int cols)
{
    std::string type = std::string("matrix") + getArgumentType(argumentList.at(0));
    BaseData* data = fetchData(source, type, subscriber->getArgumentName(0));
    std::string dataType = data->getValueTypeString();

    if(dataType.compare("FullMatrix<double>") == 0|| dataType.compare("FullMatrix<float>") == 0)
    {
        pthread_mutex_lock(&mutex);
        void* a = data->beginEditVoidPtr();
        FullMatrix<SReal> * b = static_cast<FullMatrix<SReal>*>(a);
        std::vector<std::string>::iterator it = argumentList.begin();
        b->resize(rows, cols);
        for(int i = 0; i < b->rows(); i++)
        {
            for(int j = 0; j < b->cols(); j++)
            {
                b->set(i, j, stod(getArgumentValue(*it)));
                ++it;
            }
        }
        pthread_mutex_unlock(&mutex);
        return true;
    }
    return false;
}

bool ServerCommunication::writeDataToContainer(SingleLink<CommunicationSubscriber, BaseObject, BaseLink::FLAG_DOUBLELINK> source, CommunicationSubscriber * subscriber, std::string subject, std::vector<std::string> argumentList)
{
    std::string type = std::string("matrix") + getArgumentType(argumentList.at(0));
    BaseData* data = fetchData(source, type, subscriber->getArgumentName(0));
    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();

    if (!typeinfo->Container())
        return false;
    std::string value = "";
    for (std::vector<std::string>::iterator it = argumentList.begin(); it != argumentList.end(); it++)
    {
        value.append(" " + getArgumentValue(*it));
    }
    pthread_mutex_lock(&mutex);
    data->read(value);
    pthread_mutex_unlock(&mutex);
    return true;
}

} /// communication

} /// component

} /// sofa
