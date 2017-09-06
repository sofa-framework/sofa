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
#include "serverCommunicationOSC.h"
#include <Communication/components/CommunicationSubscriber.h>

using sofa::core::RegisterObject ;

namespace sofa
{

namespace component
{

namespace communication
{

ServerCommunicationOSC::ServerCommunicationOSC()
    : Inherited(), osc::OscPacketListener()
    , d_packetSize(initData(&d_packetSize, (int)(1024), "packetSize", "OSC packet size (default=1024)"))
{
}

ServerCommunicationOSC::~ServerCommunicationOSC()
{
    m_socket->Break();
    free(m_socket);
    Inherited::closeCommunication();
}

ServerCommunicationOSC::OSCDataFactory* ServerCommunicationOSC::getFactoryInstance(){
    static OSCDataFactory* s_localfactory = nullptr ;
    if(s_localfactory==nullptr)
        s_localfactory = new ServerCommunicationOSC::OSCDataFactory() ;
    return s_localfactory ;
}

void ServerCommunicationOSC::initTypeFactory()
{
    getFactoryInstance()->registerCreator("f", new DataCreator<float>());
    getFactoryInstance()->registerCreator("d", new DataCreator<double>());
    getFactoryInstance()->registerCreator("i", new DataCreator<int>());
    getFactoryInstance()->registerCreator("s", new DataCreator<std::string>());
    // TODO have a look at blobs
    // TODO have a look at time tag
}

void ServerCommunicationOSC::sendData()
{
    UdpTransmitSocket transmitSocket( IpEndpointName(this->d_address.getValue().c_str(), this->d_port.getValue()));

    while (this->m_running)
    {
        std::map<std::string, CommunicationSubscriber*> subscribersMap = getSubscribers();
        if (subscribersMap.size() == 0)
            continue;
        osc::OutboundPacketStream p = createOSCMessage();
        transmitSocket.Send( p.Data(), p.Size() );
        std::this_thread::sleep_for(std::chrono::microseconds(int(1000000.0/(double)this->d_refreshRate.getValue())));
    }
}

void ServerCommunicationOSC::receiveData()
{
    m_socket = new UdpListeningReceiveSocket(IpEndpointName( IpEndpointName::ANY_ADDRESS, this->d_port.getValue()), this);
    m_socket->Run();
}

osc::OutboundPacketStream ServerCommunicationOSC::createOSCMessage()
{
    int bufferSize = d_packetSize.getValue();
    char buffer[bufferSize];
    osc::OutboundPacketStream p(buffer, bufferSize);
    std::map<std::string, CommunicationSubscriber*> subscribersMap = getSubscribers();
    std::string messageName;

    MapData dataMap = getDataAliases();
    for (std::map<std::string, CommunicationSubscriber*>::iterator it = subscribersMap.begin(); it != subscribersMap.end(); it++)
    {
        CommunicationSubscriber* subscriber = it->second;
        SingleLink<CommunicationSubscriber,  BaseObject, BaseLink::FLAG_DOUBLELINK> source = subscriber->getSource();

        std::vector<std::string> argumentList = subscriber->getArgumentList();
        for (std::vector<std::string>::iterator itArgument = argumentList.begin(); itArgument != argumentList.end(); itArgument++ )
        {
            MapData::const_iterator itData = source->getDataAliases().find(*itArgument);
            // handle no argument
            if (itData == dataMap.end())
            {
                messageName = subscriber->getSubject();
                p << osc::BeginMessage(messageName.c_str());
                p << osc::EndMessage;
            } else
            {
                BaseData* data = itData->second;
                const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
                const void* valueVoidPtr = data->getValueVoidPtr();

                messageName = subscriber->getSubject();

                p << osc::BeginMessage(messageName.c_str());

                if (typeinfo->Container())
                {
                    int rowWidth = typeinfo->size();
                    int nbRows = typeinfo->size(data->getValueVoidPtr()) / typeinfo->size();
                    p << rowWidth << nbRows;
                    /// this is a vector; return a python list of the corresponding type (ints, scalars or strings)
                    if( !typeinfo->Text() && !typeinfo->Scalar() && !typeinfo->Integer() )
                    {
                        msg_advice(data->getOwner()) << "BaseData_getAttr_value unsupported native type="<<data->getValueTypeString()<<" for data "<<data->getName()<<" ; returning string value" ;
                        p <<  (data->getValueString().c_str());
                    }

                    for (int i=0; i < nbRows; i++)
                    {
                        for (int j=0; j<rowWidth; j++)
                        {
                            if (typeinfo->Text())
                            {
                                p << (typeinfo->getTextValue(valueVoidPtr,0).c_str());
                            }
                            else if (typeinfo->Scalar())
                            {
                                p << (typeinfo->getScalarValue(valueVoidPtr,0));
                            }
                            else if (typeinfo->Integer())
                            {
                                p << ((int)typeinfo->getIntegerValue(valueVoidPtr,0));
                            }
                            else {
                                msg_advice(data->getOwner()) << "BaseData_getAttr_value unsupported native type="<<data->getValueTypeString()<<" for data "<<data->getName()<<" ; returning string value" ;
                                p <<  (data->getValueString().c_str());
                            }
                        }
                    }
                }
                else
                {
                    if (typeinfo->Text())
                    {
                        p << (typeinfo->getTextValue(valueVoidPtr,0).c_str());
                    }
                    else if (typeinfo->Scalar())
                    {
                        p << (typeinfo->getScalarValue(valueVoidPtr,0));
                    }
                    else if (typeinfo->Integer())
                    {
                        p << ((int)typeinfo->getIntegerValue(valueVoidPtr,0));
                    }
                    else {
                        msg_advice(data->getOwner()) << "BaseData_getAttr_value unsupported native type="<<data->getValueTypeString()<<" for data "<<data->getName()<<" ; returning string value" ;
                        p <<  (data->getValueString().c_str());
                    }
                }
                p << osc::EndMessage;
            }
        }
    }
    return p;
}

void ServerCommunicationOSC::ProcessMessage( const osc::ReceivedMessage& m, const IpEndpointName& remoteEndpoint )
{
    const char* address = m.AddressPattern();
    if (!isSubscribedTo(m.AddressPattern(), m.ArgumentCount()))
        return;

    CommunicationSubscriber * subscriber = getSubscriberFor(address);
    SingleLink<CommunicationSubscriber,  BaseObject, BaseLink::FLAG_DOUBLELINK> source = subscriber->getSource();

    int i = 0;
    for ( osc::ReceivedMessageArgumentIterator it = m.ArgumentsBegin()++; it != m.ArgumentsEnd(); it++)
    {
        std::string keyTypeMessage = std::string(1, it->TypeTag());
        std::string argumentName = subscriber->getArgumentName(i);
        MapData dataMap = source->getDataAliases();

        MapData::const_iterator itData = dataMap.find(argumentName);
        if (itData == dataMap.end())
        {
            BaseData* data = getFactoryInstance()->createObject(keyTypeMessage, sofa::helper::NoArgument());
            if (data == nullptr)
                msg_warning() << keyTypeMessage << " is not a known type";
            else
            {
                data->setName(argumentName);
                data->setHelp("Auto generated help from OSC communication");
                source->addData(data, argumentName);
                data->read(convertArgumentToStringValue(it));
                msg_info(source->getName()) << "data field named : " << argumentName << " has been created";
            }
        } else
        {
            BaseData* data = itData->second;
            data->read(convertArgumentToStringValue(it));
        }
        i++;
    }
}

std::string ServerCommunicationOSC::convertArgumentToStringValue(osc::ReceivedMessageArgumentIterator it)
{
    std::string stringData;
    try
    {
        stringData = it->AsString();
    } catch (osc::WrongArgumentTypeException e)
    {
        std::stringstream stream;
        stream << (*it);
        std::string s = stream.str();
        size_t pos = s.find(":"); // That's how OSC message works -> Type:Value
        s.erase(0, pos+1);
        stream.str(s);
        stringData = stream.str();
    }
    return stringData;
}


} /// communication

} /// component

} /// sofa
