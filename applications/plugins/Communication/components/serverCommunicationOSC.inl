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

    getFactoryInstance()->registerCreator("matrixf", new DataCreator<vector<float>>());
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

    for (std::map<std::string, CommunicationSubscriber*>::iterator it = subscribersMap.begin(); it != subscribersMap.end(); it++)
    {
        CommunicationSubscriber* subscriber = it->second;
        SingleLink<CommunicationSubscriber,  BaseObject, BaseLink::FLAG_DOUBLELINK> source = subscriber->getSource();

        std::vector<std::string> argumentList = subscriber->getArgumentList();
        for (std::vector<std::string>::iterator itArgument = argumentList.begin(); itArgument != argumentList.end(); itArgument++ )
        {

            BaseData* data = fetchData(source, "s", *itArgument); // s for std::string in case of non existing argument
            if (!data)
            {
                messageName = subscriber->getSubject();
                p << osc::BeginMessage(messageName.c_str());
                p << osc::EndMessage;
            }

            const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
            const void* valueVoidPtr = data->getValueVoidPtr();
            messageName = subscriber->getSubject();

            p << osc::BeginMessage(messageName.c_str());

            if (typeinfo->Container())
            {
                int nbRows = typeinfo->size();
                int nbCols  = typeinfo->size(data->getValueVoidPtr()) / typeinfo->size();
                p  << nbRows << nbCols;

                if( !typeinfo->Text() && !typeinfo->Scalar() && !typeinfo->Integer() )
                {
                    msg_advice(data->getOwner()) << "BaseData_getAttr_value unsupported native type="<<data->getValueTypeString()<<" for data "<<data->getName()<<" ; returning string value" ;
                    p <<  (data->getValueString().c_str());
                }
                else if (typeinfo->Text())
                    for (int i=0; i < nbRows; i++)
                        for (int j=0; j<nbCols; j++)
                            p << typeinfo->getTextValue(valueVoidPtr,(i*nbCols) + j).c_str();
                else if (typeinfo->Scalar())
                    for (int i=0; i < nbRows; i++)
                        for (int j=0; j<nbCols; j++)
                            p << typeinfo->getScalarValue(valueVoidPtr,(i*nbCols) + j);
                else if (typeinfo->Integer())
                    for (int i=0; i < nbRows; i++)
                        for (int j=0; j<nbCols; j++)
                            p << (int)typeinfo->getIntegerValue(valueVoidPtr,(i*nbCols) + j);
            }
            else
            {
                if (typeinfo->Text())
                    p << (typeinfo->getTextValue(valueVoidPtr,0).c_str());
                else if (typeinfo->Scalar())
                    p << (typeinfo->getScalarValue(valueVoidPtr,0));
                else if (typeinfo->Integer())
                    p << ((int)typeinfo->getIntegerValue(valueVoidPtr,0));
                else
                {
                    msg_advice(data->getOwner()) << "BaseData_getAttr_value unsupported native type="<<data->getValueTypeString()<<" for data "<<data->getName()<<" ; returning string value" ;
                    p <<  (data->getValueString().c_str());
                }
            }
            p << osc::EndMessage;
        }
    }
    return p;
}

void ServerCommunicationOSC::ProcessMessage( const osc::ReceivedMessage& m, const IpEndpointName& remoteEndpoint )
{
    if (!m_running)
        m_socket->Break();
    const char* address = m.AddressPattern();
    osc::ReceivedMessageArgumentIterator it = m.ArgumentsBegin();

    BaseData* data;
    CommunicationSubscriber * subscriber = getSubscriberFor(address);
    if (!subscriber)
        return;
    SingleLink<CommunicationSubscriber,  BaseObject, BaseLink::FLAG_DOUBLELINK> source = subscriber->getSource();

    std::string firstArg = convertArgumentToStringValue(it);
    if (firstArg.compare("matrix") == 0)
    {
        int row = 0, col = 0;
        if (m.ArgumentCount() >= 3)
        {
            try
            {
                row = (++it)->AsInt32();
                col = (++it)->AsInt32();
                if (row < 0 || col < 0)
                    return;
            } catch (osc::WrongArgumentTypeException e)
            {
                msg_error() << "row or column is not an int";
                return;
            }
        } else
            msg_warning() << address << " is matrix, but message size is not correct. Should be : /subject matrix width height value value value... ";

        data = fetchData(source, "matrix" + std::string(1, (++it)->TypeTag()), subscriber->getArgumentName(0));
        if (!data)
            return;

        std::stringstream stream;
        for ( it ; it != m.ArgumentsEnd(); it++)
            stream << convertArgumentToStringValue(it) << " ";
        data->read(stream.str());
    }
    else
    {
        if (!isSubscribedTo(m.AddressPattern(), m.ArgumentCount()))
            return;
        int i = 0;
        for ( it ; it != m.ArgumentsEnd(); it++)
        {
            data = fetchData(source, std::string(1, it->TypeTag()), subscriber->getArgumentName(i));
            if (!data)
                continue;
            data->read(convertArgumentToStringValue(it));
            i++;
        }
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
