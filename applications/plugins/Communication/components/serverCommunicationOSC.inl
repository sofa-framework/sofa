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
    if (d_job.getValueString().compare("receiver") == 0)
    {
        m_socket->Break();
        delete m_socket;
    }
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
    getFactoryInstance()->registerCreator("float32", new DataCreator<float>());
    getFactoryInstance()->registerCreator("double", new DataCreator<double>());
    getFactoryInstance()->registerCreator("int32", new DataCreator<int>());
    getFactoryInstance()->registerCreator("OSC-string", new DataCreator<std::string>());

    getFactoryInstance()->registerCreator("matrixfloat32", new DataCreator<FullMatrix<float>>());
    getFactoryInstance()->registerCreator("matrixdouble", new DataCreator<FullMatrix<double>>());
    getFactoryInstance()->registerCreator("matrixint32", new DataCreator<FullMatrix<int>>());
}

std::string ServerCommunicationOSC::defaultDataType()
{
    return "OSC-string";
}

/******************************************************************************
*                                                                             *
* SEND PART                                                                   *
*                                                                             *
******************************************************************************/

void ServerCommunicationOSC::sendData()
{
    UdpTransmitSocket transmitSocket( IpEndpointName(this->d_address.getValue().c_str(), this->d_port.getValue()));

    while (this->m_running)
    {
        std::map<std::string, CommunicationSubscriber*> subscribersMap = getSubscribers();
        if (subscribersMap.size() == 0)
            continue;

        for (std::map<std::string, CommunicationSubscriber*>::iterator it = subscribersMap.begin(); it != subscribersMap.end(); it++)
        {
            CommunicationSubscriber* subscriber = it->second;
            ArgumentList argumentList = subscriber->getArgumentList();
            int bufferSize = d_packetSize.getValue();
            char* buffer = new char[bufferSize];
            osc::OutboundPacketStream packet(buffer, bufferSize);
            std::string messageName = subscriber->getSubject();
            packet << osc::BeginMessage(messageName.c_str());
            try
            {
                for (std::vector<std::string>::iterator itArgument = argumentList.begin(); itArgument != argumentList.end(); itArgument++ )
                    createOSCMessage(subscriber, *itArgument, packet);
                packet << osc::EndMessage;
                transmitSocket.Send(packet.Data(), packet.Size());
            } catch(const std::exception& e) {
                if (isVerbose())
                    msg_info("ServerCommunicationOSC") << e.what();
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(int(1000000.0/(double)this->d_refreshRate.getValue())));
    }
}

void ServerCommunicationOSC::createOSCMessage(CommunicationSubscriber* subscriber, std::string argument, osc::OutboundPacketStream& packet)
{    
    BaseData* data = fetchDataFromSenderBuffer(subscriber, argument);
    if (!data)
        throw std::invalid_argument("data is null");

    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
    const void* valueVoidPtr = data->getValueVoidPtr();

    if (typeinfo->Container())
    {
        int nbRows = typeinfo->size();
        int nbCols  = typeinfo->size(data->getValueVoidPtr()) / typeinfo->size();
        packet << "matrix" << nbRows << nbCols;

        if( !typeinfo->Text() && !typeinfo->Scalar() && !typeinfo->Integer() )
        {
            msg_advice(data->getOwner()) << "BaseData_getAttr_value unsupported native type="<<data->getValueTypeString()<<" for data "<<data->getName()<<" ; returning string value" ;
            packet <<  (data->getValueString().c_str());
        }
        else if (typeinfo->Text())
            for (int i=0; i < nbRows; i++)
                for (int j=0; j<nbCols; j++)
                    packet << typeinfo->getTextValue(valueVoidPtr,(i*nbCols) + j).c_str();
        else if (typeinfo->Scalar())
            for (int i=0; i < nbRows; i++)
                for (int j=0; j<nbCols; j++)
                    packet << typeinfo->getScalarValue(valueVoidPtr,(i*nbCols) + j);
        else if (typeinfo->Integer())
            for (int i=0; i < nbRows; i++)
                for (int j=0; j<nbCols; j++)
                    packet << (int)typeinfo->getIntegerValue(valueVoidPtr,(i*nbCols) + j);
    }
    else
    {
        if (typeinfo->Text())
            packet << (typeinfo->getTextValue(valueVoidPtr,0).c_str());
        else if (typeinfo->Scalar())
            packet << (typeinfo->getScalarValue(valueVoidPtr,0));
        else if (typeinfo->Integer())
            packet << ((int)typeinfo->getIntegerValue(valueVoidPtr,0));
        else
        {
            msg_info(data->getOwner()) << "BaseData_getAttr_value unsupported native type="<<data->getValueTypeString()<<" for data "<<data->getName()<<" ; returning string value" ;
            packet <<  (data->getValueString().c_str());
        }
    }
    delete data;
}

/******************************************************************************
*                                                                             *
* RECEIVE PART                                                                *
*                                                                             *
******************************************************************************/

void ServerCommunicationOSC::receiveData()
{
    m_socket = new UdpListeningReceiveSocket(IpEndpointName( IpEndpointName::ANY_ADDRESS, this->d_port.getValue()), this);
    m_socket->Run();
}

void ServerCommunicationOSC::ProcessMessage( const osc::ReceivedMessage& m, const IpEndpointName& /*remoteEndpoint*/ )
{
    if (!m_running)
        m_socket->Break();
    const char* subject = m.AddressPattern();
    if (!getSubscriberFor(subject))
        return;

    ArgumentList argumentList = convertMessagesToArgumentList(m.ArgumentsBegin(), m.ArgumentsEnd());
    if (argumentList.size() == 0)
        return;

    std::string firstArg = getArgumentValue(argumentList.at(0));
    if (firstArg.compare("matrix") == 0)
    {
        osc::ReceivedMessageArgumentIterator it = m.ArgumentsBegin();
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
            msg_warning() << subject << " is matrix, but message size is not correct. Should be : /subject matrix rows cols value value value... ";

        argumentList = convertMessagesToArgumentList(++it, m.ArgumentsEnd());

        if(argumentList.size() == 0)
        {
            msg_error() << "argument list size is empty";
            return;
        }

        if((unsigned int)row*col != argumentList.size())
        {
            msg_error() << "argument list size is != row/cols; " << argumentList.size() << " instead of " << row*col;
            return;
        }
        saveDatasToReceivedBuffer(subject, argumentList, row, col);
    }
    else
    {
        saveDatasToReceivedBuffer(subject, argumentList, -1, -1);
    }
}

/******************************************************************************
*                                                                             *
* MESSAGE CONVERTION PART                                                     *
*                                                                             *
******************************************************************************/

ArgumentList ServerCommunicationOSC::convertMessagesToArgumentList(osc::ReceivedMessageArgumentIterator it, osc::ReceivedMessageArgumentIterator itEnd)
{
    ArgumentList argumentList;
    for(; it != itEnd; it++)
    {
        std::stringstream stream;
        stream << (*it);
        argumentList.push_back(stream.str());
    }
    return argumentList;
}

std::string ServerCommunicationOSC::getArgumentValue(std::string value)
{
    std::string stringData = value;
    std::string returnValue;
    size_t pos = stringData.find(":");
    stringData.erase(0, pos+1);
    std::remove_copy(stringData.begin(), stringData.end(), std::back_inserter(returnValue), '\'');
    /// remove the first character for OSC-String which is "`"
    returnValue.erase(std::remove(returnValue.begin(), returnValue.end(), '`'), returnValue.end());
    return returnValue;
}

std::string ServerCommunicationOSC::getArgumentType(std::string value)
{
    std::string stringType = value;
    size_t pos = stringType.find(":");
    if (pos == std::string::npos)
        return "s";
    stringType.erase(pos, stringType.size()-1);
    return stringType;
}

} /// communication

} /// component

} /// sofa
