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
{
}

ServerCommunicationOSC::OSCDataFactory* ServerCommunicationOSC::getFactoryInstance(){
    static OSCDataFactory* s_localfactory = nullptr ;
    if(s_localfactory==nullptr)
        s_localfactory = new ServerCommunicationOSC::OSCDataFactory() ;
    return s_localfactory ;
}

ServerCommunicationOSC::~ServerCommunicationOSC()
{
    m_socket->Break();
    free(m_socket);
    Inherited::closeCommunication();
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
    while (this->m_running)
    {
#if BENCHMARK
        // Uncorrect results if frequency == 1hz, due to tv_usec precision
        gettimeofday(&t1, NULL);
        if(d_refreshRate.getValue() <= 1.0)
            std::cout << "Thread Sender OSC : " << fabs((t1.tv_sec - t2.tv_sec)) << " s or " << fabs(1.0 / ((t1.tv_sec - t2.tv_sec))) << " hz"<< std::endl;
        else
            std::cout << "Thread Sender OSC : " << fabs((t1.tv_usec - t2.tv_usec) / 1000.0) << " ms or " << fabs(1000000.0 / ((t1.tv_usec - t2.tv_usec))) << " hz"<< std::endl;
        gettimeofday(&t2, NULL);
#endif
        std::cout << "sernder" << std::endl;
        UdpTransmitSocket transmitSocket( IpEndpointName(this->d_address.getValue().c_str(), this->d_port.getValue()));
        osc::OutboundPacketStream p = createOSCMessage();
        transmitSocket.Send( p.Data(), p.Size() );
        usleep(1000000.0/(double) this->d_refreshRate.getValue());
    }
}

void ServerCommunicationOSC::receiveData()
{
    m_socket = new UdpListeningReceiveSocket(IpEndpointName( IpEndpointName::ANY_ADDRESS, this->d_port.getValue()), this);
    m_socket->Run();
}

osc::OutboundPacketStream ServerCommunicationOSC::createOSCMessage()
{
    //    char buffer[OUTPUT_BUFFER_SIZE];
    //    osc::OutboundPacketStream p(buffer, OUTPUT_BUFFER_SIZE );

    //    for(unsigned int i=0; i<this->d_data_copy.size(); i++)
    //    {
    //        std::string messageName = "/" + this->getName() + "/" + this->d_data_copy[i]->getName();
    //        p << osc::BeginMessage(messageName.c_str());
    //        ReadAccessor<Data<DataTypes>> data = this->d_data_copy[i];
    //        p << data;
    //        p << osc::EndMessage;
    //    }//  << osc::EndBundle; Don't know why but osc::EndBundle made it crash, anyway it's working ... TODO have a look
    //    return p;
}

void ServerCommunicationOSC::ProcessMessage( const osc::ReceivedMessage& m, const IpEndpointName& remoteEndpoint )
{

#if BENCHMARK
    // Uncorrect results if frequency == 1hz, due to tv_usec precision
    gettimeofday(&t1, NULL);
    if(d_refreshRate.getValue() <= 1.0)
        std::cout << "Thread Receiver OSC : " << fabs((t1.tv_sec - t2.tv_sec)) << " s or " << fabs(1.0 / ((t1.tv_sec - t2.tv_sec))) << " hz"<< std::endl;
    else
        std::cout << "Thread Receiver OSC : " << fabs((t1.tv_usec - t2.tv_usec) / 1000.0) << " ms or " << fabs(1000000.0 / ((t1.tv_usec - t2.tv_usec))) << " hz"<< std::endl;
    gettimeofday(&t2, NULL);
#endif

    const char* address = m.AddressPattern();
    if (!isSubscribedTo(m.AddressPattern(), m.ArgumentCount()))
        return;

    CommunicationSubscriber * subscriber = getSubscriberFor(address);

    int i = 0;
    for ( osc::ReceivedMessageArgumentIterator it = m.ArgumentsBegin()++; it != m.ArgumentsEnd(); it++)
    {
        std::string keyTypeMessage = std::string(1, it->TypeTag());
        sofa::helper::NoArgument noArg;
        std::string argumentName = subscriber->getArgumentName(i);
        MapData dataMap = getDataAliases();

        std::cout << "DATA =---------------" << std::endl;
        for (MapData::const_iterator itDrawData = dataMap.begin(); itDrawData != dataMap.end(); itDrawData++)
        {
            BaseData * data = itDrawData->second;
            std::cout << itDrawData->first << " : " << data->getValueString() << " type: " << data->getValueTypeString()  << std::endl;
        }

        MapData::const_iterator itData = dataMap.find(argumentName);
        if (itData == dataMap.end())
        {
            BaseData* data = getFactoryInstance()->createObject(keyTypeMessage, noArg);
            if (data == nullptr)
                msg_warning() << keyTypeMessage << " is not a known type";
            else
                addData(data, argumentName);
        } else
        {
            BaseData* data = itData->second;
            std::stringstream stream;
            stream << (*it);
            std::string s = stream.str();
            size_t pos = s.find(":"); // That's how OSC message works -> Type:Value
            s.erase(0, pos+1);
            stream.str(s);
            data->read(stream.str());
        }
        i++;
    }
}


} /// communication

} /// component

} /// sofa
