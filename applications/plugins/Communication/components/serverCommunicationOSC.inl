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
#include <Communication/components/serverCommunicationOSC.h>

using sofa::core::RegisterObject ;

namespace sofa
{

namespace component
{

namespace communication
{

template <class DataTypes>
ServerCommunicationOSC<DataTypes>::ServerCommunicationOSC()
    : osc::OscPacketListener(), ServerCommunication<DataTypes>()
{
}

template <class DataTypes>
ServerCommunicationOSC<DataTypes>::~ServerCommunicationOSC()
{
    d_socket->Break();
    free(d_socket);
}

template <class DataTypes>
void ServerCommunicationOSC<DataTypes>::sendData()
{
    UdpTransmitSocket transmitSocket( IpEndpointName(this->d_adress.getValue().c_str(), this->d_port.getValue()));
#if BENCHMARK
    // Uncorrect results if frequency == 1hz, due to tv_usec precision
    gettimeofday(&t1, NULL);
    if(d_refreshRate.getValue() <= 1.0)
        std::cout << "Thread Sender OSC : " << fabs((t1.tv_sec - t2.tv_sec)) << " s or " << fabs(1.0 / ((t1.tv_sec - t2.tv_sec))) << " hz"<< std::endl;
    else
        std::cout << "Thread Sender OSC : " << fabs((t1.tv_usec - t2.tv_usec) / 1000.0) << " ms or " << fabs(1000000.0 / ((t1.tv_usec - t2.tv_usec))) << " hz"<< std::endl;
    gettimeofday(&t2, NULL);
#endif

    osc::OutboundPacketStream p = createOSCMessage();
    transmitSocket.Send( p.Data(), p.Size() );
    usleep(1000000.0/(double) this->d_refreshRate.getValue());
}

template <class DataTypes>
osc::OutboundPacketStream ServerCommunicationOSC<DataTypes>::createOSCMessage()
{
    char buffer[OUTPUT_BUFFER_SIZE];
    osc::OutboundPacketStream p(buffer, OUTPUT_BUFFER_SIZE );

    p << osc::BeginBundleImmediate;
    std::string messageName = "/" + this->getName();
    p << osc::BeginMessage(messageName.c_str());
//    mutex.lock();
    for(unsigned int i=0; i<this->d_data_copy.size(); i++)
    {
        ReadAccessor<Data<DataTypes>> data = this->d_data_copy[i];
        p << data;
    }
//    mutex.unlock();
    p << osc::EndMessage; //  << osc::EndBundle; Don't know why but osc::EndBundle made it crash, anyway it's working ... TODO have a look
    return p;
}

template <class DataTypes>
void ServerCommunicationOSC<DataTypes>::receiveData()
{
    d_socket = new UdpListeningReceiveSocket(IpEndpointName( IpEndpointName::ANY_ADDRESS, this->d_port.getValue()), this);
    d_socket->Run();
}

template <class DataTypes>
void ServerCommunicationOSC<DataTypes>::ProcessMessage( const osc::ReceivedMessage& m, const IpEndpointName& remoteEndpoint )
{
    if(this->d_data.size() != m.ArgumentCount())
    {
        msg_warning("") << "Warning: received " << m.ArgumentCount() << " argument(s) from OSC but defined size is " << this->d_nbDataField.getValue();
        return;
    }

#if BENCHMARK
    // Uncorrect results if frequency == 1hz, due to tv_usec precision
    gettimeofday(&t1, NULL);
    if(d_refreshRate.getValue() <= 1.0)
        std::cout << "Thread Receiver OSC : " << fabs((t1.tv_sec - t2.tv_sec)) << " s or " << fabs(1.0 / ((t1.tv_sec - t2.tv_sec))) << " hz"<< std::endl;
    else
        std::cout << "Thread Receiver OSC : " << fabs((t1.tv_usec - t2.tv_usec) / 1000.0) << " ms or " << fabs(1000000.0 / ((t1.tv_usec - t2.tv_usec))) << " hz"<< std::endl;
    gettimeofday(&t2, NULL);
#endif

    int i = 0;
//    mutex.lock();
    for ( osc::ReceivedMessageArgumentIterator it = m.ArgumentsBegin()++; it != m.ArgumentsEnd(); it++)
    {
        std::stringstream stream;
        stream << (*it);
        std::string s = stream.str();
        size_t pos = s.find(":"); // That's how OSC message works -> Type:Value
        s.erase(0, pos+1);
        stream.str(s);
        WriteAccessor<Data<DataTypes>> data = this->d_data[i];
        stream >> data;
        std::cout << stream.str() << std::endl;
        i++;
    }
//    mutex.unlock();
}

} /// communication

} /// component

} /// sofa
