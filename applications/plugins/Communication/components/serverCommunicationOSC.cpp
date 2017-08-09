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
#include "serverCommunicationOSC.inl"

using sofa::core::RegisterObject ;

namespace sofa
{

namespace component
{

namespace communication
{


/******************************************************************************************************************* TEMPLATEDEFINITION
***************************************************************************************************************************************
***************************************************************************************************************************************
***************************************************************************************************************************************/

template <>
void ServerCommunicationOSC<std::string>::sendData()
{
    UdpTransmitSocket transmitSocket( IpEndpointName(d_adress.getValue().c_str(), d_port.getValue()));
#if BENCHMARK
    // Uncorrect results if frequency == 1hz, due to tv_usec precision
    gettimeofday(&t1, NULL);
    if(d_refreshRate.getValue() <= 1.0)
        std::cout << "Thread Sender OSC : " << fabs((t1.tv_sec - t2.tv_sec)) << " s or " << fabs(1.0 / ((t1.tv_sec - t2.tv_sec))) << " hz"<< std::endl;
    else
        std::cout << "Thread Sender OSC : " << fabs((t1.tv_usec - t2.tv_usec) / 1000.0) << " ms or " << fabs(1000000.0 / ((t1.tv_usec - t2.tv_usec))) << " hz"<< std::endl;
    gettimeofday(&t2, NULL);
#endif
    char buffer[OUTPUT_BUFFER_SIZE];
    osc::OutboundPacketStream p(buffer, OUTPUT_BUFFER_SIZE );

    p << osc::BeginBundleImmediate;
    std::string messageName = "/" + this->getName();
    p << osc::BeginMessage(messageName.c_str());

    mutex.lock();
    for(unsigned int i=0; i<d_data.size(); i++)
    {
        std::ostringstream messageStream;
        ReadAccessor<Data<std::string>> data = d_data[i];
        messageStream << data;
        p << messageStream.str().c_str();

    }
    mutex.unlock();
    p << osc::EndMessage; //  << osc::EndBundle; Don't know why but osc::EndBundle made it crash, anyway it's working ... TODO have a look to EndBundle
    transmitSocket.Send( p.Data(), p.Size() );
    usleep(1000000.0/(double)d_refreshRate.getValue());
}

template<>
std::string ServerCommunicationOSC<double>::templateName(const ServerCommunicationOSC<double>* object)
{
    SOFA_UNUSED(object);
    return "double";
}

template<>
std::string ServerCommunicationOSC<float>::templateName(const ServerCommunicationOSC<float>* object)
{
    SOFA_UNUSED(object);
    return "float";
}

template<>
std::string ServerCommunicationOSC<int>::templateName(const ServerCommunicationOSC<int>* object)
{
    SOFA_UNUSED(object);
    return "int";
}

template<>
std::string ServerCommunicationOSC<std::string>::templateName(const ServerCommunicationOSC<std::string>* object)
{
    SOFA_UNUSED(object);
    return "string";
}

int ServerCommunicationOSCClass = RegisterObject("OSC Communication Server.")
        #ifndef SOFA_FLOAT
        .add< ServerCommunicationOSC<float> >()
        #endif
        #ifndef SOFA_DOUBLE
        .add< ServerCommunicationOSC<double> >()
        #endif
        .add< ServerCommunicationOSC<int> >()
        .add< ServerCommunicationOSC<std::string> >(true);

} /// communication

} /// component

} /// sofa
