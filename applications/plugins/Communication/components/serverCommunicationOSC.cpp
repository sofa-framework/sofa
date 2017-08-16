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

template <>
osc::OutboundPacketStream ServerCommunicationOSC<std::string>::createOSCMessage()
{
    char buffer[OUTPUT_BUFFER_SIZE];
    osc::OutboundPacketStream p(buffer, OUTPUT_BUFFER_SIZE );

    p << osc::BeginBundleImmediate;
    std::string messageName = "/" + this->getName();
    p << osc::BeginMessage(messageName.c_str());

//    mutex.lock();
    for(unsigned int i=0; i<d_data_copy.size(); i++)
    {
        std::ostringstream messageStream;
        ReadAccessor<Data<std::string>> data = d_data_copy[i];
        messageStream << data;
        p << messageStream.str().c_str();

    }
//    mutex.unlock();
    p << osc::EndMessage;
    return p;
}

template <>
osc::OutboundPacketStream ServerCommunicationOSC<vector<Vec3d>>::createOSCMessage()
{
    char buffer[OUTPUT_BUFFER_SIZE];
    osc::OutboundPacketStream p(buffer, OUTPUT_BUFFER_SIZE );

    p << osc::BeginBundleImmediate;
    std::string messageName = "/" + this->getName();
    p << osc::BeginMessage(messageName.c_str());

    pthread_mutex_lock(&mutex);
    for(unsigned int i=0; i<d_data_copy.size(); i++)
    {

        ReadAccessor<Data<vector<Vec3d>>> data = *d_data_copy[i];
        for(unsigned int j=0; j<data.size(); j++)
        {
            for(int k=0; k<3; k++)
                p << data[j][k];
        }
    }
    pthread_mutex_unlock(&mutex);
    p << osc::EndMessage;
    return p;
}

template <>
osc::OutboundPacketStream ServerCommunicationOSC<vector<Vec3f>>::createOSCMessage()
{
    char buffer[OUTPUT_BUFFER_SIZE];
    osc::OutboundPacketStream p(buffer, OUTPUT_BUFFER_SIZE );

    p << osc::BeginBundleImmediate;
    std::string messageName = "/" + this->getName();
    p << osc::BeginMessage(messageName.c_str());

    pthread_mutex_lock(&mutex);
    for(unsigned int i=0; i<d_data_copy.size(); i++)
    {

        ReadAccessor<Data<vector<Vec3f>>> data = *d_data_copy[i];
        for(unsigned int j=0; j<data.size(); j++)
        {
            for(int k=0; k<3; k++)
                p << data[j][k];
        }
    }
    pthread_mutex_unlock(&mutex);
    p << osc::EndMessage;
    return p;
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

template<>
std::string ServerCommunicationOSC<vector<Vec3d>>::templateName(const ServerCommunicationOSC<vector<Vec3d>>* object){
    SOFA_UNUSED(object);
    return "Vec3d";
}


template<>
std::string ServerCommunicationOSC<vector<Vec3f>>::templateName(const ServerCommunicationOSC<vector<Vec3f>>* object){
    SOFA_UNUSED(object);
    return "Vec3f";
}

int ServerCommunicationOSCClass = RegisterObject("OSC Communication Server.")
        #ifndef SOFA_FLOAT
        .add< ServerCommunicationOSC<float> >()
        .add< ServerCommunicationOSC<vector<Vec3f>> >()
        #endif

        #ifndef SOFA_DOUBLE
        .add< ServerCommunicationOSC<double> >()
        .add< ServerCommunicationOSC<vector<Vec3d>> >()
        #endif

        .add< ServerCommunicationOSC<int> >()
        .add< ServerCommunicationOSC<std::string> >(true);

} /// communication

} /// component

} /// sofa
