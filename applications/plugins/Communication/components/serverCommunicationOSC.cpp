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

///// STD::STRING

template <>
osc::OutboundPacketStream ServerCommunicationOSC<std::string>::createOSCMessage()
{
    char buffer[OUTPUT_BUFFER_SIZE];
    osc::OutboundPacketStream p(buffer, OUTPUT_BUFFER_SIZE );

    pthread_mutex_lock(&mutex);
    for(unsigned int i=0; i<d_data_copy.size(); i++)
    {
        std::string messageName = "/" + this->getName() + "/" + this->d_data_copy[i]->getName();
        p << osc::BeginMessage(messageName.c_str());
        std::ostringstream messageStream;
        ReadAccessor<Data<std::string>> data = d_data_copy[i];
        messageStream << data;
        p << messageStream.str().c_str();
        p << osc::EndMessage;
    }
    pthread_mutex_unlock(&mutex);
    return p;
}

///// VEC3D

template <>
osc::OutboundPacketStream ServerCommunicationOSC<vector<Vec3d>>::createOSCMessage()
{
    char buffer[OUTPUT_BUFFER_SIZE];
    osc::OutboundPacketStream p(buffer, OUTPUT_BUFFER_SIZE );
    p << osc::BeginBundle();
    pthread_mutex_lock(&mutex);
    for(unsigned int i=0; i<d_data_copy.size(); i++)
    {
        ReadAccessor<Data<vector<Vec3d>>> data = *d_data_copy[i];

        if (data.size() == 0)
        {
            std::string messageName = "/" + this->getName() + "/" + this->d_data_copy[i]->getName();
            p << osc::BeginMessage(messageName.c_str());
            p << "empty";
            p << osc::EndMessage;
        }
        for(unsigned int j=0; j<data.size(); j++)
        {
            std::string messageName = "/" + this->getName() + "/" + this->d_data_copy[i]->getName() + "/" + std::to_string(j);
            p << osc::BeginMessage(messageName.c_str());
            for(int k=0; k<3; k++)
            {
                p << (double)data[j][k];
            }
            p << osc::EndMessage;
        }
    }
    pthread_mutex_unlock(&mutex);
    p << osc::EndBundle;
}

///// VEC3F

template <>
osc::OutboundPacketStream ServerCommunicationOSC<vector<Vec3f>>::createOSCMessage()
{
    char buffer[OUTPUT_BUFFER_SIZE];
    osc::OutboundPacketStream p(buffer, OUTPUT_BUFFER_SIZE );
    pthread_mutex_lock(&mutex);
    for(unsigned int i=0; i<d_data_copy.size(); i++)
    {
        ReadAccessor<Data<vector<Vec3f>>> data = *d_data_copy[i];

        if (data.size() == 0)
        {
            std::string messageName = "/" + this->getName() + "/" + this->d_data_copy[i]->getName();
            p << osc::BeginMessage(messageName.c_str());
            p << "empty";
            p << osc::EndMessage;
        }
        for(unsigned int j=0; j<data.size(); j++)
        {
            std::string messageName = "/" + this->getName() + "/" + this->d_data_copy[i]->getName() + "/" + std::to_string(j);
            p << osc::BeginMessage(messageName.c_str());
            for(int k=0; k<3; k++)
            {
                p << data[j][k];
            }
            p << osc::EndMessage;
        }
    }
    pthread_mutex_unlock(&mutex);
    std::cout << osc::ReceivedPacket(p.Data(), p.Size()) << std:: endl;
    std::cout << p.Size() << " " << 0x7FFFFFFF << " " << osc::IsMultipleOf4(p.Size()) << std::endl;
    std::cout << osc::IsValidElementSizeValue(p.Size()) <<  std::endl;
    std::cout << p.Data() << std::endl;
    return p;
}

//////////////////////////////// Template name definition

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


////////////////////////////////////////////    FACTORY    ////////////////////////////////////////////

// Registering the component
// see: http://wiki.sofa-framework.org/wiki/ObjectFactory
// 1-SOFA_DECL_CLASS(componentName) : Set the class name of the component
// 2-RegisterObject("description") + .add<> : Register the component
// 3-.add<>(true) : Set default template
SOFA_DECL_CLASS(ServerCommunicationOSC)
int ServerCommunicationOSCClass = RegisterObject("This component is used to build a communication between two simulations using OSC protocol")
#ifdef SOFA_WITH_DOUBLE
.add< ServerCommunicationOSC<float> >()
.add< ServerCommunicationOSC<vector<Vec3f>> >()
#endif
#ifdef SOFA_WITH_DOUBLE
.add< ServerCommunicationOSC<double> >()
.add< ServerCommunicationOSC<vector<Vec3d>> >()
#endif
.add< ServerCommunicationOSC<int> >()
.add< ServerCommunicationOSC<std::string> >(true);

///////////////////////////////////////////////////////////////////////////////////////////////////////

// Force template specialization for the most common sofa floating point related type.
// This goes with the extern template declaration in the .h. Declaring extern template
// avoid the code generation of the template for each compilation unit.
// see: http://www.stroustrup.com/C++11FAQ.html#extern-templates
#ifdef SOFA_WITH_DOUBLE
template class SOFA_COMMUNICATION_API ServerCommunicationOSC<double>;
template class SOFA_COMMUNICATION_API ServerCommunicationOSC<vector<Vec3d>>;
#endif
#ifdef SOFA_WITH_FLOAT
template class SOFA_COMMUNICATION_API ServerCommunicationOSC<float>;
template class SOFA_COMMUNICATION_API ServerCommunicationOSC<vector<Vec3f>>;
#endif
template class SOFA_COMMUNICATION_API ServerCommunicationOSC<int>;
template class SOFA_COMMUNICATION_API ServerCommunicationOSC<std::string>;

} /// communication

} /// component

} /// sofa
