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

#define BENCHMARK 1;

using sofa::core::RegisterObject ;

namespace sofa
{

namespace component
{

namespace communication
{

template <class DataTypes>
ServerCommunication<DataTypes>::ServerCommunication()
    : osc::OscPacketListener()
    , d_adress(initData(&d_adress, "127.0.0.1", "adress", "Scale for object display. (default=localhost)"))
    , d_port(initData(&d_port, (int)(6000), "port", "Port to listen (default=6000)"))
    , d_refreshRate(initData(&d_refreshRate, (int)(30), "refreshRate", "Refresh rate aka frequency (default=30)"))
    , d_nbDataField(initData(&d_nbDataField, (unsigned int)1, "nbData",
                             "Number of field 'data' the user want to send or receive.\n"
                             "Default value is 1."))
    , d_data(this, "data", "Data to send or receive.")
{
    d_data.resize(d_nbDataField.getValue());
    gettimeofday(&t2, NULL);
    gettimeofday(&t1, NULL);
}

template <class DataTypes>
ServerCommunication<DataTypes>::~ServerCommunication()
{
    d_socket->Break();
    pthread_join(m_thread, NULL);
    free(d_socket);
}

template <class DataTypes>
void ServerCommunication<DataTypes>::openCommunication()
{
    d_socket = new UdpListeningReceiveSocket(IpEndpointName( IpEndpointName::ANY_ADDRESS, d_port.getValue()), this);
    d_socket->Run();
}

template <class DataTypes>
void * ServerCommunication<DataTypes>::thread_launcher(void *voidArgs)
{
    ServerCommunication *args = (ServerCommunication*)voidArgs;
    args->openCommunication();
}

template <class DataTypes>
void ServerCommunication<DataTypes>::init()
{
    d_data.resize(d_nbDataField.getValue());
    f_listening = true;
    pthread_create(&m_thread, NULL, &ServerCommunication::thread_launcher, this);
}

template <class DataTypes>
void ServerCommunication<DataTypes>::handleEvent(Event* event)
{
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
    {
        gettimeofday(&t1, NULL);
        std::cout << "Delta mainloop ANIMATION : " << (t1.tv_usec - t2.tv_usec) / 1000.0 << " ms or " << 1000000.0 / ((t1.tv_usec - t2.tv_usec)) << " hz"<< std::endl;
        gettimeofday(&t2, NULL);
        mutex.lock();
        for(unsigned int i=0; i<d_data.size(); i++)
        {
            ReadAccessor<Data<DataTypes>> data = d_data[i];
        }
        mutex.unlock();

    }
}

template <class DataTypes>
void ServerCommunication<DataTypes>::ProcessMessage( const osc::ReceivedMessage& m, const IpEndpointName& remoteEndpoint )
{
    if(d_data.size() != m.ArgumentCount())
    {
        std::cout << "Error : received " << m.ArgumentCount() << " argument(s) but defined size is " << d_nbDataField.getValue() << std::endl;
        return;
    }

    try{
        ///       Max speed : up to 14khz
        gettimeofday(&t1, NULL);
        std::cout << "Delta thread server OSC : " << (t1.tv_usec - t2.tv_usec) / 1000.0 << " ms or " << 1000000.0 / ((t1.tv_usec - t2.tv_usec)) << " hz"<< std::endl;
        gettimeofday(&t2, NULL);

        osc::ReceivedMessageArgumentStream args = m.ArgumentStream();
        int i = 0;
        mutex.lock();
        for ( osc::ReceivedMessageArgumentIterator it = m.ArgumentsBegin()++; it != m.ArgumentsEnd(); it++)
        {
            std::stringstream stream;
            stream << (*it);
            std::string s = stream.str();
            size_t pos = s.find(":"); // That's how OSC message works -> Type:Value
            s.erase(0, pos+1);
            stream.str(s);
            WriteAccessor<Data<DataTypes>> data = d_data[i];
            stream >> data;
            i++;
        }
        mutex.unlock();
    }catch( osc::Exception& e ){
        std::cout << "error while parsing message: " << m.AddressPattern() << ": " << e.what() << "\n";
    }
    usleep(1000000.0/(double)d_refreshRate.getValue());
}



/******************************************************************************************************************* TEMPLATEDEFINITION
***************************************************************************************************************************************
***************************************************************************************************************************************
***************************************************************************************************************************************/

template<>
std::string ServerCommunication<double>::templateName(const ServerCommunication<double>* object){
    SOFA_UNUSED(object);
    return "double";
}

template<>
std::string ServerCommunication<float>::templateName(const ServerCommunication<float>* object){
    SOFA_UNUSED(object);
    return "float";
}

template<>
std::string ServerCommunication<int>::templateName(const ServerCommunication<int>* object){
    SOFA_UNUSED(object);
    return "int";
}

template<>
std::string ServerCommunication<unsigned int>::templateName(const ServerCommunication<unsigned int>* object){
    SOFA_UNUSED(object);
    return "unsigned int";
}

int ServerCommunicationClass = RegisterObject("OSC Communication Server.")
        #ifndef SOFA_FLOAT
        .add< ServerCommunication<float> >()
        #endif
        #ifndef SOFA_DOUBLE
        .add< ServerCommunication<double> >()
        #endif
        .add< ServerCommunication<int> >(true)
        .add< ServerCommunication<unsigned int> >()
        ;
} /// communication

} /// component

} /// sofa
