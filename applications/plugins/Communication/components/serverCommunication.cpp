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
#include <Communication/config.h>
#include <Communication/components/serverCommunication.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/BaseMapping.h>

#define BENCHMARK 1;

using sofa::core::RegisterObject ;

#include <iostream>
struct timeval start, end;

long mtime, seconds, useconds;

namespace sofa
{

namespace component
{

namespace communication
{


/******************************************************************************************************************* OSCMESSAGELISTENER
***************************************************************************************************************************************
***************************************************************************************************************************************
***************************************************************************************************************************************/

template <class DataTypes>
OSCMessageListener<DataTypes>::OSCMessageListener() : osc::OscPacketListener()
{
    gettimeofday(&t2, NULL);
    gettimeofday(&t1, NULL);
    m_vector.resize(1);
}

template <class DataTypes>
OSCMessageListener<DataTypes>::OSCMessageListener(unsigned int size) : osc::OscPacketListener()
{
    m_size = size;
    m_vector.resize(m_size);
}

template <class DataTypes>
void OSCMessageListener<DataTypes>::ProcessMessage( const osc::ReceivedMessage& m, const IpEndpointName& remoteEndpoint )
{
    if(m_size != m.ArgumentCount())
    {
        std::cout << "Error : received " << m.ArgumentCount() << " argument(s) but defined size is " << m_size << std::endl;
        return;
    }

    try{
        //        gettimeofday(&t1, NULL);
        //        std::cout << "Delta thread server OSC : " << (t1.tv_usec - t2.tv_usec) / 1000.0 << " ms or " << 1000000.0 / ((t1.tv_usec - t2.tv_usec)) << " hz"<< std::endl;
        //        gettimeofday(&t2, NULL);

        mutex.lock();
        osc::ReceivedMessageArgumentStream args = m.ArgumentStream();

//        for (unsigned int i= 0; i<m_vector.size(); i++)
//        {
//            WriteAccessorVector<Data<DataTypes>> data = d_data[i];
//            (*stream) >> data;
//        }
        mutex.unlock();

    }catch( osc::Exception& e ){
        std::cout << "error while parsing message: " << m.AddressPattern() << ": " << e.what() << "\n";
    }
}

template <class DataTypes>
std::vector<DataTypes> OSCMessageListener<DataTypes>::getDataVector()
{
    mutex.lock();
    std::vector<DataTypes> tmp = m_vector;
    mutex.unlock();
    return tmp;
}

/******************************************************************************************************************* SERVERCOMMUNICATION
***************************************************************************************************************************************
***************************************************************************************************************************************
***************************************************************************************************************************************/

template <class DataTypes>
ServerCommunication<DataTypes>::ServerCommunication()
    : d_adress(initData(&d_adress, "127.0.0.1", "adress", "Scale for object display. (default=localhost)"))
    , d_port(initData(&d_port, (int)(6000), "port", "Port to listen (default=6000)"))
    , d_nbDataField(initData(&d_nbDataField, (unsigned int)3, "nbDataField",
                             "Number of field 'data' the user want to send or receive.\n"
                             "Default value is 1."))
    , d_data(this, "data", "Data to send or receive.")
{
    d_listener = OSCMessageListener<DataTypes>(d_nbDataField.getValue());
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
    d_socket = new UdpListeningReceiveSocket(IpEndpointName( IpEndpointName::ANY_ADDRESS, d_port.getValue()), &d_listener);
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
    f_listening = true;
    pthread_create(&m_thread, NULL, &ServerCommunication::thread_launcher, this);
}

template <class DataTypes>
void ServerCommunication<DataTypes>::handleEvent(Event* event)
{
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
    {
        //        std::cout << "Begin ANIMATION" << std::endl;
        //        gettimeofday(&t1, NULL);
        //        std::cout << "Delta mainloop ANIMATION : " << (t1.tv_usec - t2.tv_usec) / 1000.0 << " ms or " << 1000000.0 / ((t1.tv_usec - t2.tv_usec)) << " hz"<< std::endl;
        //        gettimeofday(&t2, NULL);
        std::vector<DataTypes> vector = d_listener.getDataVector();
        for (int i = 0; i < vector.size(); i++)
        {
            std::cout << vector.at(i) << std::endl;
        }

    }
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
