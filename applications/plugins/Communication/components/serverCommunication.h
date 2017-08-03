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
#ifndef SOFA_SERVERCOMMUNICATION_H
#define SOFA_SERVERCOMMUNICATION_H

#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;

#include <sofa/core/objectmodel/Data.h>
using sofa::core::objectmodel::Data;

#include <sofa/helper/vectorData.h>
using sofa::helper::vectorData;

#include <sofa/core/objectmodel/Event.h>
using sofa::core::objectmodel::Event;

#include <sofa/simulation/AnimateBeginEvent.h>

#include <oscpack/osc/OscReceivedElements.h>
#include <oscpack/osc/OscPrintReceivedElements.h>
#include "oscpack/osc/OscPacketListener.h"
#include <oscpack/ip/UdpSocket.h>


#include <pthread.h>
#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <mutex>


namespace sofa
{

namespace component
{

namespace communication
{

std::mutex mutex;

template <class DataTypes>
class OSCMessageListener : public osc::OscPacketListener {
public:
    OSCMessageListener();
    OSCMessageListener(unsigned int);

    vectorData<DataTypes> getDataVector();
    timeval t1, t2;
protected:
    vectorData<DataTypes> m_vector;
    unsigned int m_size;

    virtual void ProcessMessage( const osc::ReceivedMessage& m, const IpEndpointName& remoteEndpoint );
};

template <class DataTypes>
class ServerCommunication : public BaseObject
{

public:
    SOFA_CLASS(SOFA_TEMPLATE(ServerCommunication, DataTypes), BaseObject);

    Data<std::string>           d_adress;
    Data<int>                   d_port;
    Data<unsigned int>          d_nbDataField;
    vectorData<DataTypes>       d_data;
    timeval t1, t2;


    ServerCommunication() ;
    virtual ~ServerCommunication() ;
    virtual void init();
    virtual void handleEvent(Event *);
    virtual std::string getTemplateName() const {return templateName(this);}
    static std::string templateName(const ServerCommunication<DataTypes>* = NULL);

    void openCommunication();

    static void* thread_launcher(void*);

protected:
    OSCMessageListener<DataTypes> d_listener;
    UdpListeningReceiveSocket* d_socket;
    pthread_t m_thread;

};

} /// communication
} /// component
} /// sofa

#endif // SOFA_SERVERCOMMUNICATION_H
