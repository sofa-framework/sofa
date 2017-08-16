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
#ifndef SOFA_SERVERCOMMUNICATIONOSC_H
#define SOFA_SERVERCOMMUNICATIONOSC_H

#include "serverCommunication.h"

#include <oscpack/osc/OscReceivedElements.h>
#include <oscpack/osc/OscPrintReceivedElements.h>
#include <oscpack/osc/OscPacketListener.h>
#include <oscpack/osc/OscOutboundPacketStream.h>
#include <oscpack/ip/UdpSocket.h>

#define OUTPUT_BUFFER_SIZE 65536


namespace sofa
{

namespace component
{

namespace communication
{

template <class DataTypes>
class ServerCommunicationOSC : public ServerCommunication<DataTypes>, public osc::OscPacketListener
{


public:

    SOFA_CLASS(SOFA_TEMPLATE(ServerCommunicationOSC, DataTypes), SOFA_TEMPLATE(ServerCommunication, DataTypes));

    ServerCommunicationOSC() ;
    virtual ~ServerCommunicationOSC();
    virtual void sendData();
    virtual void receiveData();

    virtual std::string getTemplateName() const {return templateName(this);}
    static std::string templateName(const ServerCommunicationOSC<DataTypes>* = NULL);

    virtual void ProcessMessage( const osc::ReceivedMessage& m, const IpEndpointName& remoteEndpoint );

protected:
    UdpListeningReceiveSocket* d_socket;

    osc::OutboundPacketStream createOSCMessage();

};

} /// communication
} /// component
} /// sofa

#endif // SOFA_SERVERCOMMUNICATIONOSC_H
