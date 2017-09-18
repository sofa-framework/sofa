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

namespace sofa
{

namespace component
{

namespace communication
{

class SOFA_COMMUNICATION_API ServerCommunicationOSC : public ServerCommunication, public osc::OscPacketListener
{

public:

    typedef ServerCommunication Inherited;
    SOFA_CLASS(ServerCommunicationOSC, Inherited);

    ServerCommunicationOSC() ;
    virtual ~ServerCommunicationOSC();

    //////////////////////////////// Factory OSC type /////////////////////////////////
    typedef CommunicationDataFactory OSCDataFactory;
    OSCDataFactory* getFactoryInstance() override;
    virtual void initTypeFactory() override;
    /////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////// Inherited from OscPacketListener /////////////////////////////////
    virtual void ProcessMessage( const osc::ReceivedMessage& m, const IpEndpointName& remoteEndpoint );
    /////////////////////////////////////////////////////////////////////////////////

    Data<int> d_packetSize;

protected:

    UdpListeningReceiveSocket* m_socket;

    osc::OutboundPacketStream createOSCMessage();
    std::string convertArgumentToStringValue(osc::ReceivedMessageArgumentIterator);

    //////////////////////////////// Inherited from ServerCommunication /////////////////////////////////
    virtual void sendData() override;
    virtual void receiveData() override;
    /////////////////////////////////////////////////////////////////////////////////

};

} /// namespace communication
} /// namespace component
} /// namespace sofa

#endif // SOFA_SERVERCOMMUNICATIONOSC_H
