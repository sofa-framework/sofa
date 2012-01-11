/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "OptiTrackNatNetClient.h"
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/core/ObjectFactory.h>

#include <boost/bind.hpp>

namespace SofaOptiTrackNatNet
{

//////////
// From PacketClient.cpp in NatNet SDK

#define MAX_NAMELENGTH              256

// NATNET message ids
#define NAT_PING                    0
#define NAT_PINGRESPONSE            1
#define NAT_REQUEST                 2
#define NAT_RESPONSE                3
#define NAT_REQUEST_MODELDEF        4
#define NAT_MODELDEF                5
#define NAT_REQUEST_FRAMEOFDATA     6
#define NAT_FRAMEOFDATA             7
#define NAT_MESSAGESTRING           8
#define NAT_UNRECOGNIZED_REQUEST    100
//#define UNDEFINED                   999999.9999
#if defined(__GNUC__)
#pragma pack(push,1)
#endif

// sender
struct sSender
{
    char szName[MAX_NAMELENGTH];            // sending app's name
    unsigned char Version[4];               // sending app's version [major.minor.build.revision]
    unsigned char NatNetVersion[4];         // sending app's NatNet version [major.minor.build.revision]

};

struct sPacket
{
    unsigned short iMessage;                // message ID (e.g. NAT_FRAMEOFDATA)
    unsigned short nDataBytes;              // Num bytes in payload
    union
    {
        unsigned char  cData[20000];
        char           szData[20000];
        unsigned long  lData[5000];
        float          fData[5000];
        sSender        Sender;
    } Data;                                 // Payload

};

#if defined(__GNUC__)
#pragma pack(pop)
#endif

#define MULTICAST_ADDRESS		"239.255.42.99"     // IANA, local network
#define PORT_COMMAND            1510
#define PORT_DATA  			    1511                // Default multicast group

//////////

using boost::asio::ip::udp;

boost::asio::io_service& OptiTrackNatNetClient::get_io_service()
{
    static boost::asio::io_service io_service;
    return io_service;
}

boost::asio::ip::udp::resolver& OptiTrackNatNetClient::get_resolver()
{
    static boost::asio::ip::udp::resolver resolver(get_io_service());
    return resolver;
}

OptiTrackNatNetClient::OptiTrackNatNetClient()
    : serverName(initData(&serverName, std::string("localhost"), "serverName", "NatNet server address (default to localhost)"))
    , clientName(initData(&clientName, std::string(""), "clientName", "IP to bind this client to (default to localhost)"))
    , command_socket(NULL)
    , data_socket(NULL)
    , recv_command_packet(new sPacket)
    , recv_data_packet(new sPacket)
{
    this->f_listening.setValue(true);
    this->f_printLog.setValue(true);
}

OptiTrackNatNetClient::~OptiTrackNatNetClient()
{
    delete command_socket;
    delete data_socket;
    delete recv_command_packet;
    delete recv_data_packet;
}

void OptiTrackNatNetClient::init()
{
    this->reinit();
}

void OptiTrackNatNetClient::reinit()
{
    sout << "Connecting to " << serverName.getValue() << sendl;

    bool connected = connect();
    if (!connected)
        serr << "Connection failed" << std::endl;

}

bool OptiTrackNatNetClient::connect()
{
    if (command_socket) delete command_socket; command_socket = NULL;
    if (data_socket) delete data_socket; data_socket = NULL;

    boost::system::error_code ec;

    {
        sout << "Resolving " <<  serverName.getValue() << sendl;
        udp::resolver::query query(udp::v4(), serverName.getValue(), "0");
        udp::resolver::iterator result = get_resolver().resolve(query, ec);
        if (ec)
        {
            serr << ec.category().name() << " ERROR while resolving " << serverName.getValue() << " : " << ec.message() << sendl;
            return false;
        }

        server_endpoint = *result;
        server_endpoint.port(PORT_COMMAND);
        sout << "Resolved " <<  serverName.getValue() << " to " << server_endpoint << sendl;
    }

    udp::endpoint client_endpoint(udp::v4(), PORT_DATA);
    if (!clientName.getValue().empty())
    {
        sout << "Resolving " <<  clientName.getValue() << sendl;
        udp::resolver::query query(udp::v4(), clientName.getValue(), "0");
        udp::resolver::iterator result = get_resolver().resolve(query, ec);
        if (ec)
        {
            serr << ec.category().name() << " ERROR while resolving " << clientName.getValue() << " : " << ec.message() << sendl;
            return false;
        }

        client_endpoint = *result;
        client_endpoint.port(PORT_DATA);
        sout << "Resolved " <<  clientName.getValue() << " to " << client_endpoint << sendl;
    }

    sout << "Opening data socket on " <<  client_endpoint << sendl;
    data_socket = new udp::socket(get_io_service());
    if (data_socket->open(udp::v4(), ec))
    {
        serr << ec.category().name() << " ERROR while opening data socket : " << ec.message() << sendl;
        return false;
    }

    data_socket->set_option(udp::socket::reuse_address(true));
    if (data_socket->bind(client_endpoint, ec))
    {
        serr << ec.category().name() << " ERROR while binding data socket : " << ec.message() << sendl;
        return false;
    }

    // Join the multicast group.
    if (!clientName.getValue().empty())
        data_socket->set_option(boost::asio::ip::multicast::join_group(boost::asio::ip::address::from_string(MULTICAST_ADDRESS).to_v4(), client_endpoint.address().to_v4()));
    else
        data_socket->set_option(boost::asio::ip::multicast::join_group(boost::asio::ip::address::from_string(MULTICAST_ADDRESS)));

    sout << "Data socket ready" << sendl;
    start_data_receive();

    sout << "Opening command socket" << sendl;

    command_socket = new udp::socket(get_io_service());
    if (command_socket->open(udp::v4(), ec))
    {
        serr << ec.category().name() << " ERROR while opening command socket : " << ec.message() << sendl;
        return false;
    }

    if (!clientName.getValue().empty())
    {
        client_endpoint.port(0);
        if (command_socket->bind(client_endpoint, ec))
        {
            serr << ec.category().name() << " ERROR while binding command socket : " << ec.message() << sendl;
            return false;
        }
    }

    sout << "Command socket ready" << sendl;
    start_command_receive();

    //boost::shared_pointer<sPacket> helloMsg = new sPacket;
    //helloMsg.iMessage = NAT_PING;
    //helloMsg.nDataBytes = 0;

    sout << "Sending hello message..." << sendl;

    boost::array<unsigned short, 2> helloMsg;
    helloMsg[0] = NAT_PING; helloMsg[1] = 0;
    command_socket->send_to(boost::asio::buffer(helloMsg), server_endpoint);

    return true;
}

void OptiTrackNatNetClient::start_command_receive()
{
    command_socket->async_receive_from(
        boost::asio::buffer(recv_command_packet, sizeof(sPacket)), recv_command_endpoint,
        boost::bind(&OptiTrackNatNetClient::handle_command_receive, this,
                boost::asio::placeholders::error,
                boost::asio::placeholders::bytes_transferred));
}

void OptiTrackNatNetClient::handle_command_receive(const boost::system::error_code& ec,
        std::size_t bytes_transferred)
{
    if (ec)
    {
        serr << ec.category().name() << " ERROR while receiving command from " << recv_command_endpoint << sendl;
    }
    else
    {
        sout << "Received " << bytes_transferred << "b command from " << recv_command_endpoint << sendl;
        sPacket& PacketIn = *recv_command_packet;
        switch (PacketIn.iMessage)
        {
        case NAT_MODELDEF:
        {
            sout << "Received MODELDEF" << sendl;
            decodeModelDef(PacketIn);
            break;
        }
        case NAT_FRAMEOFDATA:
        {
            sout << "Received FRAMEOFDATA" << sendl;
            decodeFrame(PacketIn);
            break;
        }
        case NAT_PINGRESPONSE:
        {
            serverString = PacketIn.Data.Sender.szName;
            for(int i=0; i<4; i++)
            {
                natNetVersion[i] = PacketIn.Data.Sender.NatNetVersion[i];
                serverVersion[i] = PacketIn.Data.Sender.Version[i];
            }
            serr << "Connected to server \"" << serverString << "\" v" << (int)serverVersion[0];
            if (serverVersion[1] || serverVersion[2] || serverVersion[3])
                serr << "." << (int)serverVersion[1];
            if (serverVersion[2] || serverVersion[3])
                serr << "." << (int)serverVersion[2];
            if (serverVersion[3])
                serr << "." << (int)serverVersion[3];
            serr << " protocol v" << (int)natNetVersion[0];
            if (natNetVersion[1] || natNetVersion[2] || natNetVersion[3])
                serr << "." << (int)natNetVersion[1];
            if (natNetVersion[2] || natNetVersion[3])
                serr << "." << (int)natNetVersion[2];
            if (natNetVersion[3])
                serr << "." << (int)natNetVersion[3];
            serr << sendl;
            // request scene info
            boost::array<unsigned short, 2> reqMsg;
            reqMsg[0] = NAT_REQUEST_MODELDEF; reqMsg[1] = 0;
            command_socket->send_to(boost::asio::buffer(reqMsg), server_endpoint);
            break;
        }
        case NAT_RESPONSE:
        {
            sout << "Received response : " << PacketIn.Data.szData << sendl;
            break;
        }
        case NAT_UNRECOGNIZED_REQUEST:
        {
            serr << "Received 'unrecognized request'" << sendl;
            break;
        }
        case NAT_MESSAGESTRING:
        {
            sout << "Received message: " << PacketIn.Data.szData << sendl;
            break;
        }
        default:
        {
            serr << "Received unrecognized command packet type: " << PacketIn.iMessage << sendl;
            break;
        }
        }
    }
    start_command_receive();
}

void OptiTrackNatNetClient::start_data_receive()
{
    data_socket->async_receive_from(
        boost::asio::buffer(recv_data_packet, sizeof(sPacket)), recv_data_endpoint,
        boost::bind(&OptiTrackNatNetClient::handle_data_receive, this,
                boost::asio::placeholders::error,
                boost::asio::placeholders::bytes_transferred));
}

void OptiTrackNatNetClient::handle_data_receive(const boost::system::error_code& ec,
        std::size_t bytes_transferred)
{
    if (ec)
    {
        serr << ec.category().name() << " ERROR while receiving data from " << recv_data_endpoint << sendl;
    }
    else
    {
        sout << "Received " << bytes_transferred << "b data from " << recv_data_endpoint << sendl;
        sPacket& PacketIn = *recv_data_packet;
        switch (PacketIn.iMessage)
        {
        case NAT_MODELDEF:
        {
            sout << "Received MODELDEF" << sendl;
            decodeModelDef(PacketIn);
            break;
        }
        case NAT_FRAMEOFDATA:
        {
            sout << "Received FRAMEOFDATA" << sendl;
            decodeFrame(PacketIn);
            break;
        }
        default:
        {
            serr << "Received unrecognized data packet type: " << PacketIn.iMessage << sendl;
            break;
        }
        }
    }
    start_data_receive();
}

template<class T>
static void memread(T& dest, const unsigned char*& ptr)
{
    dest = *(const T*)ptr;
    ptr += sizeof(T);
}

static void memread(const char*& dest, const unsigned char*& ptr)
{
    dest = (const char*) ptr;
    ptr += strlen(dest)+1;
}

template<class T>
static void memread(const T*& dest, int n, const unsigned char*& ptr)
{
    if (n <= 0)
        dest = NULL;
    else
    {
        dest = (const T*)ptr;
        ptr += n*sizeof(T);
    }
}

void OptiTrackNatNetClient::decodeFrame(const sPacket& data)
{
    const int major = natNetVersion[0];
    const int minor = natNetVersion[1];

    FrameData frame;

    const unsigned char *ptr = data.Data.cData;

    memread(frame.frameNumber,ptr);

    memread(frame.nPointClouds,ptr);
    if (frame.nPointClouds <= 0)
        frame.pointClouds = NULL;
    else
    {
        frame.pointClouds = new PointCloudData[frame.nPointClouds];
        for (int iP = 0; iP < frame.nPointClouds; ++iP)
        {
            PointCloudData& pdata = frame.pointClouds[iP];
            memread(pdata.name,ptr);
            memread(pdata.nMarkers,ptr);
            memread(pdata.markersPos, pdata.nMarkers, ptr);
        }
    }
    memread(frame.nOtherMarkers,ptr);
    memread(frame.otherMarkersPos, frame.nOtherMarkers, ptr);

    memread(frame.nRigids,ptr);
    if (frame.nRigids <= 0)
        frame.rigids = NULL;
    else
    {
        frame.rigids = new RigidData[frame.nRigids];
        for (int iR = 0; iR < frame.nRigids; ++iR)
        {
            RigidData& rdata = frame.rigids[iR];
            memread(rdata.ID,ptr);
            memread(rdata.pos,ptr);
            memread(rdata.rot,ptr);
            memread(rdata.nMarkers, ptr);
            memread(rdata.markersPos, rdata.nMarkers, ptr);
            if (major < 2)
            {
                rdata.markersID = NULL;
                rdata.markersSize = NULL;
                rdata.meanError = -1.0f;
            }
            else
            {
                memread(rdata.markersID, rdata.nMarkers, ptr);
                memread(rdata.markersSize, rdata.nMarkers, ptr);
                memread(rdata.meanError, ptr);
            }
        }
    }

    if (major <= 1 || (major == 2 && minor <= 0))
    {
        frame.nSkeletons = 0;
        frame.skeletons = NULL;
    }
    else
    {
        memread(frame.nSkeletons,ptr);
        if (frame.nSkeletons <= 0)
            frame.skeletons = NULL;
        else
        {
            frame.skeletons = new SkeletonData[frame.nSkeletons];
            for (int iS = 0; iS < frame.nSkeletons; ++iS)
            {
                SkeletonData& sdata = frame.skeletons[iS];
                memread(sdata.ID,ptr);
                memread(sdata.nRigids, ptr);
                if (sdata.nRigids <= 0)
                    sdata.rigids = NULL;
                else
                {
                    sdata.rigids = new RigidData[sdata.nRigids];
                    for (int iR = 0; iR < sdata.nRigids; ++iR)
                    {
                        RigidData& rdata = sdata.rigids[iR];
                        memread(rdata.ID,ptr);
                        memread(rdata.pos,ptr);
                        memread(rdata.rot,ptr);
                        memread(rdata.nMarkers, ptr);
                        memread(rdata.markersPos, rdata.nMarkers, ptr);
                        memread(rdata.markersID, rdata.nMarkers, ptr);
                        memread(rdata.markersSize, rdata.nMarkers, ptr);
                        memread(rdata.meanError, ptr);
                    }
                }
            }
        }
    }
    memread(frame.latency, ptr);

    processFrame(&frame);

    if (frame.pointClouds)
    {
        delete[] frame.pointClouds;
    }
    if (frame.rigids)
    {
        delete[] frame.rigids;
    }
    if (frame.skeletons)
    {
        for (int i=0; i<frame.nSkeletons; ++i)
        {
            if (frame.skeletons[i].rigids)
                delete[] frame.skeletons[i].rigids;
        }
        delete[] frame.skeletons;
    }
}

void OptiTrackNatNetClient::decodeModelDef(const sPacket& data)
{
}

void OptiTrackNatNetClient::processFrame(const FrameData* data)
{
#define ENDL sendl
//#define ENDL "\n"
    sout << "Frame # : " << data->frameNumber << ENDL;
    sout << "\tPoint Cloud Count : " << data->nPointClouds << ENDL;
    for (int iP = 0; iP < data->nPointClouds; ++iP)
    {
        sout << ENDL;
        if (data->pointClouds[iP].name)
            sout << "\t\tModel Name : " << data->pointClouds[iP].name << ENDL;
        sout << "\t\tMarkers (" << data->pointClouds[iP].nMarkers << ") :";
        for (int i = 0; i < data->pointClouds[iP].nMarkers; ++i)
            sout << " [" << data->pointClouds[iP].markersPos[i] << "]";
        sout << ENDL;
    }

    sout << "\tUnidentified Markers (" << data->nOtherMarkers << ") :";
    for (int i = 0; i < data->nOtherMarkers; ++i)
        sout << " [" << data->otherMarkersPos[i] << "]";
    sout << ENDL;

    sout << "\tRigid Body Count : " << data->nRigids << ENDL;
    for (int iR = 0; iR < data->nRigids; ++iR)
    {
        sout << ENDL;
        sout << "\t\tID : " << data->rigids[iR].ID << ENDL;
        sout << "\t\tpos : " << data->rigids[iR].pos << ENDL;
        if (data->rigids[iR].rot[0] != 0.0f
            || data->rigids[iR].rot[1] != 0.0f
            || data->rigids[iR].rot[2] != 0.0f
            || data->rigids[iR].rot[3] != 0.0f)
            sout << "\t\trot : " << data->rigids[iR].rot << ENDL;
        sout << "\t\tMarkers (" << data->rigids[iR].nMarkers << ") :";
        for (int i = 0; i < data->rigids[iR].nMarkers; ++i)
        {
            sout << " [" << data->rigids[iR].markersPos[i] << "]";
            if (data->rigids[iR].markersID)
                sout << ",id=" << data->rigids[iR].markersID[i];
            if (data->rigids[iR].markersSize)
                sout << ",size=" << data->rigids[iR].markersSize[i];
        }
        sout << ENDL;
    }

    sout << "\tSkeleton Count : " << data->nSkeletons << ENDL;
    for (int iS = 0; iS < data->nSkeletons; ++iS)
    {
        sout << ENDL;
        sout << "\t\tID : " << data->skeletons[iS].ID << ENDL;

        sout << "\t\tRigid Body Count : " << data->skeletons[iS].nRigids << ENDL;
        for (int iR = 0; iR < data->skeletons[iS].nRigids; ++iR)
        {
            sout << ENDL;
            sout << "\t\t\tID : " << data->skeletons[iS].rigids[iR].ID << ENDL;
            sout << "\t\t\tpos : " << data->skeletons[iS].rigids[iR].pos << ENDL;
            if (data->skeletons[iS].rigids[iR].rot[0] != 0.0f
                || data->skeletons[iS].rigids[iR].rot[1] != 0.0f
                || data->skeletons[iS].rigids[iR].rot[2] != 0.0f
                || data->skeletons[iS].rigids[iR].rot[3] != 0.0f)
                sout << "\t\t\trot : " << data->skeletons[iS].rigids[iR].rot << ENDL;
            sout << "\t\t\tMarkers (" << data->skeletons[iS].rigids[iR].nMarkers << ") :";
            for (int i = 0; i < data->skeletons[iS].rigids[iR].nMarkers; ++i)
            {
                sout << " [" << data->skeletons[iS].rigids[iR].markersPos[i] << "]";
                if (data->skeletons[iS].rigids[iR].markersID)
                    sout << ",id=" << data->skeletons[iS].rigids[iR].markersID[i];
                if (data->skeletons[iS].rigids[iR].markersSize)
                    sout << ",size=" << data->skeletons[iS].rigids[iR].markersSize[i];
            }
            sout << ENDL;
        }
    }
    sout << "latency : " << data->latency << ENDL;
#undef ENDL
    sout << sendl;
}

void OptiTrackNatNetClient::processModelDef(const ModelDef* data)
{

}

void OptiTrackNatNetClient::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event))
    {
        update();
    }
}

void OptiTrackNatNetClient::update()
{
    get_io_service().poll();
}


SOFA_DECL_CLASS(OptiTrackNatNetClient)

int OptiTrackNatNetClientClass = sofa::core::RegisterObject("Network client to receive tracked points and rigids from NaturalPoint OptiTrack devices using NatNet protocol")
        .add< OptiTrackNatNetClient >()
        ;

} // namespace SofaOptiTrackNatNet
