/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "OptiTrackNatNetClient.h"
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/core/visual/VisualParams.h>
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
    , scale(initData(&scale, (double)1, "scale", "Scale factor to apply to coordinates (using the global frame as fixed point)"))
    , trackedMarkers(initData(&trackedMarkers,"trackedMarkers", "Position of received known markers"))
    , otherMarkers(initData(&otherMarkers,"otherMarkers", "Position of received unknown markers"))
    , natNetReceivers(initLink("natNetReceivers", "List of data receiver components"))
    , drawTrackedMarkersSize(initData(&drawTrackedMarkersSize, 0.01f, "drawTrackedMarkersSize", "Size of displayed markers"))
    , drawTrackedMarkersColor(initData(&drawTrackedMarkersColor, sofa::type::RGBAColor::white(), "drawTrackedMarkersColor", "Color of displayed markers"))
    , drawOtherMarkersSize(initData(&drawOtherMarkersSize, 0.01f, "drawOtherMarkersSize", "Size of displayed unknown markers"))
    , drawOtherMarkersColor(initData(&drawOtherMarkersColor, sofa::type::RGBAColor::magenta(), "drawOtherMarkersColor", "Color of displayed unknown markers"))
    , command_socket(NULL)
    , data_socket(NULL)
    , recv_command_packet(new sPacket)
    , recv_data_packet(new sPacket)
    , serverInfoReceived(false)
    , modelInfoReceived(false)
{
    this->f_listening.setValue(true);
    //this->f_printLog.setValue(true);
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
    msg_info() << "Connecting to " << serverName.getValue();

    bool connected = connect();
    if (!connected)
        msg_error() << "Connection failed";

}

bool OptiTrackNatNetClient::connect()
{
    if (command_socket) delete command_socket; command_socket = NULL;
    if (data_socket) delete data_socket; data_socket = NULL;

    boost::system::error_code ec;

    {
        msg_info() << "Resolving " <<  serverName.getValue();
        udp::resolver::query query(udp::v4(), serverName.getValue(), "0");
        udp::resolver::iterator result = get_resolver().resolve(query, ec);
        if (ec)
        {
            msg_error() << ec.category().name() << " ERROR while resolving " << serverName.getValue() << " : " << ec.message();
            return false;
        }

        server_endpoint = *result;
        server_endpoint.port(PORT_COMMAND);
        msg_info() << "Resolved " <<  serverName.getValue() << " to " << server_endpoint;
    }

    udp::endpoint client_endpoint(udp::v4(), PORT_DATA);
    if (!clientName.getValue().empty())
    {
        msg_info() << "Resolving " <<  clientName.getValue();
        udp::resolver::query query(udp::v4(), clientName.getValue(), "0");
        udp::resolver::iterator result = get_resolver().resolve(query, ec);
        if (ec)
        {
            msg_error() << ec.category().name() << " ERROR while resolving " << clientName.getValue() << " : " << ec.message();
            return false;
        }

        client_endpoint = *result;
        client_endpoint.port(PORT_DATA);
        msg_info() << "Resolved " <<  clientName.getValue() << " to " << client_endpoint;
    }

    msg_info() << "Opening data socket on " <<  client_endpoint;
    data_socket = new udp::socket(get_io_service());
    if (data_socket->open(udp::v4(), ec))
    {
        msg_error() << ec.category().name() << " ERROR while opening data socket : " << ec.message();
        return false;
    }

    data_socket->set_option(udp::socket::reuse_address(true));
    if (data_socket->bind(client_endpoint, ec))
    {
        msg_error() << ec.category().name() << " ERROR while binding data socket : " << ec.message();
        return false;
    }

    // Join the multicast group.
    if (!clientName.getValue().empty())
        data_socket->set_option(boost::asio::ip::multicast::join_group(boost::asio::ip::address::from_string(MULTICAST_ADDRESS).to_v4(), client_endpoint.address().to_v4()));
    else
        data_socket->set_option(boost::asio::ip::multicast::join_group(boost::asio::ip::address::from_string(MULTICAST_ADDRESS)));

    msg_info() << "Data socket ready";
    start_data_receive();

    msg_info() << "Opening command socket";

    command_socket = new udp::socket(get_io_service());
    if (command_socket->open(udp::v4(), ec))
    {
        msg_error() << ec.category().name() << " ERROR while opening command socket : " << ec.message();
        return false;
    }

    if (!clientName.getValue().empty())
    {
        client_endpoint.port(0);
        if (command_socket->bind(client_endpoint, ec))
        {
            msg_error() << ec.category().name() << " ERROR while binding command socket : " << ec.message();
            return false;
        }
    }

    msg_info() << "Command socket ready";
    start_command_receive();

    //boost::shared_pointer<sPacket> helloMsg = new sPacket;
    //helloMsg.iMessage = NAT_PING;
    //helloMsg.nDataBytes = 0;

    msg_info() << "Sending hello message...";

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
        msg_error() << ec.category().name() << " ERROR while receiving command from " << recv_command_endpoint;
    }
    else
    {
        msg_info() << "Received " << bytes_transferred << "b command from " << recv_command_endpoint;
        sPacket& PacketIn = *recv_command_packet;
        switch (PacketIn.iMessage)
        {
        case NAT_MODELDEF:
        {
            msg_info() << "Received MODELDEF";
            if (serverInfoReceived)
            {
                decodeModelDef(PacketIn);
            }
            else if (!this->serverName.isSet())
            {
                server_endpoint = recv_command_endpoint;
                server_endpoint.port(PORT_COMMAND);
                msg_error() << "Requesting server info to " << server_endpoint;
                boost::array<unsigned short, 2> helloMsg;
                helloMsg[0] = NAT_PING; helloMsg[1] = 0;
                command_socket->send_to(boost::asio::buffer(helloMsg), server_endpoint);
            }
            break;
        }
        case NAT_FRAMEOFDATA:
        {
            msg_info() << "Received FRAMEOFDATA";
            if (serverInfoReceived)
            {
                decodeFrame(PacketIn);
            }
            else if (!this->serverName.isSet())
            {
                server_endpoint = recv_command_endpoint;
                server_endpoint.port(PORT_COMMAND);
                msg_error() << "Requesting server info to " << server_endpoint;
                boost::array<unsigned short, 2> helloMsg;
                helloMsg[0] = NAT_PING; helloMsg[1] = 0;
                command_socket->send_to(boost::asio::buffer(helloMsg), server_endpoint);
            }
            break;
        }
        case NAT_PINGRESPONSE:
        {
            serverInfoReceived = true;
            serverString = PacketIn.Data.Sender.szName;
            for(int i=0; i<4; i++)
            {
                natNetVersion[i] = PacketIn.Data.Sender.NatNetVersion[i];
                serverVersion[i] = PacketIn.Data.Sender.Version[i];
            }
            msg_error() << "Connected to server \"" << serverString << "\" v" << (int)serverVersion[0];
            if (serverVersion[1] || serverVersion[2] || serverVersion[3])
                msg_error() << "." << (int)serverVersion[1];
            if (serverVersion[2] || serverVersion[3])
                msg_error() << "." << (int)serverVersion[2];
            if (serverVersion[3])
                msg_error() << "." << (int)serverVersion[3];
            msg_error() << " protocol v" << (int)natNetVersion[0];
            if (natNetVersion[1] || natNetVersion[2] || natNetVersion[3])
                msg_error() << "." << (int)natNetVersion[1];
            if (natNetVersion[2] || natNetVersion[3])
                msg_error() << "." << (int)natNetVersion[2];
            if (natNetVersion[3])
                msg_error() << "." << (int)natNetVersion[3];
            msg_error();
            // request scene info
            boost::array<unsigned short, 2> reqMsg;
            reqMsg[0] = NAT_REQUEST_MODELDEF; reqMsg[1] = 0;
            command_socket->send_to(boost::asio::buffer(reqMsg), server_endpoint);
            break;
        }
        case NAT_RESPONSE:
        {
            msg_info() << "Received response : " << PacketIn.Data.szData;
            break;
        }
        case NAT_UNRECOGNIZED_REQUEST:
        {
            msg_error() << "Received 'unrecognized request'";
            break;
        }
        case NAT_MESSAGESTRING:
        {
            msg_info() << "Received message: " << PacketIn.Data.szData;
            break;
        }
        default:
        {
            msg_error() << "Received unrecognized command packet type: " << PacketIn.iMessage;
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
        msg_error() << ec.category().name() << " ERROR while receiving data from " << recv_data_endpoint;
    }
    else
    {
        msg_info() << "Received " << bytes_transferred << "b data from " << recv_data_endpoint;
        sPacket& PacketIn = *recv_data_packet;
        switch (PacketIn.iMessage)
        {
        case NAT_MODELDEF:
        {
            msg_info() << "Received MODELDEF";
            if (serverInfoReceived)
            {
                decodeModelDef(PacketIn);
            }
            else if (!this->serverName.isSet())
            {
                server_endpoint = recv_data_endpoint;
                server_endpoint.port(PORT_COMMAND);
                msg_error() << "Requesting server info to " << server_endpoint;
                boost::array<unsigned short, 2> helloMsg;
                helloMsg[0] = NAT_PING; helloMsg[1] = 0;
                command_socket->send_to(boost::asio::buffer(helloMsg), server_endpoint);
            }
            break;
        }
        case NAT_FRAMEOFDATA:
        {
            msg_info() << "Received FRAMEOFDATA";
            if (serverInfoReceived)
            {
                decodeFrame(PacketIn);
            }
            else if (!this->serverName.isSet())
            {
                server_endpoint = recv_data_endpoint;
                server_endpoint.port(PORT_COMMAND);
                msg_error() << "Requesting server info to " << server_endpoint;
                boost::array<unsigned short, 2> helloMsg;
                helloMsg[0] = NAT_PING; helloMsg[1] = 0;
                command_socket->send_to(boost::asio::buffer(helloMsg), server_endpoint);
            }
            break;
        }
        default:
        {
            msg_error() << "Received unrecognized data packet type: " << PacketIn.iMessage;
            break;
        }
        }
    }
    start_data_receive();
}

template<class T>
static void memread(T& dest, const unsigned char*& ptr, const unsigned char*& end, const char* fieldName=NULL)
{
    if (ptr + sizeof(T) <= end)
    {
        dest = *(const T*)ptr;
        ptr += sizeof(T);
    }
    else
    {
        memset(&dest,0,sizeof(T));
        if (ptr != end)
        {
            msg_error("OptiTrackNatNet") << "OptiTrackNatNet decode ERROR: end of message reached";
            if (fieldName) msg_error("OptiTrackNatNet") << " while reading " << fieldName;
            ptr = end;
        }
    }
}

static void memread(const char*& dest, const unsigned char*& ptr, const unsigned char*& end, const char* fieldName=NULL)
{
    unsigned int len = 0;
    while (ptr+len < end && ptr[len])
        ++len;
    if (end-ptr > len)
    {
        dest = (const char*) ptr;
        ptr += len+1;
    }
    else
    {
        dest = "";
        if (ptr != end)
        {
            msg_error("OptiTrackNatNet") << "decode ERROR: end of message reached";
            if (fieldName) msg_error("OptiTrackNatNet") << " while reading string " << fieldName;
            ptr = end;
        }
    }
}

template<class T>
static void memread(const T*& dest, int n, const unsigned char*& ptr, const unsigned char*& end, const char* fieldName=NULL)
{
    if (n <= 0)
        dest = NULL;
    else if (ptr + n*sizeof(T) <= end)
    {
        dest = (const T*)ptr;
        ptr += n*sizeof(T);
    }
    else
    {
        dest = NULL;
        if (ptr != end)
        {
            msg_error("OptiTrackNatNet") << "OptiTrackNatNet decode ERROR: end of message reached";
            if (fieldName) msg_error("OptiTrackNatNet") << " while reading " << n << " values for array " << fieldName;
            ptr = end;
        }
    }
}

void OptiTrackNatNetClient::decodeFrame(const sPacket& data)
{
    const int major = natNetVersion[0];
    const int minor = natNetVersion[1];

    FrameData frame;

    int nTrackedMarkers = 0;
    int nOtherMarkers = 0;

    const unsigned char *ptr = data.Data.cData;
    const unsigned char *end = ptr + data.nDataBytes;

    memread(frame.frameNumber,ptr,end,"frameNumber");

    memread(frame.nPointClouds,ptr,end,"nPointClouds");
    if (frame.nPointClouds <= 0)
        frame.pointClouds = NULL;
    else
    {
        frame.pointClouds = new PointCloudData[frame.nPointClouds];
        for (int iP = 0; iP < frame.nPointClouds; ++iP)
        {
            PointCloudData& pdata = frame.pointClouds[iP];
            memread(pdata.name,ptr,end,"pointCloud.name");
            memread(pdata.nMarkers,ptr,end,"pointCloud.nMarkers");
            nTrackedMarkers += pdata.nMarkers;
            memread(pdata.markersPos, pdata.nMarkers, ptr, end, "pointCloud.markersPos");
        }
    }
    memread(frame.nOtherMarkers,ptr,end,"nOtherMarkers");
    nOtherMarkers += frame.nOtherMarkers;
    memread(frame.otherMarkersPos, frame.nOtherMarkers, ptr,end,"otherMarkersPos");

    memread(frame.nRigids,ptr,end,"nRigids");
    if (frame.nRigids <= 0)
        frame.rigids = NULL;
    else
    {
        frame.rigids = new RigidData[frame.nRigids];
        for (int iR = 0; iR < frame.nRigids; ++iR)
        {
            RigidData& rdata = frame.rigids[iR];
            memread(rdata.ID,ptr,end,"rigid.ID");
            memread(rdata.pos,ptr,end,"rigid.pos");
            memread(rdata.rot,ptr,end,"rigid.rot");
            memread(rdata.nMarkers, ptr,end,"rigid.nMarkers");
            nTrackedMarkers += rdata.nMarkers;
            memread(rdata.markersPos, rdata.nMarkers, ptr,end,"rigid.markersPos");
            if (major < 2)
            {
                rdata.markersID = NULL;
                rdata.markersSize = NULL;
                rdata.meanError = -1.0f;
            }
            else
            {
                memread(rdata.markersID, rdata.nMarkers, ptr,end,"rigid.markersID");
                memread(rdata.markersSize, rdata.nMarkers, ptr,end,"rigid.markersSize");
                memread(rdata.meanError, ptr,end,"rigid.meanError");
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
        memread(frame.nSkeletons,ptr,end,"nSkeletons");
        if (frame.nSkeletons <= 0)
            frame.skeletons = NULL;
        else
        {
            frame.skeletons = new SkeletonData[frame.nSkeletons];
            for (int iS = 0; iS < frame.nSkeletons; ++iS)
            {
                SkeletonData& sdata = frame.skeletons[iS];
                memread(sdata.ID,ptr,end,"skeleton.ID");
                memread(sdata.nRigids, ptr,end,"skeleton.nRigids");
                if (sdata.nRigids <= 0)
                    sdata.rigids = NULL;
                else
                {
                    sdata.rigids = new RigidData[sdata.nRigids];
                    for (int iR = 0; iR < sdata.nRigids; ++iR)
                    {
                        RigidData& rdata = sdata.rigids[iR];
                        memread(rdata.ID,ptr,end,"rigid.ID");
                        memread(rdata.pos,ptr,end,"rigid.pos");
                        memread(rdata.rot,ptr,end,"rigid.rot");
                        memread(rdata.nMarkers, ptr,end,"rigid.nMarkers");
                        nTrackedMarkers += rdata.nMarkers;
                        memread(rdata.markersPos, rdata.nMarkers, ptr,end,"rigid.markersPos");
                        memread(rdata.markersID, rdata.nMarkers, ptr,end,"rigid.markersID");
                        memread(rdata.markersSize, rdata.nMarkers, ptr,end,"rigid.markersSize");
                        memread(rdata.meanError, ptr,end,"rigid.meanError");
                    }
                }
            }
        }
    }
    memread(frame.latency, ptr,end,"latency");
    if (ptr != end)
    {
//        msg_error() << "decodeFrame: extra " << end-ptr << " bytes at end of message";
    }
    // Copy markers to stored Data
    {
        sofa::helper::WriteAccessor<sofa::core::objectmodel::Data<sofa::type::vector<sofa::type::Vec3f> > > markers = this->otherMarkers;
        markers.resize(nOtherMarkers);
        int m0 = 0;
        for (int i = 0; i < frame.nOtherMarkers; ++i)
            markers[m0+i] = frame.otherMarkersPos[i];
        frame.otherMarkersPos = &(markers[m0]);
        m0 += frame.nOtherMarkers;
    }

    {
        sofa::helper::WriteAccessor<sofa::core::objectmodel::Data<sofa::type::vector<sofa::type::Vec3f> > > markers = this->trackedMarkers;
        markers.resize(nTrackedMarkers);
        int m0 = 0;

        for (int iP = 0; iP < frame.nPointClouds; ++iP)
        {
            for (int i = 0; i < frame.pointClouds[iP].nMarkers; ++i)
                markers[m0+i] = frame.pointClouds[iP].markersPos[i];
            frame.pointClouds[iP].markersPos = &(markers[m0]);
            m0 += frame.pointClouds[iP].nMarkers;
        }
        for (int iR = 0; iR < frame.nRigids; ++iR)
        {
            for (int i = 0; i < frame.rigids[iR].nMarkers; ++i)
                markers[m0+i] = frame.rigids[iR].markersPos[i];
            frame.rigids[iR].markersPos = &(markers[m0]);
            m0 += frame.rigids[iR].nMarkers;
        }

        for (int iS = 0; iS < frame.nSkeletons; ++iS)
        {
            for (int iR = 0; iR < frame.skeletons[iS].nRigids; ++iR)
            {
                for (int i = 0; i < frame.skeletons[iS].rigids[iR].nMarkers; ++i)
                    markers[m0+i] = frame.skeletons[iS].rigids[iR].markersPos[i];
                frame.skeletons[iS].rigids[iR].markersPos = &(markers[m0]);
                m0 += frame.skeletons[iS].rigids[iR].nMarkers;
            }
        }
    }

    // Apply scale factor
    if (this->scale.isSet())
    {
        const double scale = this->scale.getValue();
        {
            sofa::helper::WriteAccessor<sofa::core::objectmodel::Data<sofa::type::vector<sofa::type::Vec3f> > > markers = this->trackedMarkers;
            for (unsigned int i=0; i<markers.size(); ++i)
                markers[i] *= scale;
        }
        {
            sofa::helper::WriteAccessor<sofa::core::objectmodel::Data<sofa::type::vector<sofa::type::Vec3f> > > markers = this->otherMarkers;
            for (unsigned int i=0; i<markers.size(); ++i)
                markers[i] *= scale;
        }
        for (int iR = 0; iR < frame.nRigids; ++iR)
            frame.rigids[iR].pos *= scale;
        for (int iS = 0; iS < frame.nSkeletons; ++iS)
            for (int iR = 0; iR < frame.skeletons[iS].nRigids; ++iR)
                frame.skeletons[iS].rigids[iR].pos *= scale;
    }

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
    const int major = natNetVersion[0];
    //const int minor = natNetVersion[1];

    ModelDef model;
    sofa::type::vector<PointCloudDef> pointClouds;
    sofa::type::vector<RigidDef> rigids;
    sofa::type::vector<SkeletonDef> skeletons;

    const unsigned char *ptr = data.Data.cData;
    const unsigned char *end = ptr + data.nDataBytes;

    int nDatasets = 0;
    memread(nDatasets,ptr,end,"nDatasets");

    for(int i=0; i < nDatasets; i++)
    {
        int type = 0;
        memread(type,ptr,end,"type");
        switch(type)
        {
        case 0: // point cloud
        {
            PointCloudDef pdef;
            memread(pdef.name,ptr,end,"name");
            memread(pdef.nMarkers,ptr,end,"nMarkers");
            if (pdef.nMarkers <= 0)
                pdef.markers = NULL;
            else
            {
                pdef.markers = new PointCloudDef::Marker[pdef.nMarkers];
                for (int j=0; j<pdef.nMarkers; ++j)
                {
                    memread(pdef.markers[j].name,ptr,end,"markers.name");
                }
            }
            pointClouds.push_back(pdef);
            break;
        }
        case 1: // rigid
        {
            RigidDef rdef;
            if(major >= 2)
                memread(rdef.name,ptr,end,"rigid.name");
            else
                rdef.name = NULL;
            memread(rdef.ID,ptr,end,"rigid.ID");
            memread(rdef.parentID,ptr,end,"rigid.parentID");
            memread(rdef.offset,ptr,end,"rigid.offset");
            rigids.push_back(rdef);
            break;
        }
        case 2: // skeleton
        {
            SkeletonDef sdef;
            memread(sdef.name,ptr,end,"skeleton.name");
            memread(sdef.nRigids,ptr,end,"skeleton.nRigids");
            if (sdef.nRigids <= 0)
                sdef.rigids = NULL;
            else
            {
                sdef.rigids = new RigidDef[sdef.nRigids];
                for (int j=0; j<sdef.nRigids; ++j)
                {
                    RigidDef& rdef = sdef.rigids[j];
                    memread(rdef.name,ptr,end,"skeleton.rigid.name");
                    memread(rdef.ID,ptr,end,"skeleton.rigid.ID");
                    memread(rdef.parentID,ptr,end,"skeleton.rigid.parentID");
                    memread(rdef.offset,ptr,end,"skeleton.rigid.offset");
                }
            }
            skeletons.push_back(sdef);
            break;
        }
        default:
        {
            msg_error() << "decodeModelDef: unknown type " << type;
        }
        }
    }

    model.nPointClouds = pointClouds.size();
    model.pointClouds = (pointClouds.size() > 0) ? &(pointClouds[0]) : NULL;
    model.nRigids = rigids.size();
    model.rigids = (rigids.size() > 0) ? &(rigids[0]) : NULL;
    model.nSkeletons = skeletons.size();
    model.skeletons = (skeletons.size() > 0) ? &(skeletons[0]) : NULL;

    // Apply scale factor
    if (this->scale.isSet())
    {
        const double scale = this->scale.getValue();
        for (int iR = 0; iR < model.nRigids; ++iR)
        {
            model.rigids[iR].offset *= scale;
        }
        for (int iS = 0; iS < model.nSkeletons; ++iS)
        {
            for (int iR = 0; iR < model.skeletons[iS].nRigids; ++iR)
            {
                model.skeletons[iS].rigids[iR].offset *= scale;
            }
        }
    }

    processModelDef(&model);

    for (int i=0; i<model.nPointClouds; ++i)
    {
        if (model.pointClouds[i].markers)
            delete[] model.pointClouds[i].markers;
    }

    for (int i=0; i<model.nSkeletons; ++i)
    {
        if (model.skeletons[i].rigids)
            delete[] model.skeletons[i].rigids;
    }
}

void OptiTrackNatNetClient::processFrame(const FrameData* data)
{
    if (this->f_printLog.getValue())
    {
        msg_info() << "Frame # : " << data->frameNumber << "\n";
        msg_info() << "\tPoint Cloud Count : " << data->nPointClouds << "\n";
        for (int iP = 0; iP < data->nPointClouds; ++iP)
        {
            msg_info() << "\n";
            if (data->pointClouds[iP].name)
                msg_info() << "\t\tModel Name : " << data->pointClouds[iP].name << "\n";
            msg_info() << "\t\tMarkers (" << data->pointClouds[iP].nMarkers << ") :";
            for (int i = 0; i < data->pointClouds[iP].nMarkers; ++i)
                msg_info() << " [" << data->pointClouds[iP].markersPos[i] << "]";
            msg_info() << "\n";
        }

        msg_info() << "\tUnidentified Markers (" << data->nOtherMarkers << ") :";
        for (int i = 0; i < data->nOtherMarkers; ++i)
            msg_info() << " [" << data->otherMarkersPos[i] << "]";
        msg_info() << "\n";

        msg_info() << "\tRigid Body Count : " << data->nRigids << "\n";
        for (int iR = 0; iR < data->nRigids; ++iR)
        {
            msg_info() << "\n";
            msg_info() << "\t\tID : " << data->rigids[iR].ID << "\n";
            msg_info() << "\t\tpos : " << data->rigids[iR].pos << "\n";
            if (data->rigids[iR].rot[0] != 0.0f
                || data->rigids[iR].rot[1] != 0.0f
                || data->rigids[iR].rot[2] != 0.0f
                || data->rigids[iR].rot[3] != 0.0f)
                msg_info() << "\t\trot : " << data->rigids[iR].rot << "\n";
            msg_info() << "\t\tMarkers (" << data->rigids[iR].nMarkers << ") :";
            for (int i = 0; i < data->rigids[iR].nMarkers; ++i)
            {
                msg_info() << " [" << data->rigids[iR].markersPos[i] << "]";
                if (data->rigids[iR].markersID)
                    msg_info() << ",id=" << data->rigids[iR].markersID[i];
                if (data->rigids[iR].markersSize)
                    msg_info() << ",size=" << data->rigids[iR].markersSize[i];
            }
            msg_info() << "\n";
        }

        msg_info() << "\tSkeleton Count : " << data->nSkeletons << "\n";
        for (int iS = 0; iS < data->nSkeletons; ++iS)
        {
            msg_info() << "\n";
            msg_info() << "\t\tID : " << data->skeletons[iS].ID << "\n";

            msg_info() << "\t\tRigid Body Count : " << data->skeletons[iS].nRigids << "\n";
            for (int iR = 0; iR < data->skeletons[iS].nRigids; ++iR)
            {
                msg_info() << "\n";
                msg_info() << "\t\t\tID : " << data->skeletons[iS].rigids[iR].ID << "\n";
                msg_info() << "\t\t\tpos : " << data->skeletons[iS].rigids[iR].pos << "\n";
                if (data->skeletons[iS].rigids[iR].rot[0] != 0.0f
                    || data->skeletons[iS].rigids[iR].rot[1] != 0.0f
                    || data->skeletons[iS].rigids[iR].rot[2] != 0.0f
                    || data->skeletons[iS].rigids[iR].rot[3] != 0.0f)
                    msg_info() << "\t\t\trot : " << data->skeletons[iS].rigids[iR].rot << "\n";
                msg_info() << "\t\t\tMarkers (" << data->skeletons[iS].rigids[iR].nMarkers << ") :";
                for (int i = 0; i < data->skeletons[iS].rigids[iR].nMarkers; ++i)
                {
                    msg_info() << " [" << data->skeletons[iS].rigids[iR].markersPos[i] << "]";
                    if (data->skeletons[iS].rigids[iR].markersID)
                        msg_info() << ",id=" << data->skeletons[iS].rigids[iR].markersID[i];
                    if (data->skeletons[iS].rigids[iR].markersSize)
                        msg_info() << ",size=" << data->skeletons[iS].rigids[iR].markersSize[i];
                }
                msg_info() << "\n";
            }
        }
        msg_info() << "latency : " << data->latency << "\n";
    }

    for (unsigned int i=0,n=natNetReceivers.size(); i<n; ++i)
        if (natNetReceivers[i]) natNetReceivers[i]->processFrame(data);
}

void OptiTrackNatNetClient::processModelDef(const ModelDef* data)
{
    if (this->f_printLog.getValue())
    {
        msg_info() << "ModelDef : " << "\n";
        msg_info() << "\tPoint Cloud Count : " << data->nPointClouds << "\n";
        for (int iP = 0; iP < data->nPointClouds; ++iP)
        {
            msg_info() << "\n";
            if (data->pointClouds[iP].name)
                msg_info() << "\t\tModel Name : " << data->pointClouds[iP].name << "\n";
            msg_info() << "\t\tMarkers (" << data->pointClouds[iP].nMarkers << ") :";
            for (int i = 0; i < data->pointClouds[iP].nMarkers; ++i)
                msg_info() << " " << data->pointClouds[iP].name;
            msg_info() << "\n";
        }

        msg_info() << "\tRigid Body Count : " << data->nRigids << "\n";
        for (int iR = 0; iR < data->nRigids; ++iR)
        {
            msg_info() << "\n";
            if (data->rigids[iR].name)
                msg_info() << "\t\tname : " << data->rigids[iR].name << "\n";
            msg_info() << "\t\tID : " << data->rigids[iR].ID << "\n";
            msg_info() << "\t\tparentID : " << data->rigids[iR].parentID << "\n";
            if (data->rigids[iR].offset[0] != 0.0f
                || data->rigids[iR].offset[1] != 0.0f
                || data->rigids[iR].offset[2] != 0.0f)
                msg_info() << "\t\toffset : " << data->rigids[iR].offset << "\n";
            msg_info() << "\n";
        }

        msg_info() << "\tSkeleton Count : " << data->nSkeletons << "\n";
        for (int iS = 0; iS < data->nSkeletons; ++iS)
        {
            msg_info() << "\n";
            if (data->skeletons[iS].name)
                msg_info() << "\t\tname : " << data->skeletons[iS].name << "\n";
            msg_info() << "\t\tID : " << data->skeletons[iS].ID << "\n";

            msg_info() << "\t\tRigid Body Count : " << data->skeletons[iS].nRigids << "\n";
            for (int iR = 0; iR < data->skeletons[iS].nRigids; ++iR)
            {
                msg_info() << "\n";
                if (data->skeletons[iS].rigids[iR].name)
                    msg_info() << "\t\t\tname : " << data->skeletons[iS].rigids[iR].name << "\n";
                msg_info() << "\t\t\tID : " << data->skeletons[iS].rigids[iR].ID << "\n";
                msg_info() << "\t\t\tparentID : " << data->skeletons[iS].rigids[iR].parentID << "\n";
                if (data->skeletons[iS].rigids[iR].offset[0] != 0.0f
                    || data->skeletons[iS].rigids[iR].offset[1] != 0.0f
                    || data->skeletons[iS].rigids[iR].offset[2] != 0.0f)
                    msg_info() << "\t\t\toffset : " << data->skeletons[iS].rigids[iR].offset << "\n";
                msg_info() << "\n";
            }
        }
    }

    for (unsigned int i=0,n=natNetReceivers.size(); i<n; ++i)
        if (natNetReceivers[i]) natNetReceivers[i]->processModelDef(data);
}

void OptiTrackNatNetClient::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
    {
        update();
    }
}

void OptiTrackNatNetClient::update()
{
    get_io_service().poll();
}

void OptiTrackNatNetClient::draw(const sofa::core::visual::VisualParams* vparams)
{
    const float trackedMarkersSize = drawTrackedMarkersSize.getValue();
    if (trackedMarkersSize > 0)
    {
        //sofa::type::vector<sofa::type::Vector3> markers = trackedMarkers.getValue();
        const sofa::type::vector<sofa::type::Vec3f>& val = trackedMarkers.getValue();
        sofa::type::vector<sofa::type::Vector3> markers(val.size());
        for (unsigned int i=0; i<val.size(); ++i) markers[i] = val[i];
        vparams->drawTool()->drawSpheres(markers, trackedMarkersSize, drawTrackedMarkersColor.getValue());
    }
    const float otherMarkersSize = drawOtherMarkersSize.getValue();
    if (otherMarkersSize > 0)
    {
        //sofa::type::vector<sofa::type::Vector3> markers = otherMarkers.getValue();
        const sofa::type::vector<sofa::type::Vec3f>& val = otherMarkers.getValue();
        sofa::type::vector<sofa::type::Vector3> markers(val.size());
        for (unsigned int i=0; i<val.size(); ++i) markers[i] = val[i];
        vparams->drawTool()->drawSpheres(markers, otherMarkersSize, drawOtherMarkersColor.getValue());
    }
}

int OptiTrackNatNetClientClass = sofa::core::RegisterObject("Network client to receive tracked points and rigids from NaturalPoint OptiTrack devices using NatNet protocol")
        .add< OptiTrackNatNetClient >()
        ;

} // namespace SofaOptiTrackNatNet
