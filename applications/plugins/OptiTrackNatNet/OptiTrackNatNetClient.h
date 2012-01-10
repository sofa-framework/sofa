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
#ifndef OPTITRACKNATNETCLIENT_H
#define OPTITRACKNATNETCLIENT_H

#include <sofa/core/objectmodel/BaseObject.h>
//#include <sofa/core/behavior/BaseController.h>

#include <iostream>
#include <boost/array.hpp>
#include <boost/asio.hpp>

// internal message buffer class, as defined in NatNet SDK
struct sPacket;

class OptiTrackNatNetClient :  public virtual sofa::core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(OptiTrackNatNetClient,sofa::core::objectmodel::BaseObject);

protected:
    bool connect();
    void handleEvent(sofa::core::objectmodel::Event *);

    virtual void update();

public:
    sofa::core::objectmodel::Data<std::string> serverName;
    sofa::core::objectmodel::Data<std::string> clientName;

    OptiTrackNatNetClient();
    virtual ~OptiTrackNatNetClient();

    virtual void init();
    virtual void reinit();

protected:
    boost::asio::ip::udp::endpoint server_endpoint;
    boost::asio::ip::udp::socket* command_socket;
    boost::asio::ip::udp::socket* data_socket;

    sPacket* recv_command_packet;
    boost::asio::ip::udp::endpoint recv_command_endpoint;

    sPacket* recv_data_packet;
    boost::asio::ip::udp::endpoint recv_data_endpoint;
    void start_command_receive();
    void handle_command_receive(const boost::system::error_code& error,
            std::size_t bytes_transferred);

    void start_data_receive();
    void handle_data_receive(const boost::system::error_code& error,
            std::size_t bytes_transferred);
    void decodeFrame(const sPacket& data);
    void decodeModelDef(const sPacket& data);

    std::string serverString;
    sofa::helper::fixed_array<unsigned char,4> serverVersion; // sending app's version [major.minor.build.revision]
    sofa::helper::fixed_array<unsigned char,4> natNetVersion; // sending app's NatNet version [major.minor.build.revision]

    static boost::asio::io_service& get_io_service();
    static boost::asio::ip::udp::resolver& get_resolver();

};

#endif /* OPTITRACKNATNETCLIENT_H */
