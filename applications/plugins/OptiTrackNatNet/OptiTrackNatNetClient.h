/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef OPTITRACKNATNETCLIENT_H
#define OPTITRACKNATNETCLIENT_H

#include <sofa/config.h>

#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <iostream>

#include <sofa/core/objectmodel/BaseObject.h>
//#include <sofa/core/behavior/BaseController.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Quat.h>
#include <SofaUserInteraction/Controller.h>

namespace SofaOptiTrackNatNet
{

/// internal message buffer class, as defined in NatNet SDK
struct sPacket;

/// decoded definition of tracked objects
struct ModelDef;

/// decoded frame of tracked data
struct FrameData;

class OptiTrackNatNetDataReceiver : public sofa::component::controller::Controller
{
public:
    SOFA_ABSTRACT_CLASS(OptiTrackNatNetDataReceiver, sofa::component::controller::Controller);
protected:
    virtual ~OptiTrackNatNetDataReceiver() {}
public:
    virtual void processModelDef(const ModelDef* data) = 0;
    virtual void processFrame(const FrameData* data) = 0;
};

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
    sofa::core::objectmodel::Data<double> scale;

    sofa::core::objectmodel::Data<sofa::helper::vector<sofa::defaulttype::Vec3f> > trackedMarkers;
    sofa::core::objectmodel::Data<sofa::helper::vector<sofa::defaulttype::Vec3f> > otherMarkers;

    sofa::core::objectmodel::MultiLink<OptiTrackNatNetClient, OptiTrackNatNetDataReceiver, 0> natNetReceivers;

    OptiTrackNatNetClient();
    virtual ~OptiTrackNatNetClient();

    virtual void init();
    virtual void reinit();

    virtual void draw(const sofa::core::visual::VisualParams* vparams);

    sofa::core::objectmodel::Data<float> drawTrackedMarkersSize;
    sofa::core::objectmodel::Data<sofa::defaulttype::Vec4f> drawTrackedMarkersColor;
    sofa::core::objectmodel::Data<float> drawOtherMarkersSize;
    sofa::core::objectmodel::Data<sofa::defaulttype::Vec4f> drawOtherMarkersColor;

public:

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

    virtual void processFrame(const FrameData* data);
    virtual void processModelDef(const ModelDef* data);

    std::string serverString;
    sofa::helper::fixed_array<unsigned char,4> serverVersion; // sending app's version [major.minor.build.revision]
    sofa::helper::fixed_array<unsigned char,4> natNetVersion; // sending app's NatNet version [major.minor.build.revision]

    bool serverInfoReceived;
    bool modelInfoReceived;

    static boost::asio::io_service& get_io_service();
    static boost::asio::ip::udp::resolver& get_resolver();

};


struct PointCloudDef
{
    const char* name;
    int nMarkers;
    struct Marker
    {
        const char* name;
    };
    Marker* markers;
};

struct RigidDef
{
    const char* name;
    int ID;
    int parentID;
    sofa::defaulttype::Vec3f offset;
};

struct SkeletonDef
{
    const char* name;
    int ID;
    int nRigids;
    RigidDef* rigids;
};

struct ModelDef
{
    int nPointClouds;
    PointCloudDef* pointClouds;
    int nRigids;
    RigidDef* rigids;
    int nSkeletons;
    SkeletonDef* skeletons;
};

struct PointCloudData
{
    const char* name;
    int nMarkers;
    const sofa::defaulttype::Vec3f* markersPos;
};

struct RigidData
{
    int ID;
    sofa::defaulttype::Vec3f pos;
    sofa::defaulttype::Quatf rot;
    int nMarkers;
    const sofa::defaulttype::Vec3f* markersPos;
    const int* markersID; // optional (2.0+)
    const float* markersSize; // optional (2.0+)
    float meanError; // optional (2.0+)
};

struct SkeletonData
{
    int ID;
    int nRigids;
    RigidData* rigids;
};

struct FrameData
{
    int frameNumber;
    int nPointClouds;
    PointCloudData* pointClouds;
    int nRigids;
    RigidData* rigids;
    int nSkeletons;
    SkeletonData* skeletons;

    float latency;
    // unidentified markers
    int nOtherMarkers;
    const sofa::defaulttype::Vec3f* otherMarkersPos;
};

} // namespace SofaOptiTrackNatNet

#endif /* OPTITRACKNATNETCLIENT_H */
