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
#ifndef OPTITRACKNATNETDEVICE_H
#define OPTITRACKNATNETDEVICE_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/component/controller/Controller.h>
//#include <sofa/core/behavior/BaseController.h>
#include <sofa/defaulttype/RigidTypes.h>
#include "OptiTrackNatNetClient.h"

#include <iostream>
#include <boost/array.hpp>
#include <boost/asio.hpp>

namespace SofaOptiTrackNatNet
{

/// internal message buffer class, as defined in NatNet SDK
struct sPacket;

/// decoded definition of tracked objects
struct ModelDef;

/// decoded frame of tracked data
struct FrameData;

class OptiTrackNatNetDevice :  public sofa::component::controller::Controller, public OptiTrackNatNetDataReceiver
{
public:
    SOFA_CLASS2(OptiTrackNatNetDevice, sofa::component::controller::Controller, OptiTrackNatNetDataReceiver);

    typedef sofa::defaulttype::Rigid3dTypes DataTypes;
    typedef DataTypes::Real Real;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::CPos CPos;
    typedef DataTypes::CRot CRot;
    typedef DataTypes::Deriv Deriv;
    typedef DataTypes::VecCoord VecCoord;
    typedef DataTypes::VecDeriv VecDeriv;

protected:
//    void handleEvent(sofa::core::objectmodel::Event *);

    virtual void update();

public:

    virtual void processModelDef(const ModelDef* data);
    virtual void processFrame(const FrameData* data);

    virtual void onBeginAnimationStep(const double /*dt*/);

    sofa::core::objectmodel::Data<std::string> trackableName;
    sofa::core::objectmodel::Data<int> trackableID;
    sofa::core::objectmodel::SingleLink<OptiTrackNatNetDevice,OptiTrackNatNetClient,sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK> natNetClient;
    sofa::core::objectmodel::SingleLink<OptiTrackNatNetDevice,sofa::core::behavior::MechanicalState<DataTypes>,sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK> mstate;
    sofa::core::objectmodel::Data<bool> tracked;
    sofa::core::objectmodel::Data<Coord> frame;
    sofa::core::objectmodel::Data<CPos> position;
    sofa::core::objectmodel::Data<CRot> orientation;
    sofa::core::objectmodel::Data<Coord> trackedFrame;
    sofa::core::objectmodel::Data<Coord> simGlobalFrame;
    sofa::core::objectmodel::Data<Coord> inGlobalFrame;
    sofa::core::objectmodel::Data<Coord> simLocalFrame;
    sofa::core::objectmodel::Data<Coord> inLocalFrame;
    sofa::core::objectmodel::Data<Real> scale;

    OptiTrackNatNetDevice();
    virtual ~OptiTrackNatNetDevice();

    virtual void init();
    virtual void reinit();

public:

protected:

};


} // namespace SofaOptiTrackNatNet

#endif /* OPTITRACKNATNETDEVICE_H */
