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
#ifndef OPTITRACKNATNETDEVICE_H
#define OPTITRACKNATNETDEVICE_H

#include <sofa/config.h>

#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <iostream>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include "OptiTrackNatNetClient.h"

namespace SofaOptiTrackNatNet
{

/// internal message buffer class, as defined in NatNet SDK
struct sPacket;

/// decoded definition of tracked objects
struct ModelDef;

/// decoded frame of tracked data
struct FrameData;

class OptiTrackNatNetDevice : public OptiTrackNatNetDataReceiver
{
public:
    SOFA_CLASS(OptiTrackNatNetDevice, OptiTrackNatNetDataReceiver);

    typedef sofa::defaulttype::Rigid3Types DataTypes;
    typedef DataTypes::Real Real;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::CPos CPos;
    typedef DataTypes::CRot CRot;
    typedef DataTypes::Deriv Deriv;
    typedef DataTypes::VecCoord VecCoord;
    typedef DataTypes::VecDeriv VecDeriv;

protected:

    virtual void update();

public:

    OptiTrackNatNetDevice();
    virtual ~OptiTrackNatNetDevice();

    virtual void init();
    virtual void reinit();
    virtual void draw(const sofa::core::visual::VisualParams* vparams);

    virtual void processModelDef(const ModelDef* data);
    virtual void processFrame(const FrameData* data);

    virtual void onBeginAnimationStep(const double /*dt*/);
    virtual void onKeyPressedEvent(sofa::core::objectmodel::KeypressedEvent* ev);

    sofa::core::objectmodel::Data<std::string> trackableName;
    sofa::core::objectmodel::Data<int> trackableID;
    sofa::core::objectmodel::Data<bool> setRestShape;
    sofa::core::objectmodel::Data<bool> applyMappings;
    sofa::core::objectmodel::Data<bool> controlNode;
    sofa::core::objectmodel::Data<bool> isGlobalFrame;
    sofa::core::objectmodel::SingleLink<OptiTrackNatNetDevice,OptiTrackNatNetClient,sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK> natNetClient;
    sofa::core::objectmodel::SingleLink<OptiTrackNatNetDevice,sofa::core::behavior::MechanicalState<DataTypes>,sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK> mstate;
    sofa::core::objectmodel::DataFileName inMarkersMeshFile;
    sofa::core::objectmodel::DataFileName simMarkersMeshFile;
    sofa::core::objectmodel::Data<sofa::helper::vector<CPos> > inLocalMarkers;
    sofa::core::objectmodel::Data<sofa::helper::vector<CPos> > simLocalMarkers;
    sofa::core::objectmodel::Data<bool> tracked;
    sofa::core::objectmodel::Data<Coord> trackedFrame;
    sofa::core::objectmodel::Data<Coord> frame;
    sofa::core::objectmodel::Data<CPos> position;
    sofa::core::objectmodel::Data<CRot> orientation;
    sofa::core::objectmodel::Data<Coord> simGlobalFrame;
    sofa::core::objectmodel::Data<Coord> inGlobalFrame;
    sofa::core::objectmodel::Data<Coord> simLocalFrame;
    sofa::core::objectmodel::Data<Coord> inLocalFrame;
    sofa::core::objectmodel::Data<sofa::helper::vector<CPos> > markers;
    sofa::core::objectmodel::Data<sofa::helper::vector<int> > markersID;
    sofa::core::objectmodel::Data<sofa::helper::vector<Real> > markersSize;

    sofa::core::objectmodel::Data<sofa::helper::fixed_array<int,2> > distanceMarkersID;
    sofa::core::objectmodel::Data<sofa::helper::vector<CPos> > distanceMarkersPos;
    sofa::core::objectmodel::Data<Real> openDistance;
    sofa::core::objectmodel::Data<Real> closedDistance;
    sofa::core::objectmodel::Data<Real> distance;
    sofa::core::objectmodel::Data<Real> distanceFactor;
    sofa::core::objectmodel::Data<bool> open;
    sofa::core::objectmodel::Data<bool> closed;

    sofa::core::objectmodel::Data<CPos> jointCenter;
    sofa::core::objectmodel::Data<CPos> jointAxis;
    sofa::core::objectmodel::Data<Real> jointOpenAngle;
    sofa::core::objectmodel::Data<Real> jointClosedAngle;


    sofa::core::objectmodel::Data<sofa::defaulttype::Vec3f> drawAxisSize;
    sofa::core::objectmodel::Data<float> drawMarkersSize;
    sofa::core::objectmodel::Data<float> drawMarkersIDSize;
    sofa::core::objectmodel::Data<sofa::defaulttype::Vec4f> drawMarkersColor;

protected:
    int writeInMarkersMesh;
    int readSimMarkersMesh;
    Real smoothDistance;
};


} // namespace SofaOptiTrackNatNet

#endif /* OPTITRACKNATNETDEVICE_H */
