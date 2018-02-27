/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

    sofa::core::objectmodel::Data<std::string> trackableName; ///< NatNet trackable name
    sofa::core::objectmodel::Data<int> trackableID; ///< NatNet trackable number (ignored if trackableName is set)
    sofa::core::objectmodel::Data<bool> setRestShape; ///< True to control the rest position instead of the current position directly
    sofa::core::objectmodel::Data<bool> applyMappings; ///< True to enable applying the mappings after setting the position
    sofa::core::objectmodel::Data<bool> controlNode; ///< True to enable activating and disabling the node when this device appears and disappears
    sofa::core::objectmodel::Data<bool> isGlobalFrame; ///< True if this trackable should be considered as the global frame (i.e. all other trackables are computed relative to its position). This requires linking other trackables' "inGlobalFrame" to this "frame")
    sofa::core::objectmodel::SingleLink<OptiTrackNatNetDevice,OptiTrackNatNetClient,sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK> natNetClient;
    sofa::core::objectmodel::SingleLink<OptiTrackNatNetDevice,sofa::core::behavior::MechanicalState<DataTypes>,sofa::core::objectmodel::BaseLink::FLAG_STRONGLINK> mstate;
    sofa::core::objectmodel::DataFileName inMarkersMeshFile;
    sofa::core::objectmodel::DataFileName simMarkersMeshFile;
    sofa::core::objectmodel::Data<sofa::helper::vector<CPos> > inLocalMarkers;
    sofa::core::objectmodel::Data<sofa::helper::vector<CPos> > simLocalMarkers;
    sofa::core::objectmodel::Data<bool> tracked; ///< Output: true when this device is visible and tracked by the cameras
    sofa::core::objectmodel::Data<Coord> trackedFrame; ///< Output: rigid frame, as given by OptiTrack
    sofa::core::objectmodel::Data<Coord> frame; ///< Output: rigid frame
    sofa::core::objectmodel::Data<CPos> position; ///< Output: rigid position (Vec3)
    sofa::core::objectmodel::Data<CRot> orientation; ///< Output: rigid orientation (Quat)
    sofa::core::objectmodel::Data<Coord> simGlobalFrame; ///< Input: world position and orientation of the reference point in the simulation
    sofa::core::objectmodel::Data<Coord> inGlobalFrame; ///< Input: world position and orientation of the reference point in the real (camera) space
    sofa::core::objectmodel::Data<Coord> simLocalFrame; ///< Input: position and orientation of the center of the simulated object in the simulation
    sofa::core::objectmodel::Data<Coord> inLocalFrame; ///< Input: position and orientation of the center of the simulated object in the real (camera) space
    sofa::core::objectmodel::Data<sofa::helper::vector<CPos> > markers; ///< Output: markers as tracked by the cameras
    sofa::core::objectmodel::Data<sofa::helper::vector<int> > markersID; ///< Output: markers IDs
    sofa::core::objectmodel::Data<sofa::helper::vector<Real> > markersSize; ///< Output: markers sizes

    sofa::core::objectmodel::Data<sofa::helper::fixed_array<int,2> > distanceMarkersID; ///< Input: ID of markers ID used to measure distance (for articulated instruments)
    sofa::core::objectmodel::Data<sofa::helper::vector<CPos> > distanceMarkersPos; ///< Output: Positions of markers used to measure distance (for articulated instruments)
    sofa::core::objectmodel::Data<Real> openDistance; ///< Input: Distance considered as open
    sofa::core::objectmodel::Data<Real> closedDistance; ///< Input: Distance considered as closed
    sofa::core::objectmodel::Data<Real> distance; ///< Output: Measured distance
    sofa::core::objectmodel::Data<Real> distanceFactor; ///< Output: distance factor (0 = closed, 1 = open)
    sofa::core::objectmodel::Data<bool> open; ///< Output: true if measured distance is above openDistance
    sofa::core::objectmodel::Data<bool> closed; ///< Output: true if measured distance is below closedDistance

    sofa::core::objectmodel::Data<CPos> jointCenter; ///< Input: rotation center (for articulated instruments)
    sofa::core::objectmodel::Data<CPos> jointAxis; ///< Input: rotation axis (for articulated instruments)
    sofa::core::objectmodel::Data<Real> jointOpenAngle; ///< Input: rotation angle when opened (for articulated instruments)
    sofa::core::objectmodel::Data<Real> jointClosedAngle; ///< Input: rotation angle when closed (for articulated instruments)


    sofa::core::objectmodel::Data<sofa::defaulttype::Vec3f> drawAxisSize; ///< Size of displayed axis
    sofa::core::objectmodel::Data<float> drawMarkersSize; ///< Size of displayed markers
    sofa::core::objectmodel::Data<float> drawMarkersIDSize; ///< Size of displayed markers ID
    sofa::core::objectmodel::Data<sofa::defaulttype::Vec4f> drawMarkersColor; ///< Color of displayed markers

protected:
    int writeInMarkersMesh;
    int readSimMarkersMesh;
    Real smoothDistance;
};


} // namespace SofaOptiTrackNatNet

#endif /* OPTITRACKNATNETDEVICE_H */
