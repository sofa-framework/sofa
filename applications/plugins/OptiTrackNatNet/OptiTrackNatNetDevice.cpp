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
#include "OptiTrackNatNetDevice.h"
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>


namespace SofaOptiTrackNatNet
{

OptiTrackNatNetDevice::OptiTrackNatNetDevice()
    : trackableName(initData(&trackableName, std::string(""), "trackableName", "NatNet trackable name"))
    , trackableID(initData(&trackableID, -1, "trackableID", "NatNet trackable number (ignored if trackableName is set)"))
    , natNetClient(initLink("natNetClient","Main OptiTrackNatNetClient instance"))
    , mstate(initLink("mstate","MechanicalState controlled by this device"))
    , tracked(initData(&tracked,false,"tracked", "Output: true when this device is visible and tracked by the cameras"))
    , frame(initData(&frame,"frame","Output: rigid frame"))
    , position(initData(&position,"position", "Output: rigid position (Vec3)"))
    , orientation(initData(&orientation,"orientation", "Output: rigid orientation (Quat)"))
    , trackedFrame(initData(&trackedFrame,"trackedFrame", "Output: rigid frame, as given by OptiTrack"))
    , simGlobalFrame(initData(&simGlobalFrame, "simGlobalFrame", "Input: world position and orientation of the reference point in the simulation"))
    , inGlobalFrame(initData(&inGlobalFrame, "inGlobalFrame", "Input: world position and orientation of the reference point in the real (camera) space"))
    , simLocalFrame(initData(&simLocalFrame, "simLocalFrame", "Input: position and orientation of the center of the simulated object in the simulation"))
    , inLocalFrame(initData(&inLocalFrame, "inLocalFrame", "Input: position and orientation of the center of the simulated object in the real (camera) space"))
    , scale(initData(&scale, (Real)1, "scale", "Input: scale factor to apply to coordinates (using the global frame as fixed point)"))
    , markers(initData(&markers, "markers", "Output: markers as tracked by the cameras"))
    , drawAxisSize(initData(&drawAxisSize, sofa::defaulttype::Vec3f(1,1,1), "drawAxisSize", "Size of displayed axis"))
    , drawMarkersSize(initData(&drawMarkersSize, 0.1f, "drawMarkersSize", "Size of displayed markers"))
    , drawMarkersColor(initData(&drawMarkersColor, sofa::defaulttype::Vec4f(1,1,1,1), "drawMarkersColor", "Color of displayed markers"))
{
    this->f_listening.setValue(true);
    this->f_printLog.setValue(true);
}

OptiTrackNatNetDevice::~OptiTrackNatNetDevice()
{
    if (natNetClient.get())
    {
        natNetClient->natNetReceivers.remove(this);
    }
}

void OptiTrackNatNetDevice::init()
{
    Inherit1::init();
    if (!mstate.get())
        mstate.set(dynamic_cast< sofa::core::behavior::MechanicalState<DataTypes>* >(getContext()->getMechanicalState()));
    if (!natNetClient.get())
    {
        OptiTrackNatNetClient* p = NULL;
        this->getContext()->get(p);
        if (!p)
            serr << "OptiTrackNatNetClient is missing, add it in the root of the simulation." << sendl;
        else
            natNetClient.set(p);
    }
    if (natNetClient.get())
    {
        natNetClient->natNetReceivers.add(this);
    }
}

void OptiTrackNatNetDevice::reinit()
{
    Inherit1::reinit();
}

void OptiTrackNatNetDevice::processModelDef(const ModelDef* data)
{
    std::string name = trackableName.getValue();
    int id = trackableID.getValue();
    if (!name.empty())
    {
        for (int i=0; i<data->nRigids; ++i)
            if (data->rigids[i].name && name == data->rigids[i].name)
            {
                id = data->rigids[i].ID;
                serr << "Found trackable " << name << " as ID " << id << sendl;
                trackableID.setValue(id);
                break;
            }
    }
    else if (id >= 0)
    {
        bool found = false;
        for (int i=0; i<data->nRigids; ++i)
        {
            if (data->rigids[i].ID == id)
            {
                sout << "Found trackable ID " << id << sendl;
                found = true;
                break;
            }
        }
        if (!found)
            serr << "Trackable ID " << id << " not found in input data" << sendl;
    }
}

void OptiTrackNatNetDevice::processFrame(const FrameData* data)
{
    int id = trackableID.getValue();
    int index = -1;
    for (int i=0; i<data->nRigids; ++i)
    {
        if (data->rigids[i].ID == id)
        {
            index = i;
            break;
        }
    }
    if (index < 0) return; // data not found
    const RigidData& rigid = data->rigids[index];
    CPos pos = rigid.pos;
    CRot rot = rigid.rot;
    bool tracked = (rot[0]*rot[0] + rot[1]*rot[1] + rot[2]*rot[2] + rot[3]*rot[3]) > 0.0001f;
    if (tracked != this->tracked.getValue())
    {
        if (tracked)
            sout << "Device is now tracked" << sendl;
        else
            sout << "Device lost" << sendl;
        this->tracked.setValue(tracked);
    }
    if (tracked)
    {
        Coord frame (pos, rot);
//        sout << "Tracked frame: " << frame << sendl;
        this->trackedFrame.setValue(frame);
        sofa::helper::WriteAccessor<sofa::core::objectmodel::Data<sofa::helper::vector<CPos> > > markers = this->markers;
        markers.resize(rigid.nMarkers);
        for (int m=0; m<rigid.nMarkers; ++m)
            markers[m] = rigid.markersPos[m];
        if (this->inLocalFrame.isSet())
        {
            Coord frame2 = this->inLocalFrame.getValue();
            frame.getCenter() = frame.getCenter() - frame.getOrientation().rotate(frame2.getCenter());
            frame.getOrientation() = frame.getOrientation() * frame2.getOrientation().inverse();
            sout << "   inLocalFrame  " << frame2 << " -> " << frame << sendl;
        }
        if (this->simLocalFrame.isSet())
        {
            Coord frame2 = this->simLocalFrame.getValue();
            frame.getCenter() = frame.getCenter() + frame.getOrientation().rotate(frame2.getCenter());
            frame.getOrientation() = frame.getOrientation() * frame2.getOrientation();
            sout << "  simLocalFrame  " << frame2 << " -> " << frame << sendl;
        }
        if (this->inGlobalFrame.isSet())
        {
            Coord frame2 = this->inGlobalFrame.getValue();
            frame.getCenter() = frame2.getOrientation().inverse().rotate(frame.getCenter()) - frame2.getCenter();
            frame.getOrientation() = frame2.getOrientation().inverse() * frame.getOrientation();
            for (int m=0; m<rigid.nMarkers; ++m)
                markers[m] = frame2.getOrientation().inverse().rotate(markers[m]) - frame2.getCenter();
            sout << "   inGlobalFrame " << frame2 << " -> " << frame << sendl;
        }
        if (this->scale.isSet())
        {
            Real scale = this->scale.getValue();
            frame.getCenter() *= scale;
            for (int m=0; m<rigid.nMarkers; ++m)
                markers[m] *= scale;
            //sout << "   scale         " << scale << " -> " << frame << sendl;
        }
        if (this->simGlobalFrame.isSet())
        {
            Coord frame2 = this->simGlobalFrame.getValue();
            frame.getCenter() = frame2.getOrientation().rotate(frame.getCenter()) + frame2.getCenter();
            frame.getOrientation() = frame2.getOrientation() * frame.getOrientation();
            for (int m=0; m<rigid.nMarkers; ++m)
                markers[m] = frame2.getOrientation().rotate(markers[m]) + frame2.getCenter();
            sout << "  simGlobalFrame " << frame2 << " -> " << frame << sendl;
        }
//        sout << "Output frame: " << frame << sendl;
        this->frame.setValue(frame);
        pos = frame.getCenter();
        rot = frame.getOrientation();
        this->position.setValue(pos);
        this->orientation.setValue(rot);

        sofa::defaulttype::BoundingBox bb(pos, pos);
        const sofa::defaulttype::Vec3f axisSize = drawAxisSize.getValue();
        if (axisSize.norm2() > 0.0f)
        {
            bb.include(pos+rot.rotate(CPos(axisSize[0],0,0)));
            bb.include(pos+rot.rotate(CPos(0,axisSize[1],0)));
            bb.include(pos+rot.rotate(CPos(0,0,axisSize[2])));
        }
        const float markersSize = drawMarkersSize.getValue();
        if (markersSize > 0)
        {
            for (int m=0; m<rigid.nMarkers; ++m)
            {
                bb.include(markers[m] - CPos(markersSize,markersSize,markersSize));
                bb.include(markers[m] + CPos(markersSize,markersSize,markersSize));
            }
        }
        this->f_bbox.setValue(bb);
    }
}

void OptiTrackNatNetDevice::draw(const sofa::core::visual::VisualParams* vparams)
{
    if (!this->tracked.getValue()) return;
    const sofa::defaulttype::Vec3f axisSize = drawAxisSize.getValue();
    const float markersSize = drawMarkersSize.getValue();
    if (axisSize.norm2() > 0)
        vparams->drawTool()->drawFrame(position.getValue(), orientation.getValue(), axisSize);
    if (markersSize > 0)
        vparams->drawTool()->drawSpheres(markers.getValue(), markersSize, drawMarkersColor.getValue());
}

void OptiTrackNatNetDevice::update()
{
    if (!natNetClient.get()) return;
}

void OptiTrackNatNetDevice::onBeginAnimationStep(const double /*dt*/)
{
    update();
}

SOFA_DECL_CLASS(OptiTrackNatNetDevice)

int OptiTrackNatNetDeviceClass = sofa::core::RegisterObject("Tracked rigid device relying on OptiTrackNatNetClient")
        .add< OptiTrackNatNetDevice >()
        ;

} // namespace SofaOptiTrackNatNet
