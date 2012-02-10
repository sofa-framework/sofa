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
#include <algorithm>

namespace SofaOptiTrackNatNet
{

OptiTrackNatNetDevice::OptiTrackNatNetDevice()
    : trackableName(initData(&trackableName, std::string(""), "trackableName", "NatNet trackable name"))
    , trackableID(initData(&trackableID, -1, "trackableID", "NatNet trackable number (ignored if trackableName is set)"))
    , natNetClient(initLink("natNetClient","Main OptiTrackNatNetClient instance"))
    , mstate(initLink("mstate","MechanicalState controlled by this device"))
    , markersMeshFile(initData(&markersMeshFile, "markersMeshFile", "OBJ file where markers in the object's local frame are stored. This file is created if it does not exist, otherwise it is used to compute transformation applied to tracked frame (inLocalFrame/simLocalFrame)"))
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

template<class T1, class T2> bool compare_pair_first(const std::pair<T1,T2>& e1, const std::pair<T1,T2>& e2)
{
    return e1.first < e2.first;
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

        if (inLocalMarkers.getValue().empty())
        {
            // first time this object is tracked -> save input markers
            sofa::helper::vector<CPos> inMarkers;
            Real mScale = this->scale.getValue();
            inMarkers.resize(rigid.nMarkers);
            for (int m=0; m<rigid.nMarkers; ++m)
                inMarkers[m] = frame.getOrientation().inverse().rotate(markers[m] - frame.getCenter()) * mScale;
            this->inLocalMarkers.setValue(inMarkers);
            if (markersMeshFile.isSet())
            {
                std::string meshFile = markersMeshFile.getFullPath();
                std::ifstream infile(meshFile.c_str());
                if (!infile.is_open()) // file open failed, try creating it instead
                {
                    std::ofstream outfile(meshFile.c_str());
                    if( !outfile.is_open() )
                    {
                        serr << "Error opening or creating file " << meshFile << sendl;
                    }
                    else // write markers to mesh file
                    {
                        Real mSize = drawMarkersSize.getValue();
                        if (mSize <= 0)
                        {
                            // use default size
                            mSize = 0.1f; // TODO: scale with markers distances
                        }
                        for (int m=0; m<rigid.nMarkers; ++m)
                        {
                            CPos center = inMarkers[m];
                            outfile << "v " << center+CPos(mSize,0,0) << std::endl;
                            outfile << "v " << center-CPos(mSize,0,0) << std::endl;
                            outfile << "v " << center+CPos(0,mSize,0) << std::endl;
                            outfile << "v " << center-CPos(0,mSize,0) << std::endl;
                            outfile << "v " << center+CPos(0,0,mSize) << std::endl;
                            outfile << "v " << center-CPos(0,0,mSize) << std::endl;
                            int i0 = m*6+1;
                            outfile << "f " << i0+0 << " " << i0+2 << " " << i0+4;
                            outfile << "f " << i0+2 << " " << i0+1 << " " << i0+4;
                            outfile << "f " << i0+1 << " " << i0+3 << " " << i0+4;
                            outfile << "f " << i0+3 << " " << i0+0 << " " << i0+4;
                            outfile << "f " << i0+2 << " " << i0+0 << " " << i0+5;
                            outfile << "f " << i0+1 << " " << i0+2 << " " << i0+5;
                            outfile << "f " << i0+3 << " " << i0+1 << " " << i0+5;
                            outfile << "f " << i0+0 << " " << i0+3 << " " << i0+5;
                        }
                    }
                }
                else // read markers from mesh file
                {
                    sofa::helper::vector<CPos> vertices;
                    sofa::helper::vector<sofa::helper::fixed_array<int,3> > triangles;
                    std::string line, cmd;
                    while (!infile.eof())
                    {
                        std::getline(infile,line);
                        std::istringstream str(line);
                        str >> cmd;
                        if (cmd == "v")
                        {
                            Real vx,vy,vz;
                            str >> vx >> vy >> vz;
                            vertices.push_back(CPos(vx,vy,vz));
                        }
                        else if (cmd == "f")
                        {
                            sofa::helper::fixed_array<int,3> f;
                            std::string s;
                            str >> s;
                            f[0] = atoi(s.c_str())-1;
                            str >> s;
                            f[1] = atoi(s.c_str())-1;
                            while (str >> s)
                            {
                                f[2] = atoi(s.c_str())-1;
                                triangles.push_back(f);
                                f[1] = f[2];
                            }
                        }
                    }
                    // compute connected components to identify each marker
                    std::map<int,std::vector<int> > groups;
                    std::vector<int> vGroup;
                    vGroup.resize(vertices.size());
                    for (unsigned int v=0; v<vertices.size(); ++v)
                        vGroup[v] = -1;
                    int ngroups = 0;
                    for (unsigned int f=0; f<triangles.size(); ++f)
                    {
                        int group = -1;
                        for (unsigned int j=0; j<triangles[f].size(); ++j)
                        {
                            int gj = vGroup[triangles[f][j]];
                            if (gj != -1 && (group == -1 || groups[group].size() < groups[gj].size()))
                                group = gj;
                        }

                        if (group == -1) // need to create a new group
                            group = ngroups++;

                        std::vector<int>& gvec = groups[group];
                        for (unsigned int j=0; j<triangles[f].size(); ++j)
                        {
                            int gj = vGroup[triangles[f][j]];
                            if (gj == -1) // this point was not part of a group
                            {
                                gvec.push_back(triangles[f][j]);
                                vGroup[triangles[f][j]] = group;
                            }
                            else if (gj != group) // copy the whole group
                            {
                                std::vector<int>& gjvec = groups[gj];
                                gvec.reserve(gvec.size()+gjvec.size());
                                for (std::vector<int>::const_iterator it = gjvec.begin(), itend = gjvec.end(); it != itend; ++it)
                                {
                                    gvec.push_back(*it);
                                    vGroup[*it] = group;
                                }
                                gjvec.clear();
                            }
                        }
                    }
                    // fill simMarkers with the center of mass of each group
                    sofa::helper::vector<CPos> simMarkers;
                    simMarkers.reserve(rigid.nMarkers);
                    for (std::map<int,std::vector<int> >::const_iterator itg = groups.begin(), itgend = groups.end(); itg != itgend; ++itg)
                    {
                        const std::vector<int>& gvec = itg->second;
                        if (gvec.empty()) continue;
                        // if(gvec.size() != 6) continue;
                        CPos center;
                        for (std::vector<int>::const_iterator it = gvec.begin(), itend = gvec.end(); it != itend; ++it)
                            center += vertices[*it];
                        center /= gvec.size();
                        simMarkers.push_back(center);
                    }
                    this->simLocalMarkers.setValue(simMarkers);
                    sout << "Read " << simMarkers.size() << " markers from mesh file" << sendl;
                    if (simMarkers.size() == inMarkers.size())
                    {
                        sout << "Computing transformation between input markers and positions from mesh file..." << sendl;
                        // Compute center of mass of both sets of markers
                        CPos inCenter, simCenter;
                        CRot inOrientation, simOrientation;
                        for (unsigned int m=0; m<inMarkers.size(); ++m)
                            inCenter += inMarkers[m];
                        inCenter /= inMarkers.size();

                        for (unsigned int m=0; m<simMarkers.size(); ++m)
                            simCenter += simMarkers[m];
                        simCenter /= simMarkers.size();
                        if (simMarkers.size() > 2) // we need more than 2 markers to evaluate rotation
                        {
                            Real inSumDist = 0, simSumDist = 0;
                            for (unsigned int m=0; m<inMarkers.size(); ++m)
                                inSumDist += (inMarkers[m]-inCenter).norm();
                            for (unsigned int m=0; m<simMarkers.size(); ++m)
                                simSumDist += (simMarkers[m]-simCenter).norm();
                            Real in2simScale = (inSumDist == 0) ? (Real)1 : simSumDist/inSumDist;

                            // order the markers by the area of the biggest triangle they form with the center and another point
                            std::vector<std::pair<Real,CPos> > sortedMarkers;
                            sortedMarkers.resize(inMarkers.size());
                            for (unsigned int m=0; m<inMarkers.size(); ++m)
                            {
                                CPos dir = (inMarkers[m]-inCenter);
                                Real area = 0;
                                for (unsigned int m2=0; m2<inMarkers.size(); ++m2)
                                {
                                    Real a = cross(dir,inMarkers[m2]-inCenter).norm();
                                    if (a > area) area = a;
                                }
                                sortedMarkers[m].first = -area;
                                sortedMarkers[m].second = inMarkers[m];
                            }
                            std::sort(sortedMarkers.begin(), sortedMarkers.end(), compare_pair_first<Real,CPos>);
                            for (unsigned int m=0; m<inMarkers.size(); ++m)
                                inMarkers[m] = sortedMarkers[m].second;

                            sortedMarkers.resize(simMarkers.size());
                            for (unsigned int m=0; m<simMarkers.size(); ++m)
                            {
                                CPos dir = (simMarkers[m]-simCenter);
                                Real area = 0;
                                for (unsigned int m2=0; m2<simMarkers.size(); ++m2)
                                {
                                    Real a = cross(dir,simMarkers[m2]-simCenter).norm();
                                    if (a > area) area = a;
                                }
                                sortedMarkers[m].first = -area;
                                sortedMarkers[m].second = simMarkers[m];
                            }
                            std::sort(sortedMarkers.begin(), sortedMarkers.end(), compare_pair_first<Real,CPos>);
                            for (unsigned int m=0; m<simMarkers.size(); ++m)
                                simMarkers[m] = sortedMarkers[m].second;
                            Real bestSimArea = -sortedMarkers[0].first;

                            sofa::defaulttype::Mat<3,3,Real> inFrame;
                            unsigned int inMX = 0;
                            unsigned int inMY = 1;
                            inFrame[0] = inMarkers[inMX] - inCenter;
                            for (unsigned int m=2; m<inMarkers.size(); ++m)
                            {
                                if (cross(inFrame[0],inMarkers[m] - inCenter).norm2() >
                                    cross(inFrame[0],inMarkers[inMY] - inCenter).norm2())
                                    inMY = m;
                            }
                            inFrame[1] = inMarkers[inMY] - inCenter;
                            inFrame[2] = cross(inFrame[0],inFrame[1]);
                            inFrame[1] = cross(inFrame[2],inFrame[0]);
                            Real inXDist = inFrame[0].norm() * in2simScale;
                            inFrame[0].normalize();
                            inFrame[1].normalize();
                            inFrame[2].normalize();
                            std::vector<CPos> inMarkersXForm;
                            inMarkersXForm.resize(inMarkers.size());
                            for (unsigned int m=0; m<inMarkers.size(); ++m)
                                inMarkersXForm[m] = inFrame * ((inMarkers[m] - inCenter)*in2simScale);

                            sofa::defaulttype::Mat<3,3,Real> bestSimFrame;
                            Real bestMatchError = -1;

                            for (unsigned int simMX = 0; simMX < simMarkers.size(); ++simMX)
                            {
                                Real simXDist = (simMarkers[simMX] - simCenter).norm();
                                Real xError = (simXDist-inXDist)*(simXDist-inXDist);
                                if (xError >= bestMatchError) continue;
                                for (unsigned int simMY = 0; simMY < simMarkers.size(); ++simMY)
                                {
                                    if (simMY == simMX) continue;
                                    Real area = cross(simMarkers[simMX] - simCenter, simMarkers[simMY] - simCenter).norm();
                                    if (area < 0.9f*bestSimArea) continue;
                                    sofa::defaulttype::Mat<3,3,Real> simFrame;
                                    simFrame[0] = simMarkers[simMX] - simCenter;
                                    simFrame[1] = simMarkers[simMY] - simCenter;
                                    simFrame[2] = cross(simFrame[0],simFrame[1]);
                                    simFrame[1] = cross(simFrame[2],simFrame[0]);
                                    simFrame[0].normalize();
                                    simFrame[1].normalize();
                                    simFrame[2].normalize();
                                    Real err = xError;
                                    for (unsigned int m=0; m<simMarkers.size(); ++m)
                                    {
                                        if (m == simMX) continue;
                                        CPos xform = simFrame * (simMarkers[m] - simCenter);
                                        err += (inMarkersXForm[m] - xform).norm2();
                                        if (bestMatchError >= 0 && err >= bestMatchError) break;
                                    }
                                    if (bestMatchError < 0 || err < bestMatchError)
                                    {
                                        bestMatchError = err;
                                        bestSimFrame = simFrame;
                                    }
                                }
                            }
                            if (bestMatchError >= 0)
                            {
                                inOrientation.fromMatrix(inFrame.transposed());
                                simOrientation.fromMatrix(bestSimFrame.transposed());
                            }
                        }
                        this->inLocalFrame.setValue(Coord(inCenter,inOrientation));
                        this->simLocalFrame.setValue(Coord(simCenter, simOrientation));
                    }
                }
            }
        }


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
