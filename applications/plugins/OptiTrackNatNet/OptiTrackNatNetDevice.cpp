/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include "OptiTrackNatNetDevice.h"
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/DeactivatedNodeVisitor.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/glut.h>
#include <algorithm>

namespace SofaOptiTrackNatNet
{

OptiTrackNatNetDevice::OptiTrackNatNetDevice()
    : trackableName(initData(&trackableName, std::string(""), "trackableName", "NatNet trackable name"))
    , trackableID(initData(&trackableID, -1, "trackableID", "NatNet trackable number (ignored if trackableName is set)"))
    , setRestShape(initData(&setRestShape, false, "setRestShape", "True to control the rest position instead of the current position directly"))
    , applyMappings(initData(&applyMappings, true, "applyMappings", "True to enable applying the mappings after setting the position"))
    , controlNode(initData(&controlNode, false, "controlNode", "True to enable activating and disabling the node when this device appears and disappears"))
    , isGlobalFrame(initData(&isGlobalFrame, false, "isGlobalFrame", "True if this trackable should be considered as the global frame (i.e. all other trackables are computed relative to its position). This requires linking other trackables' \"inGlobalFrame\" to this \"frame\")"))
    , natNetClient(initLink("natNetClient","Main OptiTrackNatNetClient instance"))
    , mstate(initLink("mstate","MechanicalState controlled by this device"))
    , inMarkersMeshFile(initData(&inMarkersMeshFile, "inMarkersMeshFile", "OBJ file where markers in the object's input local frame are written. This file is created if it does not exist and/or when Ctrl+M is pressed"))
    , simMarkersMeshFile(initData(&simMarkersMeshFile, "simMarkersMeshFile", "OBJ file where markers in the object's simulation local frame are loaded. If this file does exist, it is used to compute transformation applied to tracked frame (inLocalFrame/simLocalFrame)"))
    , tracked(initData(&tracked,false,"tracked", "Output: true when this device is visible and tracked by the cameras"))
    , trackedFrame(initData(&trackedFrame,"trackedFrame", "Output: rigid frame, as given by OptiTrack"))
    , frame(initData(&frame,"frame","Output: rigid frame"))
    , position(initData(&position,"position", "Output: rigid position (Vec3)"))
    , orientation(initData(&orientation,"orientation", "Output: rigid orientation (Quat)"))
    , simGlobalFrame(initData(&simGlobalFrame, "simGlobalFrame", "Input: world position and orientation of the reference point in the simulation"))
    , inGlobalFrame(initData(&inGlobalFrame, "inGlobalFrame", "Input: world position and orientation of the reference point in the real (camera) space"))
    , simLocalFrame(initData(&simLocalFrame, "simLocalFrame", "Input: position and orientation of the center of the simulated object in the simulation"))
    , inLocalFrame(initData(&inLocalFrame, "inLocalFrame", "Input: position and orientation of the center of the simulated object in the real (camera) space"))
    , markers(initData(&markers, "markers", "Output: markers as tracked by the cameras"))
    , markersID(initData(&markersID, "markersID", "Output: markers IDs"))
    , markersSize(initData(&markersSize, "markersSize", "Output: markers sizes"))
    , distanceMarkersID(initData(&distanceMarkersID, "distanceMarkersID", "Input: ID of markers ID used to measure distance (for articulated instruments)"))
    , distanceMarkersPos(initData(&distanceMarkersPos, "distanceMarkersPos", "Output: Positions of markers used to measure distance (for articulated instruments)"))
    , openDistance(initData(&openDistance, (Real)0, "openDistance", "Input: Distance considered as open"))
    , closedDistance(initData(&closedDistance, (Real)0, "closedDistance", "Input: Distance considered as closed"))
    , distance(initData(&distance, (Real)0, "distance", "Output: Measured distance"))
    , distanceFactor(initData(&distanceFactor, (Real)0, "distanceFactor", "Output: distance factor (0 = closed, 1 = open)"))
    , open(initData(&open, false, "open", "Output: true if measured distance is above openDistance"))

    , closed(initData(&closed, false, "closed", "Output: true if measured distance is below closedDistance"))

    , jointCenter(initData(&jointCenter, CPos(0,0,0), "jointCenter", "Input: rotation center (for articulated instruments)"))
    , jointAxis(initData(&jointAxis, CPos(0,0,1), "jointAxis", "Input: rotation axis (for articulated instruments)"))
    , jointOpenAngle(initData(&jointOpenAngle, (Real)10, "jointOpenAngle", "Input: rotation angle when opened (for articulated instruments)"))
    , jointClosedAngle(initData(&jointClosedAngle, (Real)-10, "jointClosedAngle", "Input: rotation angle when closed (for articulated instruments)"))

    , drawAxisSize(initData(&drawAxisSize, sofa::defaulttype::Vec3f(1,1,1), "drawAxisSize", "Size of displayed axis"))
    , drawMarkersSize(initData(&drawMarkersSize, 0.1f, "drawMarkersSize", "Size of displayed markers"))
    , drawMarkersIDSize(initData(&drawMarkersIDSize, 0.0f, "drawMarkersIDSize", "Size of displayed markers ID"))
    , drawMarkersColor(initData(&drawMarkersColor, sofa::defaulttype::Vec4f(1,1,1,1), "drawMarkersColor", "Color of displayed markers"))
    , writeInMarkersMesh(100)
    , readSimMarkersMesh(1)
    , smoothDistance(-1)
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
    if (jointCenter.isSet())
    {
        if (mstate->getSize() < 3)
            mstate->resize(3);
    }
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

void OptiTrackNatNetDevice::onKeyPressedEvent(sofa::core::objectmodel::KeypressedEvent* ev)
{
    switch (ev->getKey())
    {
    case 'm':
    case 'M':
    {
        if (this->tracked.getValue())
        {
            writeInMarkersMesh = 1;
            readSimMarkersMesh = 1;
        }
        break;
    }
    }
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
        if (controlNode.getValue())
        {
            if (tracked)
                this->getContext()->setActive(tracked);
            sofa::simulation::DeactivationVisitor visitor(sofa::core::ExecParams::defaultInstance(), tracked);
            this->getContext()->executeVisitor(&visitor);
            if (!tracked)
                this->getContext()->setActive(tracked);
        }
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

        if (rigid.markersID)
        {
            sofa::helper::WriteAccessor<sofa::core::objectmodel::Data<sofa::helper::vector<int> > > markersID = this->markersID;
            markersID.resize(rigid.nMarkers);
            for (int m=0; m<rigid.nMarkers; ++m)
                markersID[m] = rigid.markersID[m];
        }

        if (rigid.markersSize)
        {
            sofa::helper::WriteAccessor<sofa::core::objectmodel::Data<sofa::helper::vector<Real> > > markersSize = this->markersSize;
            markersSize.resize(rigid.nMarkers);
            for (int m=0; m<rigid.nMarkers; ++m)
                markersSize[m] = rigid.markersSize[m];
        }

        sofa::helper::WriteAccessor<sofa::core::objectmodel::Data<sofa::helper::vector<CPos> > > inMarkers = this->inLocalMarkers;
        inMarkers.resize(rigid.nMarkers);
        for (int m=0; m<rigid.nMarkers; ++m)
            inMarkers[m] = frame.getOrientation().inverse().rotate(markers[m] - frame.getCenter());

        if (writeInMarkersMesh > 1) --writeInMarkersMesh;
        else if (writeInMarkersMesh && inMarkersMeshFile.isSet())
        {
            // first time this object is tracked -> save input markers
            std::string meshFile = inMarkersMeshFile.getFullPath();
            std::ofstream outfile(meshFile.c_str());
            if( !outfile.is_open() )
            {
                serr << "Error creating file " << meshFile << sendl;
            }
            else // write markers to mesh file
            {
                serr << "Creating input markers mesh " << meshFile << sendl;
                Real mSize = drawMarkersSize.getValue();
                if (mSize <= 0)
                {
                    // use default size
                    mSize = 0.01f; // TODO: scale with markers distances
                }
                static CPos ovpos[6] =
                {
                    CPos( 0.00000000f, 0.00000000f, 1.00000000f), //  0
                    CPos( 1.00000000f, 0.00000000f, 0.00000000f), //  1
                    CPos( 0.00000000f, 1.00000000f, 0.00000000f), //  2
                    CPos(-1.00000000f, 0.00000000f, 0.00000000f), //  3
                    CPos( 0.00000000f,-1.00000000f, 0.00000000f), //  4
                    CPos( 0.00000000f, 0.00000000f,-1.00000000f)
                };//  5

                static int ofaces[8][3] =
                {
                    {0,1,2},{0,2,3},{0,3,4},{0,4,1},
                    {5,2,1},{5,3,2},{5,4,3},{5,1,4}
                };

                static CPos mvpos[18] =
                {
                    CPos( 0.00000000f, 0.00000000f, 1.00000000f), //  0

                    CPos( 0.70710678f, 0.00000000f, 0.70710678f), //  1
                    CPos( 0.00000000f, 0.70710678f, 0.70710678f), //  2
                    CPos(-0.70710678f, 0.00000000f, 0.70710678f), //  3
                    CPos( 0.00000000f,-0.70710678f, 0.70710678f), //  4

                    CPos( 1.00000000f, 0.00000000f, 0.00000000f), //  5
                    CPos( 0.70710678f, 0.70710678f, 0.00000000f), //  6
                    CPos( 0.00000000f, 1.00000000f, 0.00000000f), //  7
                    CPos(-0.70710678f, 0.70710678f, 0.00000000f), //  8
                    CPos(-1.00000000f, 0.00000000f, 0.00000000f), //  9
                    CPos(-0.70710678f,-0.70710678f, 0.00000000f), // 10
                    CPos( 0.00000000f,-1.00000000f, 0.00000000f), // 11
                    CPos( 0.70710678f,-0.70710678f, 0.00000000f), // 12

                    CPos( 0.70710678f, 0.00000000f,-0.70710678f), // 13
                    CPos( 0.00000000f, 0.70710678f,-0.70710678f), // 14
                    CPos(-0.70710678f, 0.00000000f,-0.70710678f), // 15
                    CPos( 0.00000000f,-0.70710678f,-0.70710678f), // 16

                    CPos( 0.00000000f, 0.00000000f,-1.00000000f)
                };// 17

                static int mfaces[32][3] =
                {
                    {0,1,2},{0,2,3},{0,3,4},{0,4,1},

                    {1,5,6},{1,6,2},{2,6,7},
                    {2,7,8},{2,8,3},{3,8,9},
                    {3,9,10},{3,10,4},{4,10,11},
                    {4,11,12},{4,12,1},{1,12,5},

                    {13,6,5},{13,14,6},{14,7,6},
                    {14,8,7},{14,15,8},{15,9,8},
                    {15,10,9},{15,16,10},{16,11,10},
                    {16,12,11},{16,13,12},{13,5,12},

                    {17,14,13},{17,15,14},{17,16,15},{17,13,16}
                };

                for (int m=0; m<rigid.nMarkers; ++m)
                {
                    CPos center = inMarkers[m];
                    for (int v=0; v<18; ++v)
                        outfile << "v " << center+mvpos[v]*mSize << std::endl;
                }

                for (int m=0; m<data->nOtherMarkers; ++m)
                {
                    CPos center = frame.getOrientation().inverse().rotate(data->otherMarkersPos[m] - frame.getCenter());
                    for (int v=0; v<6; ++v)
                        outfile << "v " << center+ovpos[v]*mSize << std::endl;
                }

                for (int m=0; m<rigid.nMarkers; ++m)
                {
                    for (int v=0; v<18; ++v)
                        outfile << "vn " << mvpos[v] << std::endl;
                }
                for (int m=0; m<data->nOtherMarkers; ++m)
                {
                    for (int v=0; v<6; ++v)
                        outfile << "vn " << ovpos[v] << std::endl;
                }

                for (int m=0; m<rigid.nMarkers; ++m)
                {
                    int i0 = 1+m*18;
                    for (int f=0; f<32; ++f)
                        outfile << "f " << i0+mfaces[f][0] << "/" << i0+mfaces[f][0] << " " << i0+mfaces[f][1] << "/" << i0+mfaces[f][1] << " " << i0+mfaces[f][2] << "/" << i0+mfaces[f][2] << std::endl;
                }
                for (int m=0; m<data->nOtherMarkers; ++m)
                {
                    int i0 = 1+rigid.nMarkers*18+m*6;
                    for (int f=0; f<8; ++f)
                        outfile << "f " << i0+ofaces[f][0] << "/" << i0+ofaces[f][0] << " " << i0+ofaces[f][1] << "/" << i0+ofaces[f][1] << " " << i0+ofaces[f][2] << "/" << i0+ofaces[f][2] << std::endl;
                }
            }
            writeInMarkersMesh = 0;
        }

        if (readSimMarkersMesh > 1) --readSimMarkersMesh;
        else if (readSimMarkersMesh && simMarkersMeshFile.isSet())
        {
            // first time this object is tracked -> save input markers
            std::string meshFile = simMarkersMeshFile.getFullPath();
            std::ifstream infile(meshFile.c_str());
            if( !infile.is_open() )
            {
                serr << "Error reading file " << meshFile << sendl;
            }
            else // read markers from mesh file
            {
                serr << "Reading simulation markers mesh " << meshFile << sendl;
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
                    if(gvec.size() == 6) continue; // ignore octagons (untracked markers)
                    //if(gvec.size() != 18) continue; // our markers have 18 vertices
                    CPos center;
                    for (std::vector<int>::const_iterator it = gvec.begin(), itend = gvec.end(); it != itend; ++it)
                        center += vertices[*it];
                    center /= gvec.size();
                    sout << "simMarker[" << simMarkers.size() << "] = " << center << " ( from " << gvec.size() << " vertices ) " << sendl;
                    simMarkers.push_back(center);
                }
                this->simLocalMarkers.setValue(simMarkers);
                sout << "Read " << simMarkers.size() << " markers from mesh file" << sendl;
                sofa::helper::vector<CPos> inMarkers = this->inLocalMarkers.getValue();
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
                    sout << " inCenter = " <<  inCenter << sendl;
                    sout << "simCenter = " << simCenter << sendl;
                    if (simMarkers.size() > 2) // we need more than 2 markers to evaluate rotation
                    {
                        Real inSumDist = 0, simSumDist = 0;
                        for (unsigned int m=0; m<inMarkers.size(); ++m)
                            inSumDist += (inMarkers[m]-inCenter).norm();
                        for (unsigned int m=0; m<simMarkers.size(); ++m)
                            simSumDist += (simMarkers[m]-simCenter).norm();
                        Real in2simScale = (inSumDist == 0) ? (Real)1 : simSumDist/inSumDist;
                        sout << "in2simScale = " << in2simScale << sendl;
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
                        Real bestInArea = -sortedMarkers[0].first;

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
                        //sout << "sim best frame: dirX = " << (simMarkers[0] - simCenter) << "    area = " << bestSimArea << sendl;

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
                        sout << " in best frame: dirX = " << inFrame[0] << "    dirY = " << inFrame[1] << "    area = " << bestInArea << sendl;
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
                            if (bestMatchError >= 0 && xError >= bestMatchError) continue;
                            for (unsigned int simMY = 0; simMY < simMarkers.size(); ++simMY)
                            {
                                if (simMY == simMX) continue;
                                Real area = cross(simMarkers[simMX] - simCenter, simMarkers[simMY] - simCenter).norm();
                                if (area < 0.5f*bestSimArea) continue;
                                sofa::defaulttype::Mat<3,3,Real> simFrame;
                                simFrame[0] = simMarkers[simMX] - simCenter;
                                simFrame[1] = simMarkers[simMY] - simCenter;
                                simFrame[2] = cross(simFrame[0],simFrame[1]);
                                sout << "Checking sim frame " << simMX << "-" << simMY <<" : dirX = " << simFrame[0] << "    dirY = " << inFrame[1] << "   area = " << area << sendl;
                                simFrame[1] = cross(simFrame[2],simFrame[0]);
                                simFrame[0].normalize();
                                simFrame[1].normalize();
                                simFrame[2].normalize();
                                Real err = xError;
                                for (unsigned int m=0; m<simMarkers.size(); ++m)
                                {
                                    if (m == simMX) continue;
                                    CPos xform = simFrame * (simMarkers[m] - simCenter);
                                    Real minE = (inMarkersXForm[0]-xform).norm2();
                                    for (unsigned int m2=1; m2<inMarkersXForm.size(); ++m2)
                                    {
                                        Real e = (inMarkersXForm[m2] - xform).norm2();
                                        if (e < minE) minE = e;
                                    }
                                    err += minE;
                                    if (bestMatchError >= 0 && err >= bestMatchError) break;
                                }
                                if (bestMatchError < 0 || err < bestMatchError)
                                {
                                    sout << "NEW BEST : residual = " << err << sendl;
                                    bestMatchError = err;
                                    bestSimFrame = simFrame;
                                }
                            }
                        }
                        if (bestMatchError >= 0)
                        {
                            inOrientation.fromMatrix(inFrame.transposed());
                            simOrientation.fromMatrix(bestSimFrame.transposed());
                            sout << " in orientation = " <<  inOrientation << sendl;
                            sout << "sim orientation = " << simOrientation << sendl;
                            for (unsigned int m=0; m<simMarkers.size(); ++m)
                            {
                                CPos xform = simOrientation.inverseRotate(simMarkers[m] - simCenter);
                                Real minE = (inMarkersXForm[0]-xform).norm();
                                int minM2 = -1;
                                for (unsigned int m2=0; m2<inMarkers.size(); ++m2)
                                {
                                    Real e = (inOrientation.inverseRotate(inMarkers[m2] - inCenter) - xform).norm();
                                    if (e < minE) { minE = e; minM2 = m2; }
                                }
                                if (minM2 >= 0)
                                    sout << "  sim/in marker: " << m << " / " << minM2 << "    position = " << simMarkers[m] << " / " << inMarkers[minM2] << "    xform = " << xform << " / " << inMarkersXForm[minM2] << "    dist = " << minE << sendl;
                            }
                        }
                    }
                    this-> inLocalFrame.setValue(Coord( inCenter,  inOrientation));
                    this->simLocalFrame.setValue(Coord(simCenter, simOrientation));
                }
            }
            readSimMarkersMesh = 0;
        }

        if (this->inLocalFrame.isSet())
        {
            Coord frame2 = this->inLocalFrame.getValue();
            frame.getCenter() = frame.getCenter() + frame.getOrientation().rotate(frame2.getCenter());
            frame.getOrientation() = frame.getOrientation() * frame2.getOrientation();
            //sout << "   inLocalFrame  " << frame2 << " -> " << frame << sendl;
        }
        if (this->simLocalFrame.isSet())
        {
            Coord frame2 = this->simLocalFrame.getValue();
            frame.getOrientation() = frame.getOrientation() * frame2.getOrientation().inverse();
            frame.getCenter() = frame.getCenter() - frame.getOrientation().rotate(frame2.getCenter());
            //sout << "  simLocalFrame  " << frame2 << " -> " << frame << sendl;
        }
        if (this->inGlobalFrame.isSet())
        {
            Coord frame2 = this->inGlobalFrame.getValue();
            frame.getCenter() = frame2.getOrientation().inverse().rotate(frame.getCenter() - frame2.getCenter());
            frame.getOrientation() = frame2.getOrientation().inverse() * frame.getOrientation();
            for (int m=0; m<rigid.nMarkers; ++m)
                markers[m] = frame2.getOrientation().inverse().rotate(markers[m] - frame2.getCenter());
            //sout << "   inGlobalFrame " << frame2 << " -> " << frame << sendl;
        }
        if (this->simGlobalFrame.isSet())
        {
            Coord frame2 = this->simGlobalFrame.getValue();
            frame.getCenter() = frame2.getOrientation().rotate(frame.getCenter()) + frame2.getCenter();
            frame.getOrientation() = frame2.getOrientation() * frame.getOrientation();
            for (int m=0; m<rigid.nMarkers; ++m)
                markers[m] = frame2.getOrientation().rotate(markers[m]) + frame2.getCenter();
            //sout << "  simGlobalFrame " << frame2 << " -> " << frame << sendl;
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

        if (distanceMarkersID.isSet())
        {
            sofa::helper::fixed_array<int,2> distM = distanceMarkersID.getValue();
            const sofa::helper::vector<int>& markersID = this->markersID.getValue();
            if (!markersID.empty())
            {
                sofa::helper::fixed_array<int,2> distMIndex(-1,-1);

                for (unsigned int i=0; i<markersID.size(); ++i)
                {
                    if (markersID[i] == distM[0]) distMIndex[0] = i;
                    if (markersID[i] == distM[1]) distMIndex[1] = i;
                }
                distM = distMIndex;
            }
            sofa::helper::WriteAccessor<sofa::core::objectmodel::Data<sofa::helper::vector<CPos> > > distanceMarkersPos = this->distanceMarkersPos;
            distanceMarkersPos.resize(0);

            if (distM[0] != distM[1] && (unsigned)distM[0] < markers.size() && (unsigned)distM[1] < markers.size())
            {
                distanceMarkersPos.resize(2);
                distanceMarkersPos[0] = markers[distM[0]];
                distanceMarkersPos[1] = markers[distM[1]];
                Real dist = (markers[distM[0]] - markers[distM[1]]).norm();
                this->distance.setValue(dist);
                Real smoothDist = dist;
                if (smoothDistance >= 0)
                {
                    smoothDist = (smoothDist+smoothDistance)*(Real)0.5;
                }
                smoothDistance = smoothDist;
                Real openDist = openDistance.isSet() ? openDistance.getValue() : -smoothDist;
                Real closedDist = closedDistance.isSet() ? closedDistance.getValue() : -smoothDist;
                if (openDist < 0 || closedDist < 0)
                {
                    bool setOpenDist = !openDistance.isSet();
                    bool setClosedDist = !closedDistance.isSet();
                    if (smoothDist > -openDist) { openDist = -smoothDist; setOpenDist = true; }
                    if (smoothDist < -closedDist) { closedDist = -smoothDist; setClosedDist = true; }
                    if (setOpenDist)  { openDistance.setValue(openDist); sout << "Measured distance <= " << -openDist << sendl; }
                    if (setClosedDist)  { closedDistance.setValue(closedDist); sout << "Measured distance >= " << -closedDist << sendl; }
                    // we take the 80% percentile values into consideration
                    Real meanDist = (openDist+closedDist)*(Real)(-0.5);
                    Real varDist = (closedDist-openDist)*(Real)(0.8*0.5);
                    openDist = meanDist + varDist;
                    closedDist = meanDist - varDist;
                }
                if (closedDist < openDist*0.99)
                {
                    Real meanDist = (openDist+closedDist)*(Real)(0.5);
                    bool isOpen   = (smoothDist >=   openDist) || (  open.isSet() && open.getValue() && (smoothDist > meanDist));
                    bool isClosed = (smoothDist <= closedDist) || (closed.isSet() && closed.getValue() && (smoothDist < meanDist));
                    //bool isOpen = (smoothDist >= openDist);
                    //bool isClosed = (smoothDist <= closedDist);
                    if (!open.isSet() || isOpen != open.getValue() || !closed.isSet() || isClosed != closed.getValue())
                        sout << "Dist " << std::fixed << dist << " smooth " << std::fixed << smoothDist << "  interval " << closedDist << " - " << openDist << (isOpen ? " OPEN" : "") << (isClosed ? " CLOSED" : "") << sendl;
                    if (!open.isSet() || isOpen != open.getValue())
                    {
                        open.setValue(isOpen);
                    }
                    if (!closed.isSet() || isClosed != closed.getValue())
                    {
                        closed.setValue(isClosed);
                    }
                    Real distFact = (dist - closedDist) / (openDist - closedDist);
                    if (distFact < 0) distFact = 0; //else if (distFact > 1) distFact = 1;
                    if (!distanceFactor.isSet() || distFact != distanceFactor.getValue())
                        distanceFactor.setValue(distFact);
                }
            }
        }
    }
}

void OptiTrackNatNetDevice::draw(const sofa::core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowInteractionForceFields()) return;
    if (!this->tracked.getValue()) return;
    const sofa::defaulttype::Vec3f axisSize = drawAxisSize.getValue();
    const float markersSize = drawMarkersSize.getValue();
    const float markersIDSize = drawMarkersIDSize.getValue();
    if (isGlobalFrame.getValue())
    {
        vparams->drawTool()->pushMatrix();
        float glTransform[16];
        Coord xform = frame.getValue();
        xform.getOrientation() = xform.getOrientation().inverse();
        xform.getCenter() = -xform.getOrientation().rotate(xform.getCenter());
        xform.writeOpenGlMatrix ( glTransform );
        vparams->drawTool()->multMatrix( glTransform );
    }
    if (axisSize.norm2() > 0)
        vparams->drawTool()->drawFrame(position.getValue(), orientation.getValue(), axisSize);
    if (markersSize > 0)
        vparams->drawTool()->drawSpheres(markers.getValue(), markersSize, drawMarkersColor.getValue());
    if (markersSize > 0 && !distanceMarkersPos.getValue().empty())
    {
        vparams->drawTool()->drawLines(distanceMarkersPos.getValue(), markersSize/5, (open.getValue() ? sofa::defaulttype::Vec4f(0,1,0,1) : closed.getValue() ? sofa::defaulttype::Vec4f(1,0,0,1) : drawMarkersColor.getValue()));
    }
    if (markersIDSize && !markersID.getValue().empty())
    {
        sofa::defaulttype::Mat<4,4, GLfloat> modelviewM;
        glDisable(GL_LIGHTING);
        const sofa::helper::vector<CPos>& markers = this->markers.getValue();
        //const sofa::helper::vector<CPos>& inMarkers = this->inLocalMarkers.getValue();
        const sofa::helper::vector<int>& markersID = this->markersID.getValue();
        glPushMatrix();

        // Makes text always face the viewer by removing the scene rotation
        // get the current modelview matrix
        glGetFloatv(GL_MODELVIEW_MATRIX , modelviewM.ptr() );
        modelviewM.transpose();
        const float scale = markersIDSize/50;
        for (unsigned int i =0; i<markersID.size() && i<markers.size(); i++)
        {
            sofa::defaulttype::Vec3f center; center = markers[i];
            sofa::defaulttype::Vec3f temp = modelviewM.transform(center);
            {
                std::ostringstream oss;
                oss << std::hex << markersID[i];
                if ((int)i+1 != markersID[i]) oss << "-" << (i+1);
                std::string str = oss.str();
                glLoadIdentity();
                glTranslatef(temp[0], temp[1], temp[2]);
                glScalef(scale,scale,scale);
                glColor4fv(drawMarkersColor.getValue().ptr());
                for (unsigned int c=0; c<str.size(); ++c)
                    glutStrokeCharacter(GLUT_STROKE_ROMAN, str[c]);
            }
            /*
                        if (i < inMarkers.size())
                        {
                            std::ostringstream oss;
                            oss << inMarkers[i];
                            std::string str = oss.str();
                            glLoadIdentity();
                            glTranslatef(temp[0], temp[1]-scale*100, temp[2]);
                            glScalef(scale/2,scale/2,scale/2);
                            glColor4f(1,1,1,1);
                            for (unsigned int c=0;c<str.size();++c)
                                glutStrokeCharacter(GLUT_STROKE_ROMAN, str[c]);
                        }
            */
        }
        glPopMatrix();
    }
    if (jointCenter.isSet())
    {
        sofa::helper::vector<CPos> points;
        Coord xform = frame.getValue();
        points.push_back(xform.pointToParent(jointCenter.getValue() - jointAxis.getValue()));
        points.push_back(xform.pointToParent(jointCenter.getValue() + jointAxis.getValue()));
        vparams->drawTool()->drawLines(points, markersSize, sofa::defaulttype::Vec4f(0,0,1,1));
    }
    if (isGlobalFrame.getValue())
    {
        vparams->drawTool()->popMatrix();
    }
}

void OptiTrackNatNetDevice::update()
{
    if (!natNetClient.get()) return;
    if (mstate.get() && this->tracked.getValue())
    {
        sofa::helper::WriteAccessor<sofa::core::objectmodel::Data<VecCoord> > x = *this->mstate->write(this->setRestShape.getValue() ? sofa::core::VecCoordId::restPosition() : sofa::core::VecCoordId::position());
        sofa::helper::WriteAccessor<sofa::core::objectmodel::Data<VecCoord> > xfree = *this->mstate->write(this->setRestShape.getValue() ? sofa::core::VecCoordId::restPosition() : sofa::core::VecCoordId::freePosition());
        Coord pos = frame.getValue();
        if (!isGlobalFrame.getValue())
            pos = frame.getValue();
        x[0] = pos;
        xfree[0] = pos;

        if (jointCenter.isSet() && x.size() >= 3)
        {
            CPos jointCenter = this->jointCenter.getValue();
            Real angle = jointClosedAngle.getValue() + distanceFactor.getValue() * (jointOpenAngle.getValue() - jointClosedAngle.getValue());
            CRot rotation(jointAxis.getValue(), (Real)(angle*(M_PI/180.0)));

            Coord posLeft, posRight;
            posLeft.getOrientation() = pos.getOrientation() * rotation;
            posRight.getOrientation() = pos.getOrientation() * rotation.inverse();
            // pos.center + pos.orientation*(jointCenter) = posLeft.center + posLeft.orientation*(jointCenter)
            // posLeft.center = pos.center + pos.orientation(jointCenter) - posLeft.orientation(jointCenter)
            posLeft.getCenter() = pos.getCenter() + pos.getOrientation().rotate(jointCenter) - posLeft.getOrientation().rotate(jointCenter);
            posRight.getCenter() = pos.getCenter() + pos.getOrientation().rotate(jointCenter) - posRight.getOrientation().rotate(jointCenter);
            x[1] = posLeft;
            xfree[1] = posLeft;
            x[2] = posRight;
            xfree[2] = posRight;
        }

        if (applyMappings.getValue())
        {
            sofa::simulation::Node *node = dynamic_cast<sofa::simulation::Node*> (this->getContext());
            if (node)
            {
                sofa::simulation::MechanicalPropagatePositionAndVelocityVisitor mechaVisitor(sofa::core::MechanicalParams::defaultInstance()); mechaVisitor.execute(node);
                sofa::simulation::UpdateMappingVisitor updateVisitor(sofa::core::ExecParams::defaultInstance()); updateVisitor.execute(node);
            }
        }
    }
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
