/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/topology/CMTopologyHandler.h>


namespace sofa
{

namespace core
{

namespace cm_topology
{

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////   Generic Handling of Topology Event    /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TopologyHandler::ApplyTopologyChanges(const std::list<const core::cm_topology::TopologyChange *> &_topologyChangeEvents, const unsigned int _dataSize)
{
    if(!this->isTopologyDataRegistered())
        return;

    sofa::helper::list<const core::cm_topology::TopologyChange *>::iterator changeIt;
    sofa::helper::list<const core::cm_topology::TopologyChange *> _changeList = _topologyChangeEvents;

    this->setDataSetArraySize(_dataSize);

    for (changeIt=_changeList.begin(); changeIt!=_changeList.end(); ++changeIt)
    {
        core::cm_topology::TopologyChangeType changeType = (*changeIt)->getChangeType();

        switch( changeType )
        {
#define SOFA_CASE_EVENT(name,type) \
        case core::cm_topology::name: \
            this->ApplyTopologyChange(static_cast< const type* >( *changeIt ) ); \
            break

        SOFA_CASE_EVENT(ENDING_EVENT,EndingEvent);

        SOFA_CASE_EVENT(POINTSINDICESSWAP,PointsIndicesSwap);
        SOFA_CASE_EVENT(POINTSADDED,PointsAdded);
        SOFA_CASE_EVENT(POINTSREMOVED,PointsRemoved);
        SOFA_CASE_EVENT(POINTSMOVED,PointsMoved);
        SOFA_CASE_EVENT(POINTSRENUMBERING,PointsRenumbering);

        SOFA_CASE_EVENT(EDGESINDICESSWAP,EdgesIndicesSwap);
        SOFA_CASE_EVENT(EDGESADDED,EdgesAdded);
        SOFA_CASE_EVENT(EDGESREMOVED,EdgesRemoved);
        SOFA_CASE_EVENT(EDGESMOVED_REMOVING,EdgesMoved_Removing);
        SOFA_CASE_EVENT(EDGESMOVED_ADDING,EdgesMoved_Adding);
        SOFA_CASE_EVENT(EDGESRENUMBERING,EdgesRenumbering);

        SOFA_CASE_EVENT(FACESINDICESSWAP,FacesIndicesSwap);
        SOFA_CASE_EVENT(FACESADDED,FacesAdded);
        SOFA_CASE_EVENT(FACESREMOVED,FacesRemoved);
        SOFA_CASE_EVENT(FACESMOVED_REMOVING,FacesMoved_Removing);
        SOFA_CASE_EVENT(FACESMOVED_ADDING,FacesMoved_Adding);
        SOFA_CASE_EVENT(FACESRENUMBERING,FacesRenumbering);

        SOFA_CASE_EVENT(VOLUMESINDICESSWAP,VolumesIndicesSwap);
        SOFA_CASE_EVENT(VOLUMESADDED,VolumesAdded);
        SOFA_CASE_EVENT(VOLUMESREMOVED,VolumesRemoved);
        SOFA_CASE_EVENT(VOLUMESMOVED_REMOVING,VolumesMoved_Removing);
        SOFA_CASE_EVENT(VOLUMESMOVED_ADDING,VolumesMoved_Adding);
        SOFA_CASE_EVENT(VOLUMESRENUMBERING,VolumesRenumbering);
#undef SOFA_CASE_EVENT
        default:
            break;
        }; // switch( changeType )

        //++changeIt;
    }
}

} // namespace cm_topology

} // namespace core

} // namespace sofa
