/******************************************************************************
 *                 SOFA, Simulation Open-Framework Architecture                *
 *                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
 *                                                                             *
 * This program is free software; you can redistribute it and/or modify it     *
 * under the terms of the GNU Lesser General Public License as published by    *
 * the Free Software Foundation; either version 2.1 of the License, or (at     *
 * your option) any later version.                                             *
 *                                                                             *
 * This program is distributed in the hope that it will be useful, but WITHOUT *
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
 * for more details.                                                           *
 *                                                                             *
 * You should have received a copy of the GNU Lesser General Public License    *
 * along with this program. If not, see <http://www.gnu.org/licenses/>.        *
 *******************************************************************************
 * Authors: The SOFA Team and external contributors (see Authors.txt)          *
 *                                                                             *
 * Contact information: contact@sofa-framework.org                             *
 ******************************************************************************/
#pragma once

#include <sofa/config.h>

#include <sofa/component/engine/generate/VolumeFromTriangles.h>

namespace sofa::component::engine::generate
{

using sofa::core::objectmodel::ComponentState;
using sofa::helper::ReadAccessor;
using sofa::type::vector;
using sofa::core::ConstVecCoordId;
using sofa::core::objectmodel::BaseData ;

template <class DataTypes>
VolumeFromTriangles<DataTypes>::VolumeFromTriangles()
    : d_positions(initData(&d_positions,"positions","If not set by user, find the context mechanical."))
    , d_triangles(initData(&d_triangles,"triangles","If not set by user, find the context topology."))
    , d_quads(initData(&d_quads,"quads","If not set by user, find the context topology."))
    , d_volume(initData(&d_volume,Real(0.0),"volume","The volume is only relevant if the surface is closed."))
    , d_doUpdate(initData(&d_doUpdate,false,"update","If true, will update the volume at each time step of the simulation."))
{
    d_volume.setReadOnly(true);
}


template <class DataTypes>
VolumeFromTriangles<DataTypes>::~VolumeFromTriangles()
{
}


template <class DataTypes>
void VolumeFromTriangles<DataTypes>::init()
{
    d_componentState.setValue(ComponentState::Valid);

    addInput(&d_positions);
    addInput(&d_triangles);
    addInput(&d_quads);

    addOutput(&d_volume);

    if(!d_positions.isSet())
    {
        m_state = dynamic_cast<MechanicalState*>(getContext()->getMechanicalState());

        if(m_state == nullptr)
        {
            msg_error() << "No positions given by the user and no mechanical state found in the context. The component cannot work.";
            d_componentState.setValue(ComponentState::Invalid);
            return;
        }

        d_positions.setParent(m_state->findData("position")); // Links d_positions to m_state.position
    }

    initTopology();
    checkTopology();
    updateVolume();
}


template <class DataTypes>
void VolumeFromTriangles<DataTypes>::reinit()
{
    if(d_componentState.getValue() != ComponentState::Valid)
        return ;

    updateVolume();
}


template <class DataTypes>
void VolumeFromTriangles<DataTypes>::initTopology()
{
    m_topology = getContext()->getMeshTopology();

    if(!d_triangles.isSet() && m_topology)
        d_triangles.setValue(m_topology->getTriangles());

    if(!d_quads.isSet() && m_topology)
        d_quads.setValue(m_topology->getQuads());

    if(!d_quads.isSet() && !d_triangles.isSet() && !m_topology)
    {
        msg_error() << "No quads or triangles given by the user and no topology context. The component cannot work";
        d_componentState.setValue(ComponentState::Invalid);
    }
}


template <class DataTypes>
void VolumeFromTriangles<DataTypes>::checkTopology()
{
    ReadAccessor<sofa::Data<VecCoord> >       positions = d_positions;
    ReadAccessor<sofa::Data<VecTriangles> >   triangles = d_triangles;
    ReadAccessor<sofa::Data<VecQuads> >       quads     = d_quads;

    /// Check that the triangles datafield does not contains indices that would crash the
    /// component.
    int nbTriangles = triangles.size() ;
    for(int i=0;i<nbTriangles;i++)
    {
        for(int j=0;j<3;j++)
        {
            if( triangles[i][j] >= positions.size() )
            {
                msg_error() << "triangles[" << i << "]["<< j << "]="<< triangles[i][j]
                              <<". is too large regarding positions size of(" << positions.size() << ")" ;
                d_componentState.setValue(ComponentState::Invalid);
            }
        }
    }

    /// Check that the quads datafield does not contains indices that would crash the
    /// component.
    int nbQuads = quads.size() ;
    for(int i=0;i<nbQuads;i++)
    {
        for(int j=0;j<4;j++)
        {
            if( quads[i][j] >= positions.size() )
            {
                msg_error() << "quads [" <<i << "][" << j << "]=" << quads[i][j]
                              << " is too large regarding positions size of("
                              << positions.size() << ")" ;
                d_componentState.setValue(ComponentState::Invalid);
            }
        }
    }
}


template <class DataTypes>
void VolumeFromTriangles<DataTypes>::doUpdate()
{
    if(d_componentState.getValue() != ComponentState::Valid)
        return ;

    if(m_state && d_doUpdate.getValue())
    {
        ReadAccessor<sofa::Data<VecCoord> > positions = m_state->readPositions();
        d_positions.setValue(positions.ref());
        updateVolume();
    }
}


template <class DataTypes>
void VolumeFromTriangles<DataTypes>::updateVolume()
{
    Real volume = 0;

    ReadAccessor<sofa::Data<VecCoord>>     positions = d_positions;
    ReadAccessor<sofa::Data<VecTriangles>> triangles = d_triangles;
    ReadAccessor<sofa::Data<VecQuads>>     quads     = d_quads;

    for (unsigned int t=0; t<triangles.size(); t++)
    {
        Coord p0, p1, p2;

        p0 = positions[triangles[t][0]];
        p1 = positions[triangles[t][1]];
        p2 = positions[triangles[t][2]];

        volume += ((p1[1]-p0[1])*(p2[2]-p0[2])-(p2[1]-p0[1])*(p1[2]-p0[2]))*(p0[0]+p1[0]+p2[0])/6;
    }

    for (unsigned int q=0; q<quads.size(); q++)
    {
        Coord p0, p1, p2;

        p0 = positions[quads[q][0]];
        p1 = positions[quads[q][1]];
        p2 = positions[quads[q][2]];

        volume += ((p1[1]-p0[1])*(p2[2]-p0[2])-(p2[1]-p0[1])*(p1[2]-p0[2]))*(p0[0]+p1[0]+p2[0])/6;

        p0 = positions[quads[q][0]];
        p1 = positions[quads[q][2]];
        p2 = positions[quads[q][3]];

        volume += ((p1[1]-p0[1])*(p2[2]-p0[2])-(p2[1]-p0[1])*(p1[2]-p0[2]))*(p0[0]+p1[0]+p2[0])/6;
    }

    if(volume<0) volume = -volume;
    d_volume.setValue(volume);
}


} // namespace

