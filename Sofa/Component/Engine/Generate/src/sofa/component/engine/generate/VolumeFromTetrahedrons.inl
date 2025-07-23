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
#include <sofa/component/engine/generate/VolumeFromTetrahedrons.h>
#include <sofa/geometry/Hexahedron.h>
#include <sofa/geometry/Tetrahedron.h>

namespace sofa::component::engine::generate
{

using sofa::core::objectmodel::ComponentState;
using sofa::helper::ReadAccessor;
using sofa::type::vector;
using sofa::core::ConstVecCoordId;
using sofa::core::objectmodel::BaseData;


template <class DataTypes>
VolumeFromTetrahedrons<DataTypes>::VolumeFromTetrahedrons()
    :
      l_topology(initLink("topology", "link to the topology"))
    , l_state(initLink("mechanical", "link to the mechanical"))
    , d_positions(initData(&d_positions,"position","If not set by user, find the context mechanical."))
    , d_tetras(initData(&d_tetras,"tetras","If not set by user, find the context topology."))
    , d_hexas(initData(&d_hexas,"hexas","If not set by user, find the context topology."))
    , d_volume(initData(&d_volume,Real(0.0),"volume","The computed volume."))
    , d_doUpdate(initData(&d_doUpdate,false,"update","If true, will update the volume at each time step of the simulation."))
{
    d_volume.setReadOnly(true);
}

template <class DataTypes>
void VolumeFromTetrahedrons<DataTypes>::parse(core::objectmodel::BaseObjectDescription* arg)
{
    Inherit1::parse(arg);

    // SOFA_ATTRIBUTE_DISABLED("v25.06", "v26.06", "data positions renamed as position")
    // to be backward compatible with previous data structure
    const char* positionsChar = arg->getAttribute("positions");
    if( positionsChar )
        msg_deprecated() << "You are using a deprecated Data 'positions', please use 'position' instead.";
}

template <class DataTypes>
VolumeFromTetrahedrons<DataTypes>::~VolumeFromTetrahedrons()
{
}


template <class DataTypes>
void VolumeFromTetrahedrons<DataTypes>::init()
{
    Inherit1::init();

    d_componentState.setValue(ComponentState::Valid);

    addInput(&d_positions);
    addInput(&d_tetras);
    addInput(&d_hexas);

    addOutput(&d_volume);

    if(!d_positions.isSet())
    {
        if(!l_state.get())
        {
            msg_info() << "Link to the mechanical state should be set to ensure right behavior. First mechanical state found in current context will be used.";
            l_state.set(dynamic_cast<MechanicalState*>(this->getContext()->getMechanicalState()));
        }

        if(!l_state.get())
        {
            msg_error() << "No positions given by the user and no mechanical state found in the context. The component cannot work.";
            d_componentState.setValue(ComponentState::Invalid);
            return;
        }

        d_positions.setParent(l_state.get()->findData("position")); // Links d_positions to m_state.position
    }

    initTopology();
    checkTopology();
    updateVolume();
}



template <class DataTypes>
void VolumeFromTetrahedrons<DataTypes>::reinit()
{
    if(d_componentState.getValue() != ComponentState::Valid)
            return ;

    updateVolume();
}


template <class DataTypes>
void VolumeFromTetrahedrons<DataTypes>::initTopology()
{   
    if (!l_topology.get())
    {
        msg_info() << "Link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    const auto& topology = l_topology.get();
    const bool hasTetras = d_tetras.isSet();
    const bool hasHexas = d_hexas.isSet();

    if(!hasTetras && !hasHexas && !topology)
    {
        msg_error() << "No tetras or hexas given by the user and no topology context. The component cannot work";
        d_componentState.setValue(ComponentState::Invalid);
        return;
    }

    if(!hasTetras && topology)
        d_tetras.setValue(topology->getTetras());

    if(!hasHexas && topology)
        d_hexas.setValue(topology->getHexas());
}


template <class DataTypes>
void VolumeFromTetrahedrons<DataTypes>::checkTopology()
{
    ReadAccessor<sofa::Data<VecCoord> >  positions = d_positions;
    ReadAccessor<sofa::Data<VecTetras> > tetras    = d_tetras;
    ReadAccessor<sofa::Data<VecHexas> >  hexas     = d_hexas;

    /// Check that the tetras datafield does not contains indices that would crash the
    /// component.
    int nbTetras = tetras.size() ;
    for(int i=0;i<nbTetras;i++)
    {
        for(int j=0;j<4;j++)
        {
            if( tetras[i][j] >= positions.size() )
            {
                msg_error() << "tetras[" << i << "]["<< j << "]="<< tetras[i][j]
                              <<". is too large regarding positions size of(" << positions.size() << ")" ;
                d_componentState.setValue(ComponentState::Invalid);
            }
        }
    }

    /// Check that the hexas datafield does not contains indices that would crash the
    /// component.
    int nbHexas = hexas.size() ;
    for(int i=0;i<nbHexas;i++)
    {
        for(int j=0;j<6;j++)
        {
            if( hexas[i][j] >= positions.size() )
            {
                msg_error() << "hexas [" <<i << "][" << j << "]=" << hexas[i][j]
                              << " is too large regarding positions size of("
                              << positions.size() << ")" ;
                d_componentState.setValue(ComponentState::Invalid);
            }
        }
    }

    if (nbHexas + nbTetras == 0)
    {
        msg_error() << "Something is wrong with the topology. No tetrahedrons nor hexahedrons were given. The component cannot work.";
        d_componentState.setValue(ComponentState::Invalid);
    }
}



template <class DataTypes>
void VolumeFromTetrahedrons<DataTypes>::doUpdate()
{
    if(d_componentState.getValue() != ComponentState::Valid)
            return ;

    const auto& state = l_state.get();
    if(state && d_doUpdate.getValue())
    {
        ReadAccessor<sofa::Data<VecCoord> > positions = state->readPositions();
        d_positions.setValue(positions.ref());
        updateVolume();
    }
}


template <class DataTypes>
void VolumeFromTetrahedrons<DataTypes>::updateVolume()
{
    Real volume = 0.;

    ReadAccessor<sofa::Data<VecTetras>> tetras = d_tetras;
    ReadAccessor<sofa::Data<VecHexas>>  hexas  = d_hexas;
    ReadAccessor<sofa::Data<VecCoord> > positions = d_positions;

    for (const auto& t: tetras)
        volume += sofa::geometry::Tetrahedron::volume(positions[t[0]], positions[t[1]], positions[t[2]], positions[t[3]]);

    for (const auto& h: hexas)
        volume += sofa::geometry::Hexahedron::volume(positions[h[0]], positions[h[1]], positions[h[2]], positions[h[3]],
                                                     positions[h[4]], positions[h[5]], positions[h[6]], positions[h[7]]);

    if(volume<0) volume = -volume;
    d_volume.setValue(volume);
}


} // namespace
