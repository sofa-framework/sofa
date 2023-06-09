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
#include <sofa/component/topology/container/dynamic/DynamicSparseGridTopologyModifier.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/topology/container/dynamic/DynamicSparseGridTopologyContainer.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::topology::container::dynamic
{

int DynamicSparseGridTopologyModifierClass = core::RegisterObject ( "Hexahedron set topology modifier" )
        .add< DynamicSparseGridTopologyModifier >();


void DynamicSparseGridTopologyModifier::init()
{
    HexahedronSetTopologyModifier::init();
    this->getContext()->get ( m_DynContainer );

    if(!m_DynContainer)
    {
        msg_error() << "DynamicSparseGridTopologyContainer not found in current node: " << this->getContext()->getName();
        d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    everRenumbered = false;
}


//TODO// find a solution for this case !!!! Modifier can not access to the DOF and can not compute the indices of the added hexahedra.
// We have to find a way to automaticaly compute the indices of the added hexahedra to update the map 'm_m_DynContainer->idInRegularGrid2Hexa'
void DynamicSparseGridTopologyModifier::addHexahedraProcess ( const sofa::type::vector< Hexahedron > &hexahedra )
{
    HexahedronSetTopologyModifier::addHexahedraProcess ( hexahedra );
    msg_error() << "addHexahedraProcess( const sofa::type::vector< Hexahedron > &hexahedra ). You must not use this method. To add some voxels to the topology, you must use addHexahedraProcess ( const sofa::type::vector< Hexahedron > &hexahedra, const sofa::type::vector< unsigned int> &indices ) because, for the moment, indices maps can not be updated !";
}


void DynamicSparseGridTopologyModifier::addHexahedraProcess ( const sofa::type::vector< Hexahedron > &hexahedra, const sofa::type::vector< unsigned int> &indices )
{
    assert( hexahedra.size() == indices.size());

    const unsigned int hexaSize = m_DynContainer->getNumberOfHexahedra(); // Get the size before adding elements
    HexahedronSetTopologyModifier::addHexahedraProcess ( hexahedra );
    type::vector<core::topology::BaseMeshTopology::HexaID>& iirg = *m_DynContainer->idxInRegularGrid.beginEdit();

    std::map< unsigned int, core::topology::BaseMeshTopology::HexaID> &idrg2topo=*m_DynContainer->idInRegularGrid2IndexInTopo.beginEdit();
    for ( unsigned int i = 0; i < hexahedra.size(); i++ )  // For each element
    {
        iirg[hexaSize + i] = indices[i];
        idrg2topo.insert( std::make_pair ( indices[i], hexaSize + i ) );

        //TODO// init the values too ...
    }
    m_DynContainer->idInRegularGrid2IndexInTopo.endEdit();
    m_DynContainer->idxInRegularGrid.endEdit();
}


void DynamicSparseGridTopologyModifier::removeHexahedraProcess( const sofa::type::vector<Index> &indices, const bool removeIsolatedItems)
{
    if( !everRenumbered) renumberAttributes( indices);
    everRenumbered = false;

    HexahedronSetTopologyModifier::removeHexahedraProcess( indices, removeIsolatedItems);
}


void DynamicSparseGridTopologyModifier::renumberAttributes( const sofa::type::vector<Index> &hexahedra )
{
    type::vector<core::topology::BaseMeshTopology::HexaID>& iirg = *m_DynContainer->idxInRegularGrid.beginEdit();

    // Update the data
    unsigned int nbElt = iirg.size();
    std::map< unsigned int, core::topology::BaseMeshTopology::HexaID>& regularG2Topo = *m_DynContainer->idInRegularGrid2IndexInTopo.beginEdit();
    for ( auto it = hexahedra.begin(); it != hexahedra.end(); ++it )
    {
        nbElt--;

        // Update the voxels value
        unsigned int idHexaInRegularGrid = iirg[*it];
        (*( m_DynContainer->valuesIndexedInRegularGrid.beginEdit()))[idHexaInRegularGrid] = 0;
        m_DynContainer->valuesIndexedInRegularGrid.endEdit();

        // Renumbering the map.
        // We delete the reference of the delete elt.
        std::map< unsigned int, core::topology::BaseMeshTopology::HexaID>::iterator itMap = regularG2Topo.find( idHexaInRegularGrid);
        if( itMap != regularG2Topo.end())
        {
            regularG2Topo.erase( itMap);
        }
        // Then, we change the id of the last elt moved in the topology.
        itMap = regularG2Topo.find( iirg[nbElt]);// Index in the regular grid of the last elt in the topology
        if( itMap != regularG2Topo.end())
        {
            itMap->second = *it;
        }

        // renumber iirg
        iirg[*it] = iirg[nbElt];
    }
    iirg.resize( nbElt);

    m_DynContainer->idInRegularGrid2IndexInTopo.endEdit();
    m_DynContainer->idxInRegularGrid.endEdit();

    everRenumbered = true;
}

} // namespace sofa::component::topology::container::dynamic
