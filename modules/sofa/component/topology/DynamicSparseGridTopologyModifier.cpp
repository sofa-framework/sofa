/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/topology/DynamicSparseGridTopologyModifier.h>
#include <sofa/component/topology/HexahedronSetTopologyChange.h>
#include <sofa/component/topology/DynamicSparseGridTopologyContainer.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{
namespace component
{
namespace topology
{
SOFA_DECL_CLASS ( DynamicSparseGridTopologyModifier );
int DynamicSparseGridTopologyModifierClass = core::RegisterObject ( "Hexahedron set topology modifier" )
        .add< DynamicSparseGridTopologyModifier >();



void DynamicSparseGridTopologyModifier::init()
{
    HexahedronSetTopologyModifier::init();
    this->getContext()->get ( m_DynContainer );
    if ( ! m_DynContainer )
    {
        std::cerr << "ERROR in DynamicSparseGridTopologyModifier::init(): DynamicSparseGridTopologyContainer was not found !" << std::endl;
    }
}



//TODO// find a solution for this case !!!! Modifier can not access to the DOF and can not compute the indices of the added hexas.
// We have to find a way to automaticaly compute the indices of the added hexas to update the map 'm_m_DynContainer->idInRegularGrid2Hexa'
void DynamicSparseGridTopologyModifier::addHexahedraProcess ( const sofa::helper::vector< Hexahedron > &hexahedra )
{
    HexahedronSetTopologyModifier::addHexahedraProcess ( hexahedra );
    serr << "DynamicSparseGridTopologyModifier::addHexahedraProcess( const sofa::helper::vector< Hexahedron > &hexahedra ). You must not use this method. To add some voxels to the topology, you must use addHexahedraProcess ( const sofa::helper::vector< Hexahedron > &hexahedra, const sofa::helper::vector< unsigned int> &indices ) because, for the moment, indices maps can not be updated !" << sendl;
}


void DynamicSparseGridTopologyModifier::addHexahedraProcess ( const sofa::helper::vector< Hexahedron > &hexahedra, const sofa::helper::vector< unsigned int> &indices )
{
    assert( hexahedra.size() == indices.size());

    unsigned int hexaSize = m_DynContainer->getNumberOfHexahedra(); // Get the size before adding elements
    HexahedronSetTopologyModifier::addHexahedraProcess ( hexahedra );
    helper::vector<BaseMeshTopology::HexaID>& iirg = *(m_DynContainer->idxInRegularGrid.beginEdit());

    for ( unsigned int i = 0; i < hexahedra.size(); i++ )  // For each element
    {
        iirg[hexaSize + i] = indices[i];
        m_DynContainer->idInRegularGrid2IndexInTopo.insert( std::make_pair ( indices[i], hexaSize + i ) );
    }
    m_DynContainer->idxInRegularGrid.endEdit();
//TODO// ca suffit pas. il faut gerer le renumbering pour le vector et la map du container ! A refaire !
}

void DynamicSparseGridTopologyModifier::removeHexahedraProcess ( const sofa::helper::vector<unsigned int> &indices, const bool removeIsolatedItems )
{
    HexahedronSetTopologyModifier::removeHexahedraProcess ( indices, removeIsolatedItems );

    // Update the map idInRegularGrid2Hexa.
    for ( unsigned int i = 0; i < indices.size(); i++ )  // For each element
    {
        m_DynContainer->idInRegularGrid2IndexInTopo.erase ( indices[i] );
//TODO// ca suffit pas. il faut gerer le renumbering pour le vector et la map du container ! A refaire !
    }
}

void DynamicSparseGridTopologyModifier::removeHexahedraWarning ( sofa::helper::vector<unsigned int> &hexahedra )
{
    HexahedronSetTopologyModifier::removeHexahedraWarning ( hexahedra );
    const helper::vector<BaseMeshTopology::HexaID>& iirg = *(m_DynContainer->idxInRegularGrid.beginEdit());

    // Update the data
    unsigned int nbElt = iirg.size();
    sofa::helper::vector<unsigned int> vecHexaRemoved = hexahedra; // Is indices ever sorted?
    sort ( vecHexaRemoved.begin(), vecHexaRemoved.end() );
    for ( sofa::helper::vector<unsigned int>::const_reverse_iterator it ( vecHexaRemoved.end() ); it != sofa::helper::vector<unsigned int>::const_reverse_iterator ( vecHexaRemoved.begin() ); it++ )
    {
        nbElt--;

        // Update the voxels value
        unsigned int idHexa = iirg[*it];
        m_DynContainer->valuesIndexedInRegularGrid[idHexa] = 0;

        // Update the indices
        //iirg[*it] = iirg[ nbElt ];
    }
    //iirg.resize ( nbElt );
    m_DynContainer->idxInRegularGrid.endEdit();
//TODO// ca suffit pas. il faut gerer le renumbering pour le vector et la map du container ! A refaire !
}

} // namespace topology

} // namespace component

} // namespace sofa

