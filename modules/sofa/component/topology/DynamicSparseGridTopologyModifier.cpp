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
    this->getContext()->get ( m_container );
    if ( ! m_container )
    {
        std::cerr << "ERROR in DynamicSparseGridTopologyModifier::init(): DynamicSparseGridTopologyContainer was not found !" << std::endl;
    }
}



void DynamicSparseGridTopologyModifier::addHexahedraProcess ( const sofa::helper::vector< Hexahedron > &hexahedra )
{
//        unsigned int hexaSize = m_container->getNumberOfHexahedra(); // Get the size before adding elements
    HexahedronSetTopologyModifier::addHexahedraProcess ( hexahedra );
    /*
            for ( unsigned int i = 0; i < hexahedra.size(); i++ )  // For each element
            {
              // Compute the center
              const Hexahedron &t = m_container->getHexa( hexaSize + i);
              const typename DataTypes::VecCoord& p = *(this->object->getX());

              DataTypes::Coord coord = (p[t[0]] + p[t[1]] + p[t[2]] + p[t[3]] + p[t[4]] + p[t[5]] + p[t[6]] + p[t[7]]) * (Real) 0.125;

              // Check if the hexa is in the boundary.
              assert( true); // assert dans les limites definies.

              // Compute the index
              unsigned int index = 0; //TODO// coord * sizeDim

              // update the map idInRegularGrid2Hexa.
              m_container->idInRegularGrid2Hexa.insert ( std::make_pair ( index, hexaSize + i ) );
            }*/
}

void DynamicSparseGridTopologyModifier::removeHexahedraProcess ( const sofa::helper::vector<unsigned int> &indices, const bool removeIsolatedItems )
{
    HexahedronSetTopologyModifier::removeHexahedraProcess ( indices, removeIsolatedItems );

    /*
            //TODO// Update the map idInRegularGrid2Hexa.
            for ( unsigned int i = 0; i < indices.size(); i++ )  // For each element
            {
    //TODO// verifier s'il y a pas un erase( iterator = find()) pour eviter les surprises.
              m_container->idInRegularGrid2Hexa.erase( indices[i]);
            }
    */
}

void DynamicSparseGridTopologyModifier::removeHexahedraWarning ( sofa::helper::vector<unsigned int> &hexahedra )
{
    HexahedronSetTopologyModifier::removeHexahedraWarning( hexahedra );

    // Update the data
    unsigned int nbElt = m_container->idxInRegularGrid.size();
    sofa::helper::vector<unsigned int> vecHexaRemoved = hexahedra; // Is indices ever sorted?
    sort ( vecHexaRemoved.begin(), vecHexaRemoved.end() );
    for ( sofa::helper::vector<unsigned int>::const_reverse_iterator it ( vecHexaRemoved.end() ); it != sofa::helper::vector<unsigned int>::const_reverse_iterator ( vecHexaRemoved.begin() ); it++ )
    {
        nbElt--;

        // Update the voxels value
        unsigned int idHexa = m_container->idxInRegularGrid[*it];
        m_container->valuesIndexedInRegularGrid[idHexa] = 0;

        // Update the indices
        m_container->idxInRegularGrid[*it] = m_container->idxInRegularGrid[ nbElt ];
    }
    m_container->idxInRegularGrid.resize ( nbElt );
}

} // namespace topology

} // namespace component

} // namespace sofa

