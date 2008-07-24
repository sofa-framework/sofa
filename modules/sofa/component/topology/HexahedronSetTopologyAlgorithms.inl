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
#ifndef SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_HEXAHEDRONSETTOPOLOGYALGORITHMS_INL

#include <sofa/component/topology/HexahedronSetTopologyAlgorithms.h>
#include <sofa/component/topology/HexahedronSetTopology.h>
#include <algorithm>
#include <functional>

namespace sofa
{
namespace component
{
namespace topology
{
using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;

template<class DataTypes>
HexahedronSetTopology< DataTypes >* HexahedronSetTopologyAlgorithms< DataTypes >::getHexahedronSetTopology() const
{
    return static_cast<HexahedronSetTopology< DataTypes >* > (this->m_basicTopology);
}

template<class DataTypes>
void HexahedronSetTopologyAlgorithms< DataTypes >::removeHexahedra(sofa::helper::vector< unsigned int >& hexahedra)
{
    HexahedronSetTopology< DataTypes > *topology = getHexahedronSetTopology();
    HexahedronSetTopologyModifier< DataTypes >* modifier  = topology->getHexahedronSetTopologyModifier();

    // add the topological changes in the queue
    modifier->removeHexahedraWarning(hexahedra);

    // inform other objects that the hexa are going to be removed
    topology->propagateTopologicalChanges();

    // now destroy the old hexahedra.
    modifier->removeHexahedraProcess(  hexahedra ,true);

    topology->getHexahedronSetTopologyContainer()->checkTopology();
}

template<class DataTypes>
void HexahedronSetTopologyAlgorithms< DataTypes >::removeItems(sofa::helper::vector< unsigned int >& items)
{
    removeHexahedra(items);
}

template<class DataTypes>
void HexahedronSetTopologyAlgorithms< DataTypes >::writeMSH(const char *filename)
{
    HexahedronSetTopology< DataTypes > *topology = getHexahedronSetTopology();
    HexahedronSetTopologyModifier< DataTypes >* modifier  = topology->getHexahedronSetTopologyModifier();

    modifier->writeMSHfile(filename);
}

template<class DataTypes>
void  HexahedronSetTopologyAlgorithms<DataTypes>::renumberPoints(const sofa::helper::vector<unsigned int> &index,
        const sofa::helper::vector<unsigned int> &inv_index)
{
    HexahedronSetTopology< DataTypes > *topology = getHexahedronSetTopology();
    HexahedronSetTopologyModifier< DataTypes >* modifier  = topology->getHexahedronSetTopologyModifier();

    /// add the topological changes in the queue
    modifier->renumberPointsWarning(index, inv_index);
    // inform other objects that the triangles are going to be removed
    topology->propagateTopologicalChanges();
    // now renumber the points
    modifier->renumberPointsProcess(index, inv_index);

    topology->getHexahedronSetTopologyContainer()->checkTopology();
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_HexahedronSetTOPOLOGY_INL
