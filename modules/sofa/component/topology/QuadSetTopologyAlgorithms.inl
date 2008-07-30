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
#ifndef SOFA_COMPONENT_TOPOLOGY_QUADSETTOPOLOGYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_QUADSETTOPOLOGYALGORITHMS_INL

#include <sofa/component/topology/QuadSetTopologyAlgorithms.h>
#include <algorithm>
#include <functional>
#include <sofa/component/topology/QuadSetTopology.h>

namespace sofa
{
namespace component
{
namespace topology
{

using namespace sofa::defaulttype;

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////QuadSetTopologyAlgorithms////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////


template<class DataTypes>
void QuadSetTopologyAlgorithms< DataTypes >::init()
{
    EdgeSetTopologyAlgorithms< DataTypes >::init();
    this->getContext()->get(m_container);
    this->getContext()->get(m_modifier);
    this->getContext()->get(m_geometryAlgorithms);
}

template<class DataTypes>
void QuadSetTopologyAlgorithms< DataTypes >::removeQuads(sofa::helper::vector< unsigned int >& quads,
        const bool removeIsolatedEdges,
        const bool removeIsolatedPoints)
{
    /// add the topological changes in the queue
    m_modifier->removeQuadsWarning(quads);
    // inform other objects that the quads are going to be removed
    m_container->propagateTopologicalChanges();
    // now destroy the old quads.
    m_modifier->removeQuadsProcess( quads, removeIsolatedEdges, removeIsolatedPoints);

    m_container->checkTopology();
}

template<class DataTypes>
void QuadSetTopologyAlgorithms< DataTypes >::removeItems(sofa::helper::vector< unsigned int >& items)
{
    removeQuads(items, true, true);
}

template<class DataTypes>
void  QuadSetTopologyAlgorithms<DataTypes>::renumberPoints( const sofa::helper::vector<unsigned int> &index,
        const sofa::helper::vector<unsigned int> &inv_index)
{
    /// add the topological changes in the queue
    m_modifier->renumberPointsWarning(index, inv_index);
    // inform other objects that the triangles are going to be removed
    m_container->propagateTopologicalChanges();
    // now renumber the points
    m_modifier->renumberPointsProcess(index, inv_index);

    m_container->checkTopology();
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_QUADSETTOPOLOGYALGORITHMS_INL
