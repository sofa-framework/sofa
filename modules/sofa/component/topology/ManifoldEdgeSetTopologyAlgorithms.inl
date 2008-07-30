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
#ifndef SOFA_COMPONENT_TOPOLOGY_MANIFOLDEDGESETTOPOLOGYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_MANIFOLDEDGESETTOPOLOGYALGORITHMS_INL

#include <sofa/component/topology/ManifoldEdgeSetTopologyAlgorithms.h>
#include <algorithm>
#include <functional>
#include <sofa/component/topology/ManifoldEdgeSetTopology.h>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;

template<class DataTypes>
void ManifoldEdgeSetTopologyAlgorithms< DataTypes >::init()
{
    EdgeSetTopologyAlgorithms< DataTypes >::init();
    this->getContext()->get(m_container);
    this->getContext()->get(m_modifier);
    this->getContext()->get(m_geometryAlgorithms);
}

template<class DataTypes>
void ManifoldEdgeSetTopologyAlgorithms< DataTypes >::removeEdges(sofa::helper::vector< unsigned int >& edges,
        const bool removeIsolatedPoints)
{
    /// add the topological changes in the queue
    m_modifier->removeEdgesWarning(edges);
    // inform other objects that the edges are going to be removed
    m_container->propagateTopologicalChanges();
    // now destroy the old edges.
    m_modifier->removeEdgesProcess( edges, removeIsolatedPoints );

    m_container->checkTopology();
}

template<class DataTypes>
void ManifoldEdgeSetTopologyAlgorithms< DataTypes >::removeItems(sofa::helper::vector< unsigned int >& items)
{
    removeEdges(items);
}

template<class DataTypes>
void  ManifoldEdgeSetTopologyAlgorithms<DataTypes>::renumberPoints( const sofa::helper::vector<unsigned int> &index,
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

template<class DataTypes>
void ManifoldEdgeSetTopologyAlgorithms< DataTypes >::addEdges(const sofa::helper::vector< Edge >& edges)
{
    unsigned int nEdges = m_container->getNumberOfEdges();

    /// actually add edges in the topology container
    m_modifier->addEdgesProcess(edges);

    sofa::helper::vector<unsigned int> edgesIndex;
    edgesIndex.reserve(edges.size());

    for (unsigned int i=0; i<edges.size(); ++i)
    {
        edgesIndex.push_back(nEdges+i);
    }

    // add topology event in the stack of topological events
    m_modifier->addEdgesWarning( edges.size(), edges, edgesIndex);

    // inform other objects that the edges are already added
    m_container->propagateTopologicalChanges();
}

template<class DataTypes>
void ManifoldEdgeSetTopologyAlgorithms< DataTypes >::addEdges(const sofa::helper::vector< Edge >& edges,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors ,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    unsigned int nEdges = m_container->getNumberOfEdges();

    /// actually add edges in the topology container
    m_modifier->addEdgesProcess(edges);

    sofa::helper::vector<unsigned int> edgesIndex;

    for (unsigned int i=0; i<edges.size(); ++i)
    {
        edgesIndex[i]=nEdges+i;
    }

    // add topology event in the stack of topological events
    m_modifier->addEdgesWarning( edges.size(), edges,edgesIndex,ancestors,baryCoefs);

    // inform other objects that the edges are already added
    m_container->propagateTopologicalChanges();
}

template<class DataTypes>
void ManifoldEdgeSetTopologyAlgorithms< DataTypes >::swapEdges(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPairs)
{
    m_modifier->swapEdgesProcess(edgesPairs);
    m_container->checkTopology();
}


template<class DataTypes>
void ManifoldEdgeSetTopologyAlgorithms< DataTypes >::fuseEdges(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPairs, const bool removeIsolatedPoints)
{
    m_modifier->fuseEdgesProcess(edgesPairs, removeIsolatedPoints);
    m_container->checkTopology();
}

template<class DataTypes>
void ManifoldEdgeSetTopologyAlgorithms< DataTypes >::splitEdges( sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedPoints)
{
    m_modifier->splitEdgesProcess(indices, removeIsolatedPoints);
    m_container->checkTopology();
}

template<class DataTypes>
void ManifoldEdgeSetTopologyAlgorithms< DataTypes >::splitEdges( sofa::helper::vector<unsigned int> &indices,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
        const bool removeIsolatedPoints)
{
    m_modifier->splitEdgesProcess(indices, baryCoefs, removeIsolatedPoints);
    m_container->checkTopology();
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_MANIFOLDEDGESETTOPOLOGYALGORITHMS_INL
