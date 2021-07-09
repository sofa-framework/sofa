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

#include <ManifoldTopologies/ManifoldEdgeSetTopologyModifier.h>
#include <ManifoldTopologies/ManifoldEdgeSetTopologyContainer.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;
int ManifoldEdgeSetTopologyModifierClass = core::RegisterObject("ManifoldEdge set topology modifier")
        .add< ManifoldEdgeSetTopologyModifier >();

using namespace std;
using namespace sofa::defaulttype;


void ManifoldEdgeSetTopologyModifier::init()
{
    EdgeSetTopologyModifier::init();
    getContext()->get(m_container);
}

void ManifoldEdgeSetTopologyModifier::addEdgeProcess(Edge e)
{
    m_container->resetConnectedComponent(); // invalidate the connected components by default

    EdgeSetTopologyModifier::addEdgeProcess(e);
}

void ManifoldEdgeSetTopologyModifier::addEdgesProcess(const sofa::type::vector< Edge > &edges)
{
    m_container->resetConnectedComponent(); // invalidate the connected components by default

    EdgeSetTopologyModifier::addEdgesProcess(edges);
}

void ManifoldEdgeSetTopologyModifier::removeEdgesProcess(const sofa::type::vector<Index> &indices,
        const bool removeIsolatedItems)
{
    m_container->resetConnectedComponent(); // invalidate the connected components by default

    EdgeSetTopologyModifier::removeEdgesProcess(indices, removeIsolatedItems);
}

void ManifoldEdgeSetTopologyModifier::removeEdges(sofa::type::vector< Index >& edges,
        const bool removeIsolatedPoints)
{
    /// add the topological changes in the queue
    removeEdgesWarning(edges);
    // inform other objects that the edges are going to be removed
    propagateTopologicalChanges();
    // now destroy the old edges.
    removeEdgesProcess( edges, removeIsolatedPoints );

    m_container->checkTopology();
}

void ManifoldEdgeSetTopologyModifier::removeItems(sofa::type::vector< EdgeID >& items)
{
    removeEdges(items);
}

void ManifoldEdgeSetTopologyModifier::addEdges(const sofa::type::vector< Edge >& edges)
{
    const size_t nEdges = m_container->getNumberOfEdges();

    /// actually add edges in the topology container
    addEdgesProcess(edges);

    sofa::type::vector<EdgeID> edgesIndex;
    edgesIndex.reserve(edges.size());

    for (std::size_t i=0; i<edges.size(); ++i)
    {
        edgesIndex.push_back(EdgeID(nEdges+i));
    }

    // add topology event in the stack of topological events
    addEdgesWarning( edges.size(), edges, edgesIndex);

    // inform other objects that the edges are already added
    propagateTopologicalChanges();
}

void ManifoldEdgeSetTopologyModifier::addEdges(const sofa::type::vector< Edge >& edges,
        const sofa::type::vector< sofa::type::vector< Index > > & ancestors ,
        const sofa::type::vector< sofa::type::vector< double > >& baryCoefs)
{
    const size_t nEdges = m_container->getNumberOfEdges();

    /// actually add edges in the topology container
    addEdgesProcess(edges);

    sofa::type::vector<EdgeID> edgesIndex;

    for (std::size_t i=0; i<edges.size(); ++i)
    {
        edgesIndex[i] = EdgeID(nEdges + i);
    }

    // add topology event in the stack of topological events
    addEdgesWarning( edges.size(), edges,edgesIndex,ancestors,baryCoefs);

    // inform other objects that the edges are already added
    propagateTopologicalChanges();
}

void ManifoldEdgeSetTopologyModifier::swapEdges(const sofa::type::vector< sofa::type::vector< Index > >& edgesPairs)
{
    swapEdgesProcess(edgesPairs);
    m_container->checkTopology();
}


void ManifoldEdgeSetTopologyModifier::fuseEdges(const sofa::type::vector< sofa::type::vector< Index > >& edgesPairs, const bool removeIsolatedPoints)
{
    fuseEdgesProcess(edgesPairs, removeIsolatedPoints);
    m_container->checkTopology();
}

void ManifoldEdgeSetTopologyModifier::splitEdges( sofa::type::vector<Index> &indices,
        const bool removeIsolatedPoints)
{
    splitEdgesProcess(indices, removeIsolatedPoints);
    m_container->checkTopology();
}

void ManifoldEdgeSetTopologyModifier::splitEdges( sofa::type::vector<Index> &indices,
        const sofa::type::vector< sofa::type::vector< double > >& baryCoefs,
        const bool removeIsolatedPoints)
{
    splitEdgesProcess(indices, baryCoefs, removeIsolatedPoints);
    m_container->checkTopology();
}

} // namespace topology

} // namespace component

} // namespace sofa

