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
#ifndef SOFA_COMPONENT_TOPOLOGY_EDGESETTOPOLOGYALGORITHMS_INL
#define SOFA_COMPONENT_TOPOLOGY_EDGESETTOPOLOGYALGORITHMS_INL

#include <sofa/component/topology/EdgeSetTopology.h>
#include <sofa/component/topology/EdgeSetTopologyAlgorithms.h>
#include <algorithm>
#include <functional>

// Use BOOST GRAPH LIBRARY :

#include <boost/config.hpp>
#include <iostream>
#include <vector>
#include <utility>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/bandwidth.hpp>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;

template<class DataTypes>
void EdgeSetTopologyAlgorithms< DataTypes >::init()
{
    PointSetTopologyAlgorithms< DataTypes >::init();
    this->getContext()->get(m_container);
    this->getContext()->get(m_modifier);
    this->getContext()->get(m_geometryAlgorithms);
}

template<class DataTypes>
void EdgeSetTopologyAlgorithms< DataTypes >::removeEdges(sofa::helper::vector< unsigned int >& edges,
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
void EdgeSetTopologyAlgorithms< DataTypes >::removeItems(sofa::helper::vector< unsigned int >& items)
{
    removeEdges(items);
}

template<class DataTypes>
void  EdgeSetTopologyAlgorithms<DataTypes>::renumberPoints( const sofa::helper::vector<unsigned int> &index,
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
void EdgeSetTopologyAlgorithms< DataTypes >::addEdges(const sofa::helper::vector< Edge >& edges)
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
void EdgeSetTopologyAlgorithms< DataTypes >::addEdges(const sofa::helper::vector< Edge >& edges,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
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
    m_modifier->addEdgesWarning( edges.size(), edges, edgesIndex, ancestors, baryCoefs);

    // inform other objects that the edges are already added
    m_container->propagateTopologicalChanges();
}

template<class DataTypes>
void EdgeSetTopologyAlgorithms< DataTypes >::swapEdges(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPairs)
{
    m_modifier->swapEdgesProcess(edgesPairs);
    m_container->checkTopology();
}

template<class DataTypes>
void EdgeSetTopologyAlgorithms< DataTypes >::fuseEdges(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPairs, const bool removeIsolatedPoints)
{
    m_modifier->fuseEdgesProcess(edgesPairs, removeIsolatedPoints);
    m_container->checkTopology();
}

template<class DataTypes>
void EdgeSetTopologyAlgorithms< DataTypes >::splitEdges( sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedPoints)
{
    m_modifier->splitEdgesProcess(indices, removeIsolatedPoints);
    m_container->checkTopology();
}

template<class DataTypes>
void EdgeSetTopologyAlgorithms< DataTypes >::splitEdges( sofa::helper::vector<unsigned int> &indices,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
        const bool removeIsolatedPoints)
{
    m_modifier->splitEdgesProcess(indices, baryCoefs, removeIsolatedPoints);
    m_container->checkTopology();
}

// Give the optimal vertex permutation according to the Reverse CuthillMckee algorithm (use BOOST GRAPH LIBRAIRY)
template<class DataTypes>
void EdgeSetTopologyAlgorithms< DataTypes >::resortCuthillMckee(sofa::helper::vector<int>& inverse_permutation)
{
    using namespace boost;
    using namespace std;
    typedef adjacency_list<vecS, vecS, undirectedS,
            property<vertex_color_t, default_color_type,
            property<vertex_degree_t,int> > > Graph;
    typedef graph_traits<Graph>::vertex_descriptor Vertex;
    typedef graph_traits<Graph>::vertices_size_type size_type;

    Graph G;

    const sofa::helper::vector<Edge> &ea=m_container->getEdgeArray();

    for (unsigned int k=0; k<ea.size(); ++k)
    {
        add_edge(ea[k][0], ea[k][1], G);
    }

    inverse_permutation.resize(num_vertices(G));

    property_map<Graph, vertex_index_t>::type index_map = get(vertex_index, G);

    std::cout << "original bandwidth: " << bandwidth(G) << std::endl;

    std::vector<Vertex> inv_perm(num_vertices(G));
    std::vector<size_type> perm(num_vertices(G));

    //reverse cuthill_mckee_ordering
    cuthill_mckee_ordering(G, inv_perm.rbegin());

    //std::cout << "Reverse Cuthill-McKee ordering:" << endl;
    //std::cout << "  ";
    unsigned int ind_i = 0;
    for (std::vector<Vertex>::const_iterator it = inv_perm.begin();
            it != inv_perm.end(); ++it)
    {
        //std::cout << index_map[*it] << " ";
        inverse_permutation[ind_i++]=index_map[*it];
    }
    //std::cout << endl;

    for (size_type c=0; c!=inv_perm.size(); ++c)
        perm[index_map[inv_perm[c]]] = c;

    std::cout << "  bandwidth: "
            << bandwidth(G, make_iterator_property_map(&perm[0], index_map, perm[0]))
            << std::endl;
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_EDGESETTOPOLOGYALGORITHMS_INL
