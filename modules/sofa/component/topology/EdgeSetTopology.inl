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
#ifndef SOFA_COMPONENT_TOPOLOGY_EDGESETTOPOLOGY_INL
#define SOFA_COMPONENT_TOPOLOGY_EDGESETTOPOLOGY_INL

#include <sofa/component/topology/EdgeSetTopology.h>
#include <sofa/component/topology/PointSetTopology.inl>
#include <sofa/component/topology/TopologyChangedEvent.h>
#include <sofa/simulation/common/PropagateEventVisitor.h>
#include <sofa/simulation/tree/GNode.h>
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

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////EdgeSetTopology//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template<class DataTypes>
EdgeSetTopology<DataTypes>::EdgeSetTopology(MechanicalObject<DataTypes> *obj)
    : PointSetTopology<DataTypes>( obj)
{
}

template<class DataTypes>
void EdgeSetTopology<DataTypes>::createComponents()
{
    this->m_topologyContainer = new EdgeSetTopologyContainer(this);
    this->m_topologyModifier= new EdgeSetTopologyModifier<DataTypes>(this);
    this->m_topologyAlgorithms= new EdgeSetTopologyAlgorithms<DataTypes>(this);
    this->m_geometryAlgorithms= new EdgeSetGeometryAlgorithms<DataTypes>(this);
}

template<class DataTypes>
void EdgeSetTopology<DataTypes>::init()
{
    PointSetTopology<DataTypes>::init();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////EdgeSetTopologyAlgorithms////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template<class DataTypes>
void EdgeSetTopologyAlgorithms< DataTypes >::removeEdges(sofa::helper::vector< unsigned int >& edges,
        const bool removeIsolatedPoints)
{
    EdgeSetTopology< DataTypes > *topology = getEdgeSetTopology();
    EdgeSetTopologyModifier< DataTypes >* modifier  = topology->getEdgeSetTopologyModifier();

    /// add the topological changes in the queue
    modifier->removeEdgesWarning(edges);
    // inform other objects that the edges are going to be removed
    topology->propagateTopologicalChanges();
    // now destroy the old edges.
    modifier->removeEdgesProcess( edges, removeIsolatedPoints );

    topology->getEdgeSetTopologyContainer()->checkTopology();
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
    EdgeSetTopology< DataTypes > *topology = getEdgeSetTopology();
    EdgeSetTopologyModifier< DataTypes >* modifier  = topology->getEdgeSetTopologyModifier();

    /// add the topological changes in the queue
    modifier->renumberPointsWarning(index, inv_index);
    // inform other objects that the triangles are going to be removed
    topology->propagateTopologicalChanges();
    // now renumber the points
    modifier->renumberPointsProcess(index, inv_index);

    topology->getEdgeSetTopologyContainer()->checkTopology();
}

template<class DataTypes>
void EdgeSetTopologyAlgorithms< DataTypes >::addEdges(const sofa::helper::vector< Edge >& edges)
{
    EdgeSetTopology< DataTypes > *topology = getEdgeSetTopology();
    EdgeSetTopologyModifier< DataTypes >* modifier  = topology->getEdgeSetTopologyModifier();

    unsigned int nEdges = topology->getEdgeSetTopologyContainer()->getNumberOfEdges();

    /// actually add edges in the topology container
    modifier->addEdgesProcess(edges);

    sofa::helper::vector<unsigned int> edgesIndex;
    edgesIndex.reserve(edges.size());

    for (unsigned int i=0; i<edges.size(); ++i)
    {
        edgesIndex.push_back(nEdges+i);
    }

    // add topology event in the stack of topological events
    modifier->addEdgesWarning( edges.size(), edges, edgesIndex);

    // inform other objects that the edges are already added
    topology->propagateTopologicalChanges();
}

template<class DataTypes>
void EdgeSetTopologyAlgorithms< DataTypes >::addEdges(const sofa::helper::vector< Edge >& edges,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    EdgeSetTopology< DataTypes > *topology = getEdgeSetTopology();
    EdgeSetTopologyModifier< DataTypes >* modifier  = topology->getEdgeSetTopologyModifier();

    unsigned int nEdges = topology->getEdgeSetTopologyContainer()->getNumberOfEdges();

    /// actually add edges in the topology container
    modifier->addEdgesProcess(edges);

    sofa::helper::vector<unsigned int> edgesIndex;
    edgesIndex.reserve(edges.size());

    for (unsigned int i=0; i<edges.size(); ++i)
    {
        edgesIndex.push_back(nEdges+i);
    }

    // add topology event in the stack of topological events
    modifier->addEdgesWarning( edges.size(), edges, edgesIndex, ancestors, baryCoefs);

    // inform other objects that the edges are already added
    topology->propagateTopologicalChanges();
}

template<class DataTypes>
void EdgeSetTopologyAlgorithms< DataTypes >::swapEdges(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPairs)
{
    EdgeSetTopology< DataTypes > *topology = getEdgeSetTopology();

    topology->getEdgeSetTopologyModifier()->swapEdgesProcess(edgesPairs);
    topology->getEdgeSetTopologyContainer()->checkTopology();
}

template<class DataTypes>
void EdgeSetTopologyAlgorithms< DataTypes >::fuseEdges(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPairs, const bool removeIsolatedPoints)
{
    EdgeSetTopology< DataTypes > *topology = getEdgeSetTopology();

    topology->getEdgeSetTopologyModifier()->fuseEdgesProcess(edgesPairs, removeIsolatedPoints);
    topology->getEdgeSetTopologyContainer()->checkTopology();
}

template<class DataTypes>
void EdgeSetTopologyAlgorithms< DataTypes >::splitEdges( sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedPoints)
{
    EdgeSetTopology< DataTypes > *topology = getEdgeSetTopology();

    topology->getEdgeSetTopologyModifier()->splitEdgesProcess(indices, removeIsolatedPoints);
    topology->getEdgeSetTopologyContainer()->checkTopology();
}

template<class DataTypes>
void EdgeSetTopologyAlgorithms< DataTypes >::splitEdges( sofa::helper::vector<unsigned int> &indices,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
        const bool removeIsolatedPoints)
{
    EdgeSetTopology< DataTypes > *topology = getEdgeSetTopology();

    topology->getEdgeSetTopologyModifier()->splitEdgesProcess(indices, baryCoefs, removeIsolatedPoints);
    topology->getEdgeSetTopologyContainer()->checkTopology();
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

    EdgeSetTopology< DataTypes > *topology = getEdgeSetTopology();
    EdgeSetTopologyContainer *container = topology->getEdgeSetTopologyContainer();

    const sofa::helper::vector<Edge> &ea=container->getEdgeArray();

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

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////EdgeSetGeometryAlgorithms////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template< class DataTypes>
typename DataTypes::Real EdgeSetGeometryAlgorithms< DataTypes >::computeEdgeLength( const unsigned int i) const
{
    EdgeSetTopology< DataTypes > *topology = getEdgeSetTopology();
    EdgeSetTopologyContainer *container = topology->getEdgeSetTopologyContainer();

    const Edge &e = container->getEdge(i);
    const VecCoord& p = *topology->getDOF()->getX();
    const Real length = (p[e[0]]-p[e[1]]).norm();
    return length;
}

template< class DataTypes>
typename DataTypes::Real EdgeSetGeometryAlgorithms< DataTypes >::computeRestEdgeLength( const unsigned int i) const
{
    EdgeSetTopology< DataTypes > *topology = getEdgeSetTopology();
    EdgeSetTopologyContainer *container = topology->getEdgeSetTopologyContainer();

    const Edge &e = container->getEdge(i);
    const VecCoord& p = *topology->getDOF()->getX0();
    const Real length = (p[e[0]]-p[e[1]]).norm();
    return length;
}

template< class DataTypes>
typename DataTypes::Real EdgeSetGeometryAlgorithms< DataTypes >::computeRestSquareEdgeLength( const unsigned int i) const
{
    EdgeSetTopology< DataTypes > *topology = getEdgeSetTopology();
    EdgeSetTopologyContainer *container = topology->getEdgeSetTopologyContainer();

    const Edge &e = container->getEdge(i);
    const VecCoord& p = *topology->getDOF()->getX0();
    const Real length = (p[e[0]]-p[e[1]]).norm2();
    return length;
}

/// computes the edge length of all edges are store in the array interface
template<class DataTypes>
void EdgeSetGeometryAlgorithms<DataTypes>::computeEdgeLength( BasicArrayInterface<Real> &ai) const
{
    EdgeSetTopology< DataTypes > *topology = getEdgeSetTopology();
    EdgeSetTopologyContainer *container = topology->getEdgeSetTopologyContainer();

    const sofa::helper::vector<Edge> &ea = container->getEdgeArray();
    const typename DataTypes::VecCoord& p = *topology->getDOF()->getX();

    for (unsigned int i=0; i<ea.size(); ++i)
    {
        const Edge &e = ea[i];
        ai[i] = (p[e[0]]-p[e[1]]).norm();
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////EdgeSetTopologyModifier//////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

template<class DataTypes>
class EdgeSetTopologyLoader : public PointSetTopologyLoader<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;

    EdgeSetTopologyLoader(EdgeSetTopologyModifier<DataTypes> *tm)
        : PointSetTopologyLoader<DataTypes>(),
          estm(tm)
    {}

    virtual void addLine(int p1, int p2)
    {
        estm->addEdge(Edge((unsigned int)p1,(unsigned int)p2));
    }

public:
    VecCoord pointArray;
    EdgeSetTopologyModifier<DataTypes> *estm;
};

template<class DataTypes>
bool EdgeSetTopologyModifier<DataTypes>::load(const char *filename)
{
    EdgeSetTopologyLoader<DataTypes> loader(this);
    if (!loader.load(filename))
        return false;
    else
    {
        loadPointSet(&loader);
        return true;
    }
}

template<class DataTypes>
void EdgeSetTopologyModifier<DataTypes>::addEdge(Edge e)
{
    EdgeSetTopologyContainer* container = getEdgeSetTopology()->getEdgeSetTopologyContainer();

#ifndef NDEBUG
    // check if the 2 vertices are different
    if(e[0] == e[1])
    {
        cout << "Error: [EdgeSetTopologyModifier::addEdge] : invalid edge: "
                << e[0] << ", " << e[1] << endl;

        return;
    }

    // check if there already exists an edge.
    // Important: getEdgeIndex creates the edge vertex shell array
    if(container->hasEdgeVertexShell())
    {
        if(container->getEdgeIndex(e[0],e[1]) != -1)
        {
            cout << "Error: [EdgeSetTopologyModifier::addEdgesProcess] : Edge "
                    << e[0] << ", " << e[1] << " already exists." << endl;
            return;
        }
    }
#endif
    if (container->hasEdgeVertexShell())
    {
        const unsigned int edgeId = container->m_edge.size();

        sofa::helper::vector< unsigned int > &shell0 = container->getEdgeVertexShellForModification( e[0] );
        shell0.push_back(edgeId);
        sort(shell0.begin(), shell0.end());

        sofa::helper::vector< unsigned int > &shell1 = container->getEdgeVertexShellForModification( e[1] );
        shell1.push_back(edgeId);
        sort(shell1.begin(), shell1.end());
    }

    container->m_edge.push_back(e);
}

template<class DataTypes>
void EdgeSetTopologyModifier<DataTypes>::addEdgesProcess(const sofa::helper::vector< Edge > &edges)
{
    EdgeSetTopologyContainer* container = getEdgeSetTopology()->getEdgeSetTopologyContainer();
    container->m_edge.reserve(container->m_edge.size() + edges.size());
    for (unsigned int i=0; i<edges.size(); ++i)
    {
        addEdge(edges[i]);
    }
}

template<class DataTypes>
void EdgeSetTopologyModifier<DataTypes>::addEdgesWarning(const unsigned int nEdges)
{
    // Warning that edges just got created
    EdgesAdded *e = new EdgesAdded(nEdges);
    this->addTopologyChange(e);
}

template<class DataTypes>
void EdgeSetTopologyModifier<DataTypes>::addEdgesWarning(const unsigned int nEdges,
        const sofa::helper::vector< Edge >& edgesList,
        const sofa::helper::vector< unsigned int >& edgesIndexList)
{
    // Warning that edges just got created
    EdgesAdded *e = new EdgesAdded(nEdges, edgesList, edgesIndexList);
    this->addTopologyChange(e);
}

template<class DataTypes>
void EdgeSetTopologyModifier<DataTypes>::addEdgesWarning(const unsigned int nEdges,
        const sofa::helper::vector< Edge >& edgesList,
        const sofa::helper::vector< unsigned int >& edgesIndexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors)
{
    // Warning that edges just got created
    EdgesAdded *e = new EdgesAdded(nEdges, edgesList, edgesIndexList, ancestors);
    this->addTopologyChange(e);
}

template<class DataTypes>
void EdgeSetTopologyModifier<DataTypes>::addEdgesWarning(const unsigned int nEdges,
        const sofa::helper::vector< Edge >& edgesList,
        const sofa::helper::vector< unsigned int >& edgesIndexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    // Warning that edges just got created
    EdgesAdded *e = new EdgesAdded(nEdges, edgesList, edgesIndexList, ancestors, baryCoefs);
    this->addTopologyChange(e);
}

template<class DataTypes>
void EdgeSetTopologyModifier<DataTypes>::removeEdgesWarning(sofa::helper::vector<unsigned int> &edges )
{
    // sort vertices to remove in a descendent order
    // TODO: clarify why sorting is necessary
    std::sort( edges.begin(), edges.end(), std::greater<unsigned int>() );

    // Warning that these edges will be deleted
    EdgesRemoved *e = new EdgesRemoved(edges);
    this->addTopologyChange(e);
}

template<class DataTypes>
void EdgeSetTopologyModifier<DataTypes>::removeEdgesProcess(const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedItems)
{
    EdgeSetTopologyContainer * container = getEdgeSetTopology()->getEdgeSetTopologyContainer();

    if(!container->hasEdges())	// TODO : this method should only be called when edges exist
    {
        cout << "Error. [EdgeSetTopologyModifier::removeEdgesProcess] edge array is empty." << endl;
        return;
    }

    sofa::helper::vector<unsigned int> vertexToBeRemoved;

    for (unsigned int i=0; i<indices.size(); ++i)
    {
        const Edge &e = container->m_edge[ indices[i] ];
        const unsigned int point0 = e[0], point1 = e[1];

        // first check that the edge shell array has been initialized
        if(!container->hasEdgeVertexShell())
        {
            container->createEdgeVertexShellArray();
        }

        sofa::helper::vector< unsigned int > &shell0 = container->m_edgeVertexShell[ point0 ];
        // removes the first occurence (should be the only one) of the edge in the edge shell of the point
        //assert(std::find( shell0.begin(), shell0.end(), indices[i] ) !=shell0.end());
        shell0.erase( std::find( shell0.begin(), shell0.end(), indices[i] ) );
        if (removeIsolatedItems && shell0.empty())
        {
            vertexToBeRemoved.push_back(point0);
        }

        sofa::helper::vector< unsigned int > &shell1 = container->m_edgeVertexShell[ point1 ];
        // removes the first occurence (should be the only one) of the edge in the edge shell of the other point
        //assert(std::find( shell1.begin(), shell1.end(), indices[i] ) !=shell1.end());
        shell1.erase( std::find( shell1.begin(), shell1.end(), indices[i] ) );
        if (removeIsolatedItems && shell1.empty())
        {
            vertexToBeRemoved.push_back(point1);
        }

        // removes the edge from the edgelist
        container->m_edge[ indices[i] ] = container->m_edge[ container->m_edge.size() - 1 ]; // overwriting with last valid value.

        // now updates the shell information of the edge formely at the end of the array
        unsigned int lastEdgeIndex = container->m_edge.size() - 1;
        const Edge &e1 = container->m_edge[ indices[i] ];
        const unsigned int point3 = e1[0], point4 = e1[1];

        //replaces the edge index oldEdgeIndex with indices[i] for the first vertex
        // TODO: clarify if vector<set<int>> would not be better for shells - no duplicates and no sorting
        sofa::helper::vector< unsigned int > &shell3 = container->m_edgeVertexShell[ point3 ];
        replace(shell3.begin(), shell3.end(), lastEdgeIndex, indices[i]);
        sort(shell3.begin(), shell3.end());

        //replaces the edge index oldEdgeIndex with indices[i] for the second vertex
        sofa::helper::vector< unsigned int > &shell4 = container->m_edgeVertexShell[ point4 ];
        replace(shell4.begin(), shell4.end(), lastEdgeIndex, indices[i]);
        sort(shell4.begin(), shell4.end());

        container->m_edge.resize( container->m_edge.size() - 1 ); // resizing to erase multiple occurence of the edge.
    }

    if (! vertexToBeRemoved.empty())
    {
        this->removePointsWarning(vertexToBeRemoved);
        // inform other objects that the points are going to be removed
        getEdgeSetTopology()->propagateTopologicalChanges();
        this->removePointsProcess(vertexToBeRemoved);
    }
}

template<class DataTypes >
void EdgeSetTopologyModifier< DataTypes >::addPointsProcess(const unsigned int nPoints,
        const bool addDOF)
{
    // now update the local container structures.
    EdgeSetTopologyContainer * container = getEdgeSetTopology()->getEdgeSetTopologyContainer();

    if(container->hasEdgeVertexShell())
        container->m_edgeVertexShell.resize( container->getNumberOfVertices() + nPoints );

    // call the PointSet method.
    PointSetTopologyModifier< DataTypes >::addPointsProcess( nPoints, addDOF );
}

template<class DataTypes >
void EdgeSetTopologyModifier< DataTypes >::addPointsProcess(const unsigned int nPoints,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
        const bool addDOF)
{
    // now update the local container structures.
    EdgeSetTopologyContainer * container = getEdgeSetTopology()->getEdgeSetTopologyContainer();

    if(container->hasEdgeVertexShell())
        container->m_edgeVertexShell.resize( container->getNumberOfVertices() + nPoints );

    // call the PointSet method.
    PointSetTopologyModifier< DataTypes >::addPointsProcess( nPoints, ancestors, baryCoefs, addDOF );
}

template<class DataTypes >
void EdgeSetTopologyModifier< DataTypes >::addNewPoint(unsigned int i, const sofa::helper::vector< double >& x)
{
    // now update the local container structures.
    EdgeSetTopologyContainer * container = getEdgeSetTopology()->getEdgeSetTopologyContainer();

    if(container->hasEdgeVertexShell())
        container->m_edgeVertexShell.resize( i+1 );

    // call the PointSet method.
    PointSetTopologyModifier< DataTypes >::addNewPoint(i, x);
}

template< class DataTypes >
void EdgeSetTopologyModifier< DataTypes >::removePointsProcess(sofa::helper::vector<unsigned int> &indices,
        const bool removeDOF)
{
    EdgeSetTopologyContainer * container = getEdgeSetTopology()->getEdgeSetTopologyContainer();

    if(!container->hasEdges())	// TODO : this method should only be called when edges exist
    {
        cout << "Error. [EdgeSetTopologyModifier::removePointsProcess] edge array is empty." << endl;

        PointSetTopologyModifier< DataTypes >::removePointsProcess( indices, removeDOF );
        return;
    }

    // forces the construction of the edge shell array if it does not exists
    if(container->hasEdgeVertexShell())
        container->createEdgeVertexShellArray();

    // TODO: remove edges connected to the points being removed (or make sure it cannot occur)

    unsigned int lastPoint = container->getNumberOfVertices() - 1;
    for (unsigned int i=0; i<indices.size(); ++i, --lastPoint)
    {
        // updating the edges connected to the point replacing the removed one:
        // for all edges connected to the last point
        for (unsigned int j = 0; j < container->m_edgeVertexShell[lastPoint].size(); ++j)
        {
            // change the old index for the new one
            if ( container->m_edge[ container->m_edgeVertexShell[lastPoint][j] ][0] == lastPoint )
                container->m_edge[ container->m_edgeVertexShell[lastPoint][j] ][0] = indices[i];
            else
                container->m_edge[ container->m_edgeVertexShell[lastPoint][j] ][1] = indices[i];
        }

        // updating the edge shell itself (change the old index for the new one)
        container->m_edgeVertexShell[ indices[i] ] = container->m_edgeVertexShell[ lastPoint ];
    }

    container->m_edgeVertexShell.resize( container->m_edgeVertexShell.size() - indices.size() );

    // call the point set method.
    // Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    PointSetTopologyModifier< DataTypes >::removePointsProcess( indices, removeDOF );
}

template< class DataTypes >
void EdgeSetTopologyModifier< DataTypes >::renumberPointsProcess( const sofa::helper::vector<unsigned int> &index,
        const sofa::helper::vector<unsigned int> &inv_index,
        const bool renumberDOF)
{
    EdgeSetTopologyContainer * container = getEdgeSetTopology()->getEdgeSetTopologyContainer();

    if(!container->hasEdges())	// TODO : this method should only be called when edges exist
    {
        cout << "Error. [EdgeSetTopologyModifier::renumberPointsProcess] edge array is empty." << endl;

        // call the point set method
        PointSetTopologyModifier< DataTypes >::renumberPointsProcess( index, inv_index, renumberDOF );
        return;
    }

    if(container->hasEdgeVertexShell())
    {
        // copy of the the edge vertex shell array
        sofa::helper::vector< sofa::helper::vector< unsigned int > > edgeVertexShell_cp = container->getEdgeVertexShellArray();

        for (unsigned int i=0; i<index.size(); ++i)
        {
            container->m_edgeVertexShell[i] = edgeVertexShell_cp[ index[i] ];
        }
    }

    for (unsigned int i=0; i<container->m_edge.size(); ++i)
    {
        const unsigned int p0 = inv_index[ container->m_edge[i][0]  ];
        const unsigned int p1 = inv_index[ container->m_edge[i][1]  ];

        if(p0<p1)
        {
            container->m_edge[i][0] = p0;
            container->m_edge[i][1] = p1;
        }
        else
        {
            container->m_edge[i][0] = p1;
            container->m_edge[i][1] = p0;
        }
    }

    // call the point set method
    PointSetTopologyModifier< DataTypes >::renumberPointsProcess( index, inv_index, renumberDOF );
}

template<class DataTypes>
void EdgeSetTopologyModifier< DataTypes >::swapEdgesProcess(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPairs)
{
    EdgeSetTopology< DataTypes > *topology = getEdgeSetTopology();
    EdgeSetTopologyContainer * container = topology->getEdgeSetTopologyContainer();
    EdgeSetTopologyModifier< DataTypes >* modifier = this;

    // first create the edges
    sofa::helper::vector< Edge > v;
    v.reserve(2*edgesPairs.size());

    sofa::helper::vector< unsigned int > edgeIndexList;
    edgeIndexList.reserve(2*edgesPairs.size());

    sofa::helper::vector<sofa::helper::vector<unsigned int> > ancestorsArray;
    ancestorsArray.reserve(edgesPairs.size());

    unsigned int nbEdges=container->getNumberOfEdges();

    for (unsigned int i=0; i<edgesPairs.size(); ++i)
    {
        const unsigned int i1 = edgesPairs[i][0];
        const unsigned int i2 = edgesPairs[i][1];

        const unsigned int p11 = container->getEdge(i1)[0];
        const unsigned int p12 = container->getEdge(i1)[1];
        const unsigned int p21 = container->getEdge(i2)[0];
        const unsigned int p22 = container->getEdge(i2)[1];

        const Edge e1(p11, p21), e2(p12, p22);

        v.push_back(e1);
        v.push_back(e2);
        edgeIndexList.push_back(nbEdges);
        edgeIndexList.push_back(nbEdges+1);
        nbEdges += 2;

        sofa::helper::vector<unsigned int> ancestors(2);
        ancestors[0] = i1;
        ancestors[1] = i2;
        ancestorsArray.push_back(ancestors);
    }

    modifier->addEdgesProcess( v );

    // now warn about the creation
    modifier->addEdgesWarning( v.size(), v, edgeIndexList, ancestorsArray);

    //   EdgesAdded ea( 2 * edgesPairs.size(), v );
    // this->addTopologyChange( ea );

    // now warn about the destruction of the old edges
    sofa::helper::vector< unsigned int > indices;
    indices.reserve(2*edgesPairs.size());
    for (unsigned int i=0; i<edgesPairs.size(); ++i)
    {
        indices.push_back( edgesPairs[i][0]  );
        indices.push_back( edgesPairs[i][1] );
    }
    modifier->removeEdgesWarning(indices );

    //            EdgesRemoved er( indices );
    //            this->addTopologyChange( er );

    // propagate the warnings
    topology->propagateTopologicalChanges();

    // now destroy the old edges.
    modifier->removeEdgesProcess( indices );
}

template<class DataTypes>
void EdgeSetTopologyModifier< DataTypes >::fuseEdgesProcess(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPairs,
        const bool removeIsolatedPoints)
{
    EdgeSetTopology< DataTypes > *topology = getEdgeSetTopology();
    EdgeSetTopologyContainer * container = topology->getEdgeSetTopologyContainer();
    EdgeSetTopologyModifier< DataTypes >* modifier = this;

    // first create the edges
    sofa::helper::vector< Edge > v;
    v.reserve(edgesPairs.size());

    sofa::helper::vector< unsigned int > edgeIndexList;
    edgeIndexList.reserve(edgesPairs.size());

    sofa::helper::vector<sofa::helper::vector<unsigned int> > ancestorsArray;
    ancestorsArray.reserve(edgesPairs.size());

    unsigned int nbEdges=container->getNumberOfEdges();

    for (unsigned int i=0; i<edgesPairs.size(); ++i)
    {
        const unsigned int i1 = edgesPairs[i][0];
        const unsigned int i2 = edgesPairs[i][1];

        unsigned int p11 = container->getEdge(i1)[0];
        unsigned int p22 = container->getEdge(i2)[1];

        if(p11 == p22)
        {
            p11 = container->getEdge(i2)[0];
            p22 = container->getEdge(i1)[1];
        }

        const Edge e (p11, p22);
        v.push_back(e);

        edgeIndexList.push_back(nbEdges);
        nbEdges += 1;

        sofa::helper::vector<unsigned int> ancestors(2);
        ancestors[0] = i1;
        ancestors[1] = i2;
        ancestorsArray.push_back(ancestors);
    }

    modifier->addEdgesProcess( v );

    // now warn about the creation
    modifier->addEdgesWarning( v.size(), v, edgeIndexList, ancestorsArray);

    // now warn about the destruction of the old edges
    sofa::helper::vector< unsigned int > indices;
    indices.reserve(2*edgesPairs.size());
    for (unsigned int i=0; i<edgesPairs.size(); ++i)
    {
        indices.push_back( edgesPairs[i][0] );
        indices.push_back( edgesPairs[i][1] );
    }

    modifier->removeEdgesWarning( indices );

    // propagate the warnings
    topology->propagateTopologicalChanges();

    // now destroy the old edges.
    modifier->removeEdgesProcess( indices, removeIsolatedPoints );
}

template<class DataTypes>
void EdgeSetTopologyModifier< DataTypes >::splitEdgesProcess(sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedPoints)
{
    EdgeSetTopology< DataTypes > *topology = getEdgeSetTopology();
    EdgeSetTopologyContainer * container = topology->getEdgeSetTopologyContainer();
    EdgeSetTopologyModifier< DataTypes >* modifier = this;

    sofa::helper::vector< sofa::helper::vector< double > > defaultBaryCoefs(indices.size());

    sofa::helper::vector< sofa::helper::vector< unsigned int > > v(indices.size());

    sofa::helper::vector< Edge >  edges;
    edges.reserve(2*indices.size());

    sofa::helper::vector< unsigned int >  edgesIndex;
    edgesIndex.reserve(2*indices.size());

    unsigned int nbEdges = container->getNumberOfEdges();

    for (unsigned int i=0; i<indices.size(); ++i)
    {
        const unsigned int p1 = container->getEdge( indices[i] )[0];
        const unsigned int p2 = container->getEdge( indices[i] )[1];

        // Adding the new point
        v[i].resize(2);
        v[i][0] = p1;
        v[i][1] = p2;

        // Adding the new Edges
        const Edge e1( p1, topology->getDOFNumber() + i );
        const Edge e2( topology->getDOFNumber() + i, p2 );
        edges.push_back( e1 );
        edges.push_back( e2 );
        edgesIndex.push_back(nbEdges++);
        edgesIndex.push_back(nbEdges++);

        defaultBaryCoefs[i].resize(2, 0.5f);
    }

    modifier->addPointsProcess( indices.size(), v, defaultBaryCoefs);

    modifier->addEdgesProcess( edges );

    // warn about added points and edges
    modifier->addPointsWarning( indices.size(), v, defaultBaryCoefs);

    modifier->addEdgesWarning( edges.size(), edges, edgesIndex);

    // warn about old edges about to be removed
    modifier->removeEdgesWarning( indices );

    topology->propagateTopologicalChanges();

    // Removing the old edges
    modifier->removeEdgesProcess( indices, removeIsolatedPoints );
}

template<class DataTypes>
void EdgeSetTopologyModifier< DataTypes >::splitEdgesProcess(sofa::helper::vector<unsigned int> &indices,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
        const bool removeIsolatedPoints)
{
    EdgeSetTopology< DataTypes > *topology = getEdgeSetTopology();
    EdgeSetTopologyContainer * container = topology->getEdgeSetTopologyContainer();
    EdgeSetTopologyModifier< DataTypes >* modifier = this;

    sofa::helper::vector< sofa::helper::vector< unsigned int > > v(indices.size());

    sofa::helper::vector< Edge >  edges;
    edges.reserve(2*indices.size());

    sofa::helper::vector< unsigned int >  edgesIndex;
    edgesIndex.reserve(2*indices.size());

    unsigned int nbEdges = container->getNumberOfEdges();

    for (unsigned int i=0; i<indices.size(); ++i)
    {
        const unsigned int p1 = container->getEdge( indices[i] )[0];
        const unsigned int p2 = container->getEdge( indices[i] )[1];

        // Adding the new point
        v[i].resize(2);
        v[i][0] = p1;
        v[i][1] = p2;

        // Adding the new Edges
        const Edge e1( p1, topology->getDOFNumber() + i );
        const Edge e2( topology->getDOFNumber() + i, p2 );
        edges.push_back( e1 );
        edges.push_back( e2 );
        edgesIndex.push_back(nbEdges++);
        edgesIndex.push_back(nbEdges++);
    }

    modifier->addPointsProcess( indices.size(), v, baryCoefs);

    modifier->addEdgesProcess( edges );

    // warn about added points and edges
    modifier->addPointsWarning( indices.size(), v, baryCoefs);

    modifier->addEdgesWarning( edges.size(), edges, edgesIndex);

    // warn about old edges about to be removed
    modifier->removeEdgesWarning( indices );

    topology->propagateTopologicalChanges();

    // Removing the old edges
    modifier->removeEdgesProcess( indices, removeIsolatedPoints );
}

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_EDGESETTOPOLOGY_INL
