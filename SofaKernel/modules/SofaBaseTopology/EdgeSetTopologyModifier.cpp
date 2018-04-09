/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaBaseTopology/EdgeSetTopologyModifier.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/TopologyChange.h>
#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <sofa/core/ObjectFactory.h>


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
SOFA_DECL_CLASS(EdgeSetTopologyModifier)
int EdgeSetTopologyModifierClass = core::RegisterObject("Edge set topology modifier")
        .add< EdgeSetTopologyModifier >();

using namespace std;
using namespace sofa::defaulttype;
using namespace sofa::core::topology;

void EdgeSetTopologyModifier::init()
{
    PointSetTopologyModifier::init();
    getContext()->get(m_container);
}

void EdgeSetTopologyModifier::addEdgeProcess(Edge e)
{
#ifndef NDEBUG
    // check if the 2 vertices are different
    if(e[0] == e[1])
    {
        serr << "Error: [EdgeSetTopologyModifier::addEdge] : invalid edge: "
                << e[0] << ", " << e[1] << sendl;

        return;
    }

    // check if there already exists an edge.
    // Important: getEdgeIndex creates the edge vertex shell array
    if(m_container->hasEdgesAroundVertex())
    {
        if(m_container->getEdgeIndex(e[0],e[1]) != -1)
        {
            serr << "Error: [EdgeSetTopologyModifier::addEdgesProcess] : Edge "
                    << e[0] << ", " << e[1] << " already exists." << sendl;
            return;
        }
    }
#endif
    if (m_container->hasEdgesAroundVertex())
    {
        const unsigned int edgeId = m_container->getNumberOfEdges();

        sofa::helper::vector< unsigned int > &shell0 = m_container->getEdgesAroundVertexForModification( e[0] );
        shell0.push_back(edgeId);

        sofa::helper::vector< unsigned int > &shell1 = m_container->getEdgesAroundVertexForModification( e[1] );
        shell1.push_back(edgeId);
    }

    helper::WriteAccessor< Data< sofa::helper::vector<Edge> > > m_edge = m_container->d_edge;
    m_edge.push_back(e);
}


void EdgeSetTopologyModifier::addEdgesProcess(const sofa::helper::vector< Edge > &edges)
{
    for (unsigned int i=0; i<edges.size(); ++i)
    {
        addEdgeProcess(edges[i]);
    }
}


void EdgeSetTopologyModifier::addEdgesWarning(const unsigned int nEdges)
{
    m_container->setEdgeTopologyToDirty();

    // Warning that edges just got created
    EdgesAdded *e = new EdgesAdded(nEdges);
    addTopologyChange(e);
}


void EdgeSetTopologyModifier::addEdgesWarning(const unsigned int nEdges,
        const sofa::helper::vector< Edge >& edgesList,
        const sofa::helper::vector< unsigned int >& edgesIndexList)
{
    m_container->setEdgeTopologyToDirty();

    // Warning that edges just got created
    EdgesAdded *e = new EdgesAdded(nEdges, edgesList, edgesIndexList);
    addTopologyChange(e);
}


void EdgeSetTopologyModifier::addEdgesWarning(const unsigned int nEdges,
        const sofa::helper::vector< Edge >& edgesList,
        const sofa::helper::vector< unsigned int >& edgesIndexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors)
{
    m_container->setEdgeTopologyToDirty();

    // Warning that edges just got created
    EdgesAdded *e = new EdgesAdded(nEdges, edgesList, edgesIndexList, ancestors);
    addTopologyChange(e);
}


void EdgeSetTopologyModifier::addEdgesWarning(const unsigned int nEdges,
        const sofa::helper::vector< Edge >& edgesList,
        const sofa::helper::vector< unsigned int >& edgesIndexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    m_container->setEdgeTopologyToDirty();

    // Warning that edges just got created
    EdgesAdded *e = new EdgesAdded(nEdges, edgesList, edgesIndexList, ancestors, baryCoefs);
    addTopologyChange(e);
}

void EdgeSetTopologyModifier::addEdgesWarning(const unsigned int nEdges,
        const sofa::helper::vector< Edge >& edgesList,
        const sofa::helper::vector< unsigned int >& edgesIndexList,
        const sofa::helper::vector< core::topology::EdgeAncestorElem >& ancestorElems)
{
    m_container->setEdgeTopologyToDirty();

    sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestors;
    sofa::helper::vector< sofa::helper::vector< double > > baryCoefs;
    ancestors.resize(nEdges);
    baryCoefs.resize(nEdges);

    for (unsigned int i=0; i<nEdges; ++i)
    {
        for (unsigned int j=0; j<ancestorElems[i].srcElems.size(); ++j)
        {
            sofa::core::topology::TopologyElemID src = ancestorElems[i].srcElems[j];
            if (src.type == sofa::core::topology::EDGE && src.index != sofa::core::topology::Topology::InvalidID)
                ancestors[i].push_back(src.index);
        }
        if (!ancestors[i].empty())
        {
            baryCoefs[i].insert(baryCoefs[i].begin(), ancestors[i].size(), 1.0/ancestors[i].size());
        }
    }

    // Warning that edges just got created
    EdgesAdded *e = new EdgesAdded(nEdges, edgesList, edgesIndexList, ancestorElems, ancestors, baryCoefs);
    addTopologyChange(e);
}


void EdgeSetTopologyModifier::removeEdgesWarning(sofa::helper::vector<unsigned int> &edges )
{
    m_container->setEdgeTopologyToDirty();

    // sort edges to remove in a descendent order
    std::sort( edges.begin(), edges.end(), std::greater<unsigned int>() );

    // Warning that these edges will be deleted
    EdgesRemoved *e = new EdgesRemoved(edges);
    addTopologyChange(e);
}


void EdgeSetTopologyModifier::removeEdgesProcess(const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedItems)
{
    if(!m_container->hasEdges())	// this method should only be called when edges exist
    {
        sout << "Warning. [EdgeSetTopologyModifier::removeEdgesProcess] edge array is empty." << sendl;
        return;
    }

    if(removeIsolatedItems && !m_container->hasEdgesAroundVertex())
    {
        m_container->createEdgesAroundVertexArray();
    }

    sofa::helper::vector<unsigned int> vertexToBeRemoved;
    helper::WriteAccessor< Data< sofa::helper::vector<Edge> > > m_edge = m_container->d_edge;

    unsigned int lastEdgeIndex = m_container->getNumberOfEdges() - 1;
    for (unsigned int i=0; i<indices.size(); ++i, --lastEdgeIndex)
    {
        // now updates the shell information of the edge formely at the end of the array
        if(m_container->hasEdgesAroundVertex())
        {


            const Edge &e = m_edge[ indices[i] ];
            const Edge &q = m_edge[ lastEdgeIndex ];
            const unsigned int point0 = e[0], point1 = e[1];
            const unsigned int point2 = q[0], point3 = q[1];

            sofa::helper::vector< unsigned int > &shell0 = m_container->m_edgesAroundVertex[ point0 ];
            shell0.erase( std::remove( shell0.begin(), shell0.end(), indices[i] ), shell0.end() );
            if(removeIsolatedItems && shell0.empty())
            {
                vertexToBeRemoved.push_back(point0);
            }

            sofa::helper::vector< unsigned int > &shell1 = m_container->m_edgesAroundVertex[ point1 ];
            shell1.erase( std::remove( shell1.begin(), shell1.end(), indices[i] ), shell1.end() );
            if(removeIsolatedItems && shell1.empty())
            {
                vertexToBeRemoved.push_back(point1);
            }

            if(indices[i] < lastEdgeIndex)
            {
                //replaces the edge index oldEdgeIndex with indices[i] for the first vertex
                sofa::helper::vector< unsigned int > &shell2 = m_container->m_edgesAroundVertex[ point2 ];
                replace(shell2.begin(), shell2.end(), lastEdgeIndex, indices[i]);

                //replaces the edge index oldEdgeIndex with indices[i] for the second vertex
                sofa::helper::vector< unsigned int > &shell3 = m_container->m_edgesAroundVertex[ point3 ];
                replace(shell3.begin(), shell3.end(), lastEdgeIndex, indices[i]);
            }
        }

        // removes the edge from the edgelist
        m_edge[ indices[i] ] = m_edge[ lastEdgeIndex ]; // overwriting with last valid value.
        m_edge.resize( lastEdgeIndex ); // resizing to erase multiple occurence of the edge.
    }

    if (! vertexToBeRemoved.empty())
    {
        removePointsWarning(vertexToBeRemoved);
        // inform other objects that the points are going to be removed
        propagateTopologicalChanges();
        removePointsProcess(vertexToBeRemoved);
    }
}

void EdgeSetTopologyModifier::addPointsProcess(const unsigned int nPoints)
{
    // start by calling the parent's method.
    PointSetTopologyModifier::addPointsProcess( nPoints );

    if(m_container->hasEdgesAroundVertex())
        m_container->m_edgesAroundVertex.resize( m_container->getNbPoints() );
}

void EdgeSetTopologyModifier::removePointsProcess(const sofa::helper::vector<unsigned int> &indices,
        const bool removeDOF)
{
    // Note: edges connected to the points being removed are not removed here (this situation should not occur)

    if(m_container->hasEdges())
    {
        helper::WriteAccessor< Data< sofa::helper::vector<Edge> > > m_edge = m_container->d_edge;
        // forces the construction of the edge shell array if it does not exists
        if(!m_container->hasEdgesAroundVertex())
            m_container->createEdgesAroundVertexArray();

        unsigned int lastPoint = m_container->getNbPoints() - 1;
        for (unsigned int i=0; i<indices.size(); ++i, --lastPoint)
        {
            // updating the edges connected to the point replacing the removed one:
            // for all edges connected to the last point
            for (unsigned int j=0; j<m_container->m_edgesAroundVertex[lastPoint].size(); ++j)
            {
                const int edgeId = m_container->m_edgesAroundVertex[lastPoint][j];
                // change the old index for the new one
                if ( m_edge[ edgeId ][0] == lastPoint )
                    m_edge[ edgeId ][0] = indices[i];
                else
                    m_edge[ edgeId ][1] = indices[i];
            }

            // updating the edge shell itself (change the old index for the new one)
            m_container->m_edgesAroundVertex[ indices[i] ] = m_container->m_edgesAroundVertex[ lastPoint ];
        }

        m_container->m_edgesAroundVertex.resize( m_container->m_edgesAroundVertex.size() - indices.size() );
    }

    // Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    // call the parent method.
    PointSetTopologyModifier::removePointsProcess( indices, removeDOF );
}

void EdgeSetTopologyModifier::renumberPointsProcess( const sofa::helper::vector<unsigned int> &index,
        const sofa::helper::vector<unsigned int> &inv_index,
        const bool renumberDOF)
{
    if(m_container->hasEdges())
    {
        helper::WriteAccessor< Data< sofa::helper::vector<Edge> > > m_edge = m_container->d_edge;
        if(m_container->hasEdgesAroundVertex())
        {
            // copy of the the edge vertex shell array
            sofa::helper::vector< sofa::helper::vector< unsigned int > > edgesAroundVertex_cp = m_container->getEdgesAroundVertexArray();

            for (unsigned int i=0; i<index.size(); ++i)
            {
                m_container->m_edgesAroundVertex[i] = edgesAroundVertex_cp[ index[i] ];
            }
        }

        for (unsigned int i=0; i<m_edge.size(); ++i)
        {
            const unsigned int p0 = inv_index[ m_edge[i][0]  ];
            const unsigned int p1 = inv_index[ m_edge[i][1]  ];

            // FIXME : Edges should not be flipped during simulations as it will break code such as FEM storing a rest shape.
            // Commented by pierre-jean.bensoussan@digital-trainers.com
            /*
            if(p0 < p1)
            {
                m_container->m_edge[i][0] = p0;
                m_container->m_edge[i][1] = p1;
            }
            else
            {
                m_container->m_edge[i][0] = p1;
                m_container->m_edge[i][1] = p0;
            }
            */

            m_edge[i][0] = p0;
            m_edge[i][1] = p1;
        }
    }

    // call the parent method
    PointSetTopologyModifier::renumberPointsProcess( index, inv_index, renumberDOF );
}


void EdgeSetTopologyModifier::swapEdgesProcess(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPairs)
{
    if(!m_container->hasEdges())
        return;

    // first create the edges
    sofa::helper::vector< Edge > v;
    v.reserve(2*edgesPairs.size());

    sofa::helper::vector< unsigned int > edgeIndexList;
    edgeIndexList.reserve(2*edgesPairs.size());

    sofa::helper::vector<sofa::helper::vector<unsigned int> > ancestorsArray;
    ancestorsArray.reserve(edgesPairs.size());

    unsigned int nbEdges = m_container->getNumberOfEdges();

    for (unsigned int i=0; i<edgesPairs.size(); ++i)
    {
        const unsigned int i1 = edgesPairs[i][0];
        const unsigned int i2 = edgesPairs[i][1];

        const unsigned int p11 = m_container->getEdge(i1)[0];
        const unsigned int p12 = m_container->getEdge(i1)[1];
        const unsigned int p21 = m_container->getEdge(i2)[0];
        const unsigned int p22 = m_container->getEdge(i2)[1];

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

    addEdgesProcess( v );

    // now warn about the creation
    addEdgesWarning((unsigned int)v.size(), v, edgeIndexList, ancestorsArray);

    // now warn about the destruction of the old edges
    sofa::helper::vector< unsigned int > indices;
    indices.reserve(2*edgesPairs.size());
    for (unsigned int i=0; i<edgesPairs.size(); ++i)
    {
        indices.push_back( edgesPairs[i][0]  );
        indices.push_back( edgesPairs[i][1] );
    }
    removeEdgesWarning(indices );

    // propagate the warnings
    propagateTopologicalChanges();

    // now destroy the old edges.
    removeEdgesProcess( indices );
}


void EdgeSetTopologyModifier::fuseEdgesProcess(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPairs,
        const bool removeIsolatedPoints)
{
    if(!m_container->hasEdges())
        return;

    // first create the edges
    sofa::helper::vector< Edge > v;
    v.reserve(edgesPairs.size());

    sofa::helper::vector< unsigned int > edgeIndexList;
    edgeIndexList.reserve(edgesPairs.size());

    sofa::helper::vector<sofa::helper::vector<unsigned int> > ancestorsArray;
    ancestorsArray.reserve(edgesPairs.size());

    unsigned int nbEdges=m_container->getNumberOfEdges();

    for (unsigned int i=0; i<edgesPairs.size(); ++i)
    {
        const unsigned int i1 = edgesPairs[i][0];
        const unsigned int i2 = edgesPairs[i][1];

        unsigned int p11 = m_container->getEdge(i1)[0];
        unsigned int p22 = m_container->getEdge(i2)[1];

        if(p11 == p22)
        {
            p11 = m_container->getEdge(i2)[0];
            p22 = m_container->getEdge(i1)[1];
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

    addEdgesProcess( v );

    // now warn about the creation
    addEdgesWarning((unsigned int)v.size(), v, edgeIndexList, ancestorsArray);

    // now warn about the destruction of the old edges
    sofa::helper::vector< unsigned int > indices;
    indices.reserve(2*edgesPairs.size());
    for (unsigned int i=0; i<edgesPairs.size(); ++i)
    {
        indices.push_back( edgesPairs[i][0] );
        indices.push_back( edgesPairs[i][1] );
    }

    removeEdgesWarning( indices );

    // propagate the warnings
    propagateTopologicalChanges();

    // now destroy the old edges.
    removeEdgesProcess( indices, removeIsolatedPoints );
}


void EdgeSetTopologyModifier::splitEdgesProcess(sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedPoints)
{
    if(!m_container->hasEdges())
        return;

    sofa::helper::vector< sofa::helper::vector< double > > defaultBaryCoefs(indices.size());

    sofa::helper::vector< sofa::helper::vector< unsigned int > > v(indices.size());

    sofa::helper::vector< Edge >  edges;
    edges.reserve(2*indices.size());

    sofa::helper::vector< unsigned int >  edgesIndex;
    edgesIndex.reserve(2*indices.size());

    unsigned int nbEdges = m_container->getNumberOfEdges();

    for (unsigned int i=0; i<indices.size(); ++i)
    {
        const unsigned int p1 = m_container->getEdge( indices[i] )[0];
        const unsigned int p2 = m_container->getEdge( indices[i] )[1];

        // Adding the new point
        v[i].resize(2);
        v[i][0] = p1;
        v[i][1] = p2;

        // Adding the new Edges
        const Edge e1( p1, m_container->getNbPoints() + i );
        const Edge e2( m_container->getNbPoints() + i, p2 );
        edges.push_back( e1 );
        edges.push_back( e2 );
        edgesIndex.push_back(nbEdges++);
        edgesIndex.push_back(nbEdges++);

        defaultBaryCoefs[i].resize(2, 0.5f);
    }

    addPointsProcess((unsigned int)indices.size());

    addEdgesProcess( edges );

    // warn about added points and edges
    addPointsWarning((unsigned int)indices.size(), v, defaultBaryCoefs);

    addEdgesWarning((unsigned int)edges.size(), edges, edgesIndex);

    // warn about old edges about to be removed
    removeEdgesWarning( indices );

    propagateTopologicalChanges();

    // Removing the old edges
    removeEdgesProcess( indices, removeIsolatedPoints );
}


void EdgeSetTopologyModifier::splitEdgesProcess(sofa::helper::vector<unsigned int> &indices,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
        const bool removeIsolatedPoints)
{
    if(!m_container->hasEdges())
        return;

    sofa::helper::vector< sofa::helper::vector< unsigned int > > v(indices.size());

    sofa::helper::vector< Edge >  edges;
    edges.reserve(2*indices.size());

    sofa::helper::vector< unsigned int >  edgesIndex;
    edgesIndex.reserve(2*indices.size());

    unsigned int nbEdges = m_container->getNumberOfEdges();

    for (unsigned int i=0; i<indices.size(); ++i)
    {
        const unsigned int p1 = m_container->getEdge( indices[i] )[0];
        const unsigned int p2 = m_container->getEdge( indices[i] )[1];

        // Adding the new point
        v[i].resize(2);
        v[i][0] = p1;
        v[i][1] = p2;

        // Adding the new Edges
        const Edge e1( p1, m_container->getNbPoints() + i );
        const Edge e2( m_container->getNbPoints() + i, p2 );
        edges.push_back( e1 );
        edges.push_back( e2 );
        edgesIndex.push_back(nbEdges++);
        edgesIndex.push_back(nbEdges++);
    }

    addPointsProcess((unsigned int)indices.size());

    addEdgesProcess( edges );

    // warn about added points and edges
    addPointsWarning((unsigned int)indices.size(), v, baryCoefs);

    addEdgesWarning((unsigned int)edges.size(), edges, edgesIndex);

    // warn about old edges about to be removed
    removeEdgesWarning( indices );

    propagateTopologicalChanges();

    // Removing the old edges
    removeEdgesProcess( indices, removeIsolatedPoints );
}

void EdgeSetTopologyModifier::removeEdges(const sofa::helper::vector< unsigned int >& edgeIds,
        const bool removeIsolatedPoints, const bool resetTopoChange)
{
    sofa::helper::vector<unsigned int> edgeIds_filtered;
    for (unsigned int i = 0; i < edgeIds.size(); i++)
    {
        if( edgeIds[i] >= m_container->getNumberOfEdges())
            dmsg_error() << "Unable to remoev and edges: "<< edgeIds[i] <<" is out of bound and won't be removed." ;
        else
            edgeIds_filtered.push_back(edgeIds[i]);
    }

    /// add the topological changes in the queue
    removeEdgesWarning(edgeIds_filtered);
    // inform other objects that the edges are going to be removed
    if (resetTopoChange)
        propagateTopologicalChanges();
    else
        propagateTopologicalChangesWithoutReset();
    // now destroy the old edges.
    removeEdgesProcess( edgeIds_filtered, removeIsolatedPoints );

    m_container->checkTopology();
}

void EdgeSetTopologyModifier::removeItems(const sofa::helper::vector< unsigned int >& items)
{
    removeEdges(items);
}

void EdgeSetTopologyModifier::renumberPoints( const sofa::helper::vector<unsigned int> &index,
        const sofa::helper::vector<unsigned int> &inv_index)
{
    /// add the topological changes in the queue
    renumberPointsWarning(index, inv_index);
    // inform other objects that the triangles are going to be removed
    propagateTopologicalChanges();
    // now renumber the points
    renumberPointsProcess(index, inv_index);

    m_container->checkTopology();
}

void EdgeSetTopologyModifier::addEdges(const sofa::helper::vector< Edge >& edges)
{
    unsigned int nEdges = m_container->getNumberOfEdges();

    /// actually add edges in the topology container
    addEdgesProcess(edges);

    sofa::helper::vector<unsigned int> edgesIndex;
    edgesIndex.reserve(edges.size());

    for (unsigned int i=0; i<edges.size(); ++i)
    {
        edgesIndex.push_back(nEdges+i);
    }

    // add topology event in the stack of topological events
    addEdgesWarning((unsigned int)edges.size(), edges, edgesIndex);

    // inform other objects that the edges are already added
    propagateTopologicalChanges();
}

void EdgeSetTopologyModifier::addEdges(const sofa::helper::vector< Edge >& edges,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    unsigned int nEdges = m_container->getNumberOfEdges();

    /// actually add edges in the topology container
    addEdgesProcess(edges);

    sofa::helper::vector<unsigned int> edgesIndex;
    edgesIndex.reserve(edges.size());

    for (unsigned int i=0; i<edges.size(); ++i)
    {
        edgesIndex.push_back(nEdges+i);
    }

    // add topology event in the stack of topological events
    addEdgesWarning((unsigned int)edges.size(), edges, edgesIndex, ancestors, baryCoefs);

    // inform other objects that the edges are already added
    propagateTopologicalChanges();
}

void EdgeSetTopologyModifier::addEdges(const sofa::helper::vector< Edge >& edges,
        const sofa::helper::vector< core::topology::EdgeAncestorElem >& ancestorElems)
{
    unsigned int nEdges = m_container->getNumberOfEdges();

    assert(ancestorElems.size() == edges.size());

    /// actually add edges in the topology container
    addEdgesProcess(edges);

    sofa::helper::vector<unsigned int> edgesIndex;
    edgesIndex.resize(edges.size());
    for (unsigned int i=0; i<edges.size(); ++i)
    {
        edgesIndex[i] = nEdges+i;
    }

    // add topology event in the stack of topological events
    addEdgesWarning((unsigned int)edges.size(), edges, edgesIndex, ancestorElems);

    // inform other objects that the edges are already added
    propagateTopologicalChanges();
}

void EdgeSetTopologyModifier::swapEdges(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPairs)
{
    swapEdgesProcess(edgesPairs);
    m_container->checkTopology();
}

void EdgeSetTopologyModifier::fuseEdges(const sofa::helper::vector< sofa::helper::vector< unsigned int > >& edgesPairs, const bool removeIsolatedPoints)
{
    fuseEdgesProcess(edgesPairs, removeIsolatedPoints);
    m_container->checkTopology();
}

void EdgeSetTopologyModifier::splitEdges( sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedPoints)
{
    splitEdgesProcess(indices, removeIsolatedPoints);
    m_container->checkTopology();
}

void EdgeSetTopologyModifier::splitEdges( sofa::helper::vector<unsigned int> &indices,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
        const bool removeIsolatedPoints)
{
    splitEdgesProcess(indices, baryCoefs, removeIsolatedPoints);
    m_container->checkTopology();
}

// Give the optimal vertex permutation according to the Reverse CuthillMckee algorithm (use BOOST GRAPH LIBRAIRY)
void EdgeSetTopologyModifier::resortCuthillMckee(sofa::helper::vector<int>& inverse_permutation)
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

    sout << "original bandwidth: " << bandwidth(G) << sendl;

    std::vector<Vertex> inv_perm(num_vertices(G));
    std::vector<size_type> perm(num_vertices(G));

    //reverse cuthill_mckee_ordering
    cuthill_mckee_ordering(G, inv_perm.rbegin());

    //sout << "Reverse Cuthill-McKee ordering:" << sendl;
    //sout << "  ";
    unsigned int ind_i = 0;
    for (std::vector<Vertex>::const_iterator it = inv_perm.begin();
            it != inv_perm.end(); ++it)
    {
        //sout << index_map[*it] << " ";
        inverse_permutation[ind_i++] = (int)index_map[*it];
    }
    //sout << sendl;

    for (size_type c=0; c!=inv_perm.size(); ++c)
        perm[index_map[inv_perm[c]]] = c;

    sout << "  bandwidth: "
            << bandwidth(G, make_iterator_property_map(&perm[0], index_map, perm[0]))
            << sendl;
}




void EdgeSetTopologyModifier::movePointsProcess (const sofa::helper::vector <unsigned int>& id,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& coefs,
        const bool moveDOF)
{
    SOFA_UNUSED(moveDOF);
    unsigned int nbrVertex = (unsigned int)id.size();
    bool doublet;
    sofa::helper::vector<unsigned int> edgesAroundVertex2Move;
    sofa::helper::vector< Edge > edgeArray;

    // Step 1/4 - Creating trianglesAroundVertex to moved due to moved points:
    for (unsigned int i = 0; i<nbrVertex; ++i)
    {
        const sofa::helper::vector <unsigned int>& edgesAroundVertex = m_container->getEdgesAroundVertexArray()[ id[i] ];

        for (unsigned int j = 0; j<edgesAroundVertex.size(); ++j)
        {
            doublet = false;

            for (unsigned int k =0; k<edgesAroundVertex2Move.size(); ++k) //Avoid double
            {
                if (edgesAroundVertex2Move[k] == edgesAroundVertex[j])
                {
                    doublet = true;
                    break;
                }
            }

            if(!doublet)
                edgesAroundVertex2Move.push_back (edgesAroundVertex[j]);

        }
    }

    std::sort( edgesAroundVertex2Move.begin(), edgesAroundVertex2Move.end(), std::greater<unsigned int>() );

    // Step 2/4 - Create event to delete all elements before moving and propagate it:
    EdgesMoved_Removing *ev1 = new EdgesMoved_Removing (edgesAroundVertex2Move);
    this->addTopologyChange(ev1);
    propagateTopologicalChanges();


    // Step 3/4 - Physically move all dof:
    PointSetTopologyModifier::movePointsProcess (id, ancestors, coefs);


    // Step 4/4 - Create event to recompute all elements concerned by moving and propagate it:

    // Creating the corresponding array of Triangles for ancestors
    for (unsigned int i = 0; i<edgesAroundVertex2Move.size(); i++)
        edgeArray.push_back (m_container->getEdgeArray()[ edgesAroundVertex2Move[i] ]);

    EdgesMoved_Adding *ev2 = new EdgesMoved_Adding (edgesAroundVertex2Move, edgeArray);
    this->addTopologyChange(ev2); // This event should be propagated with global workflow
}



bool EdgeSetTopologyModifier::removeConnectedComponents(unsigned int elemID)
{
    if(!m_container)
    {
        serr << "TopologyContainer pointer is empty." << sendl;
        return false;
    }

    sofa::helper::vector <unsigned int> elems = m_container->getConnectedElement(elemID);
    this->removeItems(elems);
    return true;
}


bool EdgeSetTopologyModifier::removeConnectedElements(unsigned int elemID)
{
    if(!m_container)
    {
        serr << "TopologyContainer pointer is empty." << sendl;
        return false;
    }

    sofa::helper::vector <unsigned int> elems = m_container->getElementAroundElement(elemID);
    this->removeItems(elems);
    return true;
}


bool EdgeSetTopologyModifier::removeIsolatedElements()
{
    return this->removeIsolatedElements(0);
}


bool EdgeSetTopologyModifier::removeIsolatedElements(unsigned int scaleElem)
{
    if(!m_container)
    {
        serr << "TopologyContainer pointer is empty." << sendl;
        return false;
    }

    unsigned int nbr = this->m_container->getNumberOfElements();
    sofa::helper::vector <unsigned int> elemAll = m_container->getConnectedElement(0);
    sofa::helper::vector <unsigned int> elem, elemMax, elemToRemove;

    if (nbr == elemAll.size()) // nothing to do
        return true;

    elemMax = elemAll;

    if (scaleElem == 0) //remove all isolated elements
        scaleElem = nbr;

    while (elemAll.size() < nbr)
    {
        std::sort(elemAll.begin(), elemAll.end());
        EdgeID other_edgeID = (EdgeID)elemAll.size();

        for (EdgeID i = 0; i<elemAll.size(); ++i)
            if (elemAll[i] != i)
            {
                other_edgeID = i;
                break;
            }

        elem = this->m_container->getConnectedElement(other_edgeID);
        elemAll.insert(elemAll.begin(), elem.begin(), elem.end());

        if (elemMax.size() < elem.size())
        {
            if (elemMax.size() <= scaleElem)
                elemToRemove.insert(elemToRemove.begin(), elemMax.begin(), elemMax.end());

            elemMax = elem;
        }
        else
        {
            if (elem.size() <= scaleElem)
                elemToRemove.insert(elemToRemove.begin(), elem.begin(), elem.end());
        }
    }

    this->removeItems(elemToRemove);

    return true;
}



void EdgeSetTopologyModifier::propagateTopologicalEngineChanges()
{
    if (m_container->beginChange() == m_container->endChange()) // nothing to do if no event is stored
        return;

    if (!m_container->isEdgeTopologyDirty()) // edge Data has not been touched
        return PointSetTopologyModifier::propagateTopologicalEngineChanges();

    std::list<sofa::core::topology::TopologyEngine *>::iterator it;
    for ( it = m_container->m_enginesList.begin(); it!=m_container->m_enginesList.end(); ++it)
    {
        // no need to dynamic cast this time? TO BE CHECKED!
        sofa::core::topology::TopologyEngine* topoEngine = (*it);
        if (topoEngine->isDirty())
        {
#ifndef NDEBUG
            msg_info() << "EdgeSetTopologyModifier::performing: " << topoEngine->getName() ;
#endif
            topoEngine->update();
        }
    }

    m_container->cleanEdgeTopologyFromDirty();
    PointSetTopologyModifier::propagateTopologicalEngineChanges();
}


} // namespace topology

} // namespace component

} // namespace sofa

