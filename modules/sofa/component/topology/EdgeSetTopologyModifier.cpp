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
#include <sofa/component/topology/EdgeSetTopologyModifier.h>
#include <sofa/component/topology/EdgeSetTopologyChange.h>
#include <sofa/component/topology/EdgeSetTopologyContainer.h>
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
using namespace sofa::defaulttype;
SOFA_DECL_CLASS(EdgeSetTopologyModifier)
int EdgeSetTopologyModifierClass = core::RegisterObject("Edge set topology modifier")
        .add< EdgeSetTopologyModifier >();

using namespace std;
using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;

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
        cout << "Error: [EdgeSetTopologyModifier::addEdge] : invalid edge: "
                << e[0] << ", " << e[1] << endl;

        return;
    }

    // check if there already exists an edge.
    // Important: getEdgeIndex creates the edge vertex shell array
    if(m_container->hasEdgeVertexShell())
    {
        if(m_container->getEdgeIndex(e[0],e[1]) != -1)
        {
            cout << "Error: [EdgeSetTopologyModifier::addEdgesProcess] : Edge "
                    << e[0] << ", " << e[1] << " already exists." << endl;
            return;
        }
    }
#endif
    if (m_container->hasEdgeVertexShell())
    {
        const unsigned int edgeId = m_container->getNumberOfEdges();

        sofa::helper::vector< unsigned int > &shell0 = m_container->getEdgeVertexShellForModification( e[0] );
        shell0.push_back(edgeId);

        sofa::helper::vector< unsigned int > &shell1 = m_container->getEdgeVertexShellForModification( e[1] );
        shell1.push_back(edgeId);
    }

    m_container->m_edge.push_back(e);
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
    // Warning that edges just got created
    EdgesAdded *e = new EdgesAdded(nEdges);
    addTopologyChange(e);
}


void EdgeSetTopologyModifier::addEdgesWarning(const unsigned int nEdges,
        const sofa::helper::vector< Edge >& edgesList,
        const sofa::helper::vector< unsigned int >& edgesIndexList)
{
    // Warning that edges just got created
    EdgesAdded *e = new EdgesAdded(nEdges, edgesList, edgesIndexList);
    addTopologyChange(e);
}


void EdgeSetTopologyModifier::addEdgesWarning(const unsigned int nEdges,
        const sofa::helper::vector< Edge >& edgesList,
        const sofa::helper::vector< unsigned int >& edgesIndexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors)
{
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
    // Warning that edges just got created
    EdgesAdded *e = new EdgesAdded(nEdges, edgesList, edgesIndexList, ancestors, baryCoefs);
    addTopologyChange(e);
}


void EdgeSetTopologyModifier::removeEdgesWarning(sofa::helper::vector<unsigned int> &edges )
{
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
        cout << "Warning. [EdgeSetTopologyModifier::removeEdgesProcess] edge array is empty." << endl;
        return;
    }

    if(removeIsolatedItems && !m_container->hasEdgeVertexShell())
    {
        m_container->createEdgeVertexShellArray();
    }

    sofa::helper::vector<unsigned int> vertexToBeRemoved;

    unsigned int lastEdgeIndex = m_container->getNumberOfEdges() - 1;
    for (unsigned int i=0; i<indices.size(); ++i, --lastEdgeIndex)
    {
        // now updates the shell information of the edge formely at the end of the array
        if(m_container->hasEdgeVertexShell())
        {
            const Edge &e = m_container->m_edge[ indices[i] ];
            const Edge &q = m_container->m_edge[ lastEdgeIndex ];
            const unsigned int point0 = e[0], point1 = e[1];
            const unsigned int point2 = q[0], point3 = q[1];

            sofa::helper::vector< unsigned int > &shell0 = m_container->m_edgeVertexShell[ point0 ];
            shell0.erase( std::remove( shell0.begin(), shell0.end(), indices[i] ), shell0.end() );
            if(removeIsolatedItems && shell0.empty())
            {
                vertexToBeRemoved.push_back(point0);
            }

            sofa::helper::vector< unsigned int > &shell1 = m_container->m_edgeVertexShell[ point1 ];
            shell1.erase( std::remove( shell1.begin(), shell1.end(), indices[i] ), shell1.end() );
            if(removeIsolatedItems && shell1.empty())
            {
                vertexToBeRemoved.push_back(point1);
            }

            if(indices[i] < lastEdgeIndex)
            {
                //replaces the edge index oldEdgeIndex with indices[i] for the first vertex
                sofa::helper::vector< unsigned int > &shell2 = m_container->m_edgeVertexShell[ point2 ];
                replace(shell2.begin(), shell2.end(), lastEdgeIndex, indices[i]);

                //replaces the edge index oldEdgeIndex with indices[i] for the second vertex
                sofa::helper::vector< unsigned int > &shell3 = m_container->m_edgeVertexShell[ point3 ];
                replace(shell3.begin(), shell3.end(), lastEdgeIndex, indices[i]);
            }
        }

        // removes the edge from the edgelist
        m_container->m_edge[ indices[i] ] = m_container->m_edge[ lastEdgeIndex ]; // overwriting with last valid value.
        m_container->m_edge.resize( lastEdgeIndex ); // resizing to erase multiple occurence of the edge.
    }

    if (! vertexToBeRemoved.empty())
    {
        removePointsWarning(vertexToBeRemoved);
        // inform other objects that the points are going to be removed
        m_container->propagateTopologicalChanges();
        removePointsProcess(vertexToBeRemoved);
    }
}

void EdgeSetTopologyModifier::addPointsProcess(const unsigned int nPoints)
{
    // start by calling the parent's method.
    PointSetTopologyModifier::addPointsProcess( nPoints );

    if(m_container->hasEdgeVertexShell())
        m_container->m_edgeVertexShell.resize( m_container->getNbPoints() );
}

void EdgeSetTopologyModifier::removePointsProcess(sofa::helper::vector<unsigned int> &indices,
        const bool removeDOF)
{
    // Note: edges connected to the points being removed are not removed here (this situation should not occur)

    if(m_container->hasEdges())
    {
        // forces the construction of the edge shell array if it does not exists
        if(!m_container->hasEdgeVertexShell())
            m_container->createEdgeVertexShellArray();

        unsigned int lastPoint = m_container->getNbPoints() - 1;
        for (unsigned int i=0; i<indices.size(); ++i, --lastPoint)
        {
            // updating the edges connected to the point replacing the removed one:
            // for all edges connected to the last point
            for (unsigned int j=0; j<m_container->m_edgeVertexShell[lastPoint].size(); ++j)
            {
                const int edgeId = m_container->m_edgeVertexShell[lastPoint][j];
                // change the old index for the new one
                if ( m_container->m_edge[ edgeId ][0] == lastPoint )
                    m_container->m_edge[ edgeId ][0] = indices[i];
                else
                    m_container->m_edge[ edgeId ][1] = indices[i];
            }

            // updating the edge shell itself (change the old index for the new one)
            m_container->m_edgeVertexShell[ indices[i] ] = m_container->m_edgeVertexShell[ lastPoint ];
        }

        m_container->m_edgeVertexShell.resize( m_container->m_edgeVertexShell.size() - indices.size() );
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
        if(m_container->hasEdgeVertexShell())
        {
            // copy of the the edge vertex shell array
            sofa::helper::vector< sofa::helper::vector< unsigned int > > edgeVertexShell_cp = m_container->getEdgeVertexShellArray();

            for (unsigned int i=0; i<index.size(); ++i)
            {
                m_container->m_edgeVertexShell[i] = edgeVertexShell_cp[ index[i] ];
            }
        }

        for (unsigned int i=0; i<m_container->m_edge.size(); ++i)
        {
            const unsigned int p0 = inv_index[ m_container->m_edge[i][0]  ];
            const unsigned int p1 = inv_index[ m_container->m_edge[i][1]  ];

            if(p0<p1)
            {
                m_container->m_edge[i][0] = p0;
                m_container->m_edge[i][1] = p1;
            }
            else
            {
                m_container->m_edge[i][0] = p1;
                m_container->m_edge[i][1] = p0;
            }
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
    addEdgesWarning( v.size(), v, edgeIndexList, ancestorsArray);

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
    m_container->propagateTopologicalChanges();

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
    addEdgesWarning( v.size(), v, edgeIndexList, ancestorsArray);

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
    m_container->propagateTopologicalChanges();

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

    addPointsProcess( indices.size());

    addEdgesProcess( edges );

    // warn about added points and edges
    addPointsWarning( indices.size(), v, defaultBaryCoefs);

    addEdgesWarning( edges.size(), edges, edgesIndex);

    // warn about old edges about to be removed
    removeEdgesWarning( indices );

    m_container->propagateTopologicalChanges();

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

    addPointsProcess( indices.size());

    addEdgesProcess( edges );

    // warn about added points and edges
    addPointsWarning( indices.size(), v, baryCoefs);

    addEdgesWarning( edges.size(), edges, edgesIndex);

    // warn about old edges about to be removed
    removeEdgesWarning( indices );

    m_container->propagateTopologicalChanges();

    // Removing the old edges
    removeEdgesProcess( indices, removeIsolatedPoints );
}

} // namespace topology

} // namespace component

} // namespace sofa

