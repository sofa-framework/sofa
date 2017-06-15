/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaBaseTopology/TetrahedronSetTopologyModifier.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/TopologyChange.h>
#include <SofaBaseTopology/TetrahedronSetTopologyContainer.h>
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
SOFA_DECL_CLASS(TetrahedronSetTopologyModifier)
int TetrahedronSetTopologyModifierClass = core::RegisterObject("Tetrahedron set topology modifier")
        .add< TetrahedronSetTopologyModifier >();

using namespace std;
using namespace sofa::defaulttype;
using namespace sofa::core::topology;

//const unsigned int edgesInTetrahedronArray[6][2] = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};


void TetrahedronSetTopologyModifier::init()
{
    TriangleSetTopologyModifier::init();

    this->getContext()->get(m_container);
}

void TetrahedronSetTopologyModifier::reinit()
{
    TriangleSetTopologyModifier::reinit();
}


void TetrahedronSetTopologyModifier::addTetrahedra(const sofa::helper::vector<Tetrahedron> &tetrahedra)
{
    unsigned int ntetra = m_container->getNbTetrahedra();

    /// effectively add triangles in the topology container
    addTetrahedraProcess(tetrahedra);

    sofa::helper::vector<unsigned int> tetrahedraIndex;
    tetrahedraIndex.reserve(tetrahedra.size());

    for (unsigned int i=0; i<tetrahedra.size(); ++i)
        tetrahedraIndex.push_back(ntetra+i);

    // add topology event in the stack of topological events
    addTetrahedraWarning((unsigned int)tetrahedra.size(), tetrahedra, tetrahedraIndex);

    // inform other objects that the edges are already added
    propagateTopologicalChanges();
}


void TetrahedronSetTopologyModifier::addTetrahedra(const sofa::helper::vector<Tetrahedron> &tetrahedra,
        const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors,
        const sofa::helper::vector<sofa::helper::vector<double> > &baryCoefs)
{
    unsigned int ntetra = m_container->getNbTetrahedra();

    /// effectively add triangles in the topology container
    addTetrahedraProcess(tetrahedra);

    sofa::helper::vector<unsigned int> tetrahedraIndex;
    tetrahedraIndex.reserve(tetrahedra.size());

    for (unsigned int i=0; i<tetrahedra.size(); ++i)
        tetrahedraIndex.push_back(ntetra+i);

    // add topology event in the stack of topological events
    addTetrahedraWarning((unsigned int)tetrahedra.size(), tetrahedra, tetrahedraIndex, ancestors, baryCoefs);

    // inform other objects that the edges are already added
    propagateTopologicalChanges();
}

void TetrahedronSetTopologyModifier::addTetrahedronProcess(Tetrahedron t)
{
#ifndef NDEBUG
    // check if the 3 vertices are different
    assert(t[0]!=t[1]);
    assert(t[0]!=t[2]);
    assert(t[0]!=t[3]);
    assert(t[1]!=t[2]);
    assert(t[1]!=t[3]);
    assert(t[2]!=t[3]);

    // check if there already exists a tetrahedron with the same indices
    // assert(m_container->getTetrahedronIndex(t[0], t[1], t[2], t[3])== -1);
#endif
    helper::WriteAccessor< Data< sofa::helper::vector<Tetrahedron> > > m_tetrahedron = m_container->d_tetrahedron;
    const unsigned int tetrahedronIndex = (unsigned int)m_tetrahedron.size();

    if (m_container->hasTrianglesInTetrahedron())
    {
        for (unsigned int j=0; j<4; ++j)
        {
            int triangleIndex = m_container->getTriangleIndex(t[(j+1)%4], t[(j+2)%4], t[(j+3)%4]);

            if(triangleIndex == -1)
            {
                // first create the traingle
                sofa::helper::vector< Triangle > v;
                Triangle e1 (t[(j+1)%4],t[(j+2)%4],t[(j+3)%4]);
                v.push_back(e1);

                addTrianglesProcess((const sofa::helper::vector< Triangle > &) v);

                triangleIndex = m_container->getTriangleIndex(t[(j+1)%4], t[(j+2)%4], t[(j+3)%4]);

                sofa::helper::vector< unsigned int > triangleIndexList;
                triangleIndexList.push_back(triangleIndex);
                addTrianglesWarning((unsigned int)v.size(), v, triangleIndexList);
            }

            //m_container->m_trianglesInTetrahedron.resize(triangleIndex+1);
            m_container->m_trianglesInTetrahedron.resize(tetrahedronIndex+1);
            m_container->m_trianglesInTetrahedron[tetrahedronIndex][j]= triangleIndex;
        }
    }

    if (m_container->hasEdgesInTetrahedron())
    {
        for (unsigned int j=0; j<6; ++j)
        {
            int p0=-1,p1=-1;

            // compute the index of edges in tetra
            if (j<3)
            {
                p0=0; p1=j+1;
            }
            else if (j<5)
            {
                p0=1; p1=j-1;
            }
            else
            {
                p0=2; p1=3;
            }

            int edgeIndex=m_container->getEdgeIndex(t[p0],t[p1]);
            // we must create the edge
            if (edgeIndex==-1)
            {
                sofa::helper::vector< Edge > v;
                Edge e1(t[p0],t[p1]);
                v.push_back(e1);

                addEdgesProcess((const sofa::helper::vector< Edge > &) v);

                edgeIndex=m_container->getEdgeIndex(t[p0],t[p1]);

                sofa::helper::vector< unsigned int > edgeIndexList;
                edgeIndexList.push_back(edgeIndex);
                addEdgesWarning((unsigned int)v.size(), v, edgeIndexList);
            }

            m_container->m_edgesInTetrahedron.resize(tetrahedronIndex+1);
            m_container->m_edgesInTetrahedron[tetrahedronIndex][j]= edgeIndex;
        }
    }

    if (m_container->hasTetrahedraAroundVertex())
    {
        for (unsigned int j=0; j<4; ++j)
        {
            sofa::helper::vector< unsigned int > &shell = m_container->getTetrahedraAroundVertexForModification( t[j] );
            shell.push_back( tetrahedronIndex );
        }
    }

    if (m_container->hasTetrahedraAroundEdge())
    {
        for (unsigned int j=0; j<6; ++j)
        {
            sofa::helper::vector< unsigned int > &shell = m_container->m_tetrahedraAroundEdge[m_container->m_edgesInTetrahedron[tetrahedronIndex][j]];
            shell.push_back( tetrahedronIndex );
        }
    }

    if (m_container->hasTetrahedraAroundTriangle())
    {
        for (unsigned int j=0; j<4; ++j)
        {
            sofa::helper::vector< unsigned int > &shell = m_container->m_tetrahedraAroundTriangle[m_container->m_trianglesInTetrahedron[tetrahedronIndex][j]];
            shell.push_back( tetrahedronIndex );
        }
    }

    m_tetrahedron.push_back(t);
}


void TetrahedronSetTopologyModifier::addTetrahedraProcess(const sofa::helper::vector< Tetrahedron > &tetrahedra)
{
    for (unsigned int i = 0; i < tetrahedra.size(); ++i)
    {
        addTetrahedronProcess(tetrahedra[i]);
    }
}


void TetrahedronSetTopologyModifier::addTetrahedraWarning(const unsigned int nTetrahedra,
        const sofa::helper::vector< Tetrahedron >& tetrahedraList,
        const sofa::helper::vector< unsigned int >& tetrahedraIndexList)
{
    m_container->setTetrahedronTopologyToDirty();
    // Warning that tetrahedra just got created
    TetrahedraAdded *e = new TetrahedraAdded(nTetrahedra, tetrahedraList, tetrahedraIndexList);
    addTopologyChange(e);
}


void TetrahedronSetTopologyModifier::addTetrahedraWarning(const unsigned int nTetrahedra,
        const sofa::helper::vector< Tetrahedron >& tetrahedraList,
        const sofa::helper::vector< unsigned int >& tetrahedraIndexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    m_container->setTetrahedronTopologyToDirty();
    // Warning that tetrahedra just got created
    TetrahedraAdded *e = new TetrahedraAdded(nTetrahedra, tetrahedraList, tetrahedraIndexList, ancestors, baryCoefs);
    addTopologyChange(e);
}


void TetrahedronSetTopologyModifier::removeTetrahedraWarning( sofa::helper::vector<unsigned int> &tetrahedra )
{
    m_container->setTetrahedronTopologyToDirty();
    /// sort vertices to remove in a descendent order
    std::sort( tetrahedra.begin(), tetrahedra.end(), std::greater<unsigned int>() );

    // Warning that these edges will be deleted
    TetrahedraRemoved *e=new TetrahedraRemoved(tetrahedra);
    addTopologyChange(e);
}

void TetrahedronSetTopologyModifier::removeTetrahedraProcess( const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedItems)
{
    if(!m_container->hasTetrahedra())
        return;

    bool removeIsolatedVertices = removeIsolatedItems && removeIsolated.getValue();
    bool removeIsolatedEdges = removeIsolatedItems && m_container->hasEdges();
    bool removeIsolatedTriangles = removeIsolatedItems && m_container->hasTriangles();

    if(removeIsolatedVertices)
    {
        if(!m_container->hasTetrahedraAroundVertex())
            m_container->createTetrahedraAroundVertexArray();
    }

    if(removeIsolatedEdges)
    {
        if(!m_container->hasTetrahedraAroundEdge())
            m_container->createTetrahedraAroundEdgeArray();
    }

    if(removeIsolatedTriangles)
    {
        if(!m_container->hasTetrahedraAroundTriangle())
            m_container->createTetrahedraAroundTriangleArray();
    }

    sofa::helper::vector<unsigned int> triangleToBeRemoved;
    sofa::helper::vector<unsigned int> edgeToBeRemoved;
    sofa::helper::vector<unsigned int> vertexToBeRemoved;

    helper::WriteAccessor< Data< sofa::helper::vector<Tetrahedron> > > m_tetrahedron = m_container->d_tetrahedron;

    unsigned int lastTetrahedron = m_container->getNumberOfTetrahedra() - 1;
    for (unsigned int i=0; i<indices.size(); ++i, --lastTetrahedron)
    {
        const Tetrahedron &t = m_tetrahedron[ indices[i] ];
        const Tetrahedron &h = m_tetrahedron[ lastTetrahedron ];

        if (m_container->hasTetrahedraAroundVertex())
        {
            for(unsigned int j=0; j<4; ++j)
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_tetrahedraAroundVertex[ t[j] ];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if(removeIsolatedVertices && shell.empty())
                {
                    vertexToBeRemoved.push_back(t[j]);
                }
            }
        }

        if(m_container->hasTetrahedraAroundEdge())
        {
            for(unsigned int j=0; j<6; ++j)
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_tetrahedraAroundEdge[ m_container->m_edgesInTetrahedron[indices[i]][j]];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if(removeIsolatedEdges && shell.empty())
                    edgeToBeRemoved.push_back(m_container->m_edgesInTetrahedron[indices[i]][j]);
            }
        }

        if(m_container->hasTetrahedraAroundTriangle())
        {
            for(unsigned int j=0; j<4; ++j)
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_tetrahedraAroundTriangle[ m_container->m_trianglesInTetrahedron[indices[i]][j]];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if(removeIsolatedTriangles && shell.empty())
                    triangleToBeRemoved.push_back(m_container->m_trianglesInTetrahedron[indices[i]][j]);
            }
        }

        // now updates the shell information of the edge formely at the end of the array
        if ( indices[i] < lastTetrahedron )
        {
            if (m_container->hasTetrahedraAroundVertex())
            {
                for(unsigned int j=0; j<4; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell = m_container->m_tetrahedraAroundVertex[ h[j] ];
                    replace(shell.begin(), shell.end(), lastTetrahedron, indices[i]);
                }
            }

            if (m_container->hasTetrahedraAroundEdge())
            {
                for(unsigned int j=0; j<6; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell =  m_container->m_tetrahedraAroundEdge[ m_container->m_edgesInTetrahedron[lastTetrahedron][j]];
                    replace(shell.begin(), shell.end(), lastTetrahedron, indices[i]);
                }
            }

            if (m_container->hasTetrahedraAroundTriangle())
            {
                for(unsigned int j=0; j<4; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell =  m_container->m_tetrahedraAroundTriangle[ m_container->m_trianglesInTetrahedron[lastTetrahedron][j]];
                    replace(shell.begin(), shell.end(), lastTetrahedron, indices[i]);
                }
            }
        }

        if (m_container->hasTrianglesInTetrahedron())
        {
            // removes the trianglesInTetrahedrons from the trianglesInTetrahedronArray
            m_container->m_trianglesInTetrahedron[ indices[i] ] = m_container->m_trianglesInTetrahedron[ lastTetrahedron ]; // overwriting with last valid value.
            m_container->m_trianglesInTetrahedron.resize( lastTetrahedron ); // resizing to erase multiple occurence of the tetrahedron.
        }

        if (m_container->hasEdgesInTetrahedron())
        {
            // removes the edgesInTetrahedrons from the edgesInTetrahedronArray
            m_container->m_edgesInTetrahedron[ indices[i] ] = m_container->m_edgesInTetrahedron[ lastTetrahedron ]; // overwriting with last valid value.
            m_container->m_edgesInTetrahedron.resize( lastTetrahedron ); // resizing to erase multiple occurence of the tetrahedron.
        }

        // removes the tetrahedron from the tetrahedronArray
        m_tetrahedron[ indices[i] ] = m_tetrahedron[ lastTetrahedron ]; // overwriting with last valid value.
        m_tetrahedron.resize( lastTetrahedron ); // resizing to erase multiple occurence of the tetrahedron.
    }

    if ( (!triangleToBeRemoved.empty()) || (!edgeToBeRemoved.empty()))
    {
        if (!triangleToBeRemoved.empty())
        {
            /// warn that triangles will be deleted
            removeTrianglesWarning(triangleToBeRemoved);
        }

        if (!edgeToBeRemoved.empty())
        {
            /// warn that edges will be deleted
            removeEdgesWarning(edgeToBeRemoved);
        }

        propagateTopologicalChanges();

        if (!triangleToBeRemoved.empty())
        {
            /// actually remove triangles without looking for isolated vertices
            removeTrianglesProcess(triangleToBeRemoved, false, false);

        }

        if (!edgeToBeRemoved.empty())
        {
            /// actually remove edges without looking for isolated vertices
            removeEdgesProcess(edgeToBeRemoved, false);
        }
    }

    if (!vertexToBeRemoved.empty())
    {
        removePointsWarning(vertexToBeRemoved);
        propagateTopologicalChanges();
        removePointsProcess(vertexToBeRemoved);
    }
}

void TetrahedronSetTopologyModifier::addPointsProcess(const unsigned int nPoints)
{
    // start by calling the parent's method.
    TriangleSetTopologyModifier::addPointsProcess( nPoints );

    if(m_container->hasTetrahedraAroundVertex())
        m_container->m_tetrahedraAroundVertex.resize( m_container->getNbPoints() );
}

void TetrahedronSetTopologyModifier::addEdgesProcess(const sofa::helper::vector< Edge > &edges)
{
    // start by calling the parent's method.
    TriangleSetTopologyModifier::addEdgesProcess( edges );

    if(m_container->hasTetrahedraAroundEdge())
        m_container->m_tetrahedraAroundEdge.resize( m_container->getNumberOfEdges() );
}

void TetrahedronSetTopologyModifier::addTrianglesProcess(const sofa::helper::vector< Triangle > &triangles)
{
    // start by calling the parent's method.
    TriangleSetTopologyModifier::addTrianglesProcess( triangles );
    if(m_container->hasTetrahedraAroundTriangle())
        m_container->m_tetrahedraAroundTriangle.resize( m_container->getNumberOfTriangles() );
}

void TetrahedronSetTopologyModifier::removePointsProcess(const sofa::helper::vector<unsigned int> &indices,
        const bool removeDOF)
{
    if(m_container->hasTetrahedra())
    {
        if(!m_container->hasTetrahedraAroundVertex())
        {
            m_container->createTetrahedraAroundVertexArray();
        }

        helper::WriteAccessor< Data< sofa::helper::vector<Tetrahedron> > > m_tetrahedron = m_container->d_tetrahedron;
        unsigned int lastPoint = m_container->getNbPoints() - 1;
        for (unsigned int i=0; i<indices.size(); ++i, --lastPoint)
        {
            // updating the edges connected to the point replacing the removed one:
            // for all edges connected to the last point
            for (sofa::helper::vector<unsigned int>::iterator itt=m_container->m_tetrahedraAroundVertex[lastPoint].begin();
                    itt!=m_container->m_tetrahedraAroundVertex[lastPoint].end(); ++itt)
            {
                unsigned int vertexIndex = m_container->getVertexIndexInTetrahedron(m_tetrahedron[(*itt)],lastPoint);
                m_tetrahedron[(*itt)][vertexIndex]=indices[i];
            }

            // updating the edge shell itself (change the old index for the new one)
            m_container->m_tetrahedraAroundVertex[ indices[i] ] = m_container->m_tetrahedraAroundVertex[ lastPoint ];
        }

        m_container->m_tetrahedraAroundVertex.resize( m_container->m_tetrahedraAroundVertex.size() - indices.size() );
    }

    // Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    // call the parent's method.
    TriangleSetTopologyModifier::removePointsProcess(  indices, removeDOF );
}

void TetrahedronSetTopologyModifier::removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedItems)
{
    if(!m_container->hasEdges()) // this method should only be called when edges exist
        return;

    if (m_container->hasEdgesInTetrahedron())
    {
        if(!m_container->hasTetrahedraAroundEdge())
            m_container->createTetrahedraAroundEdgeArray();

        unsigned int lastEdge = m_container->getNumberOfEdges() - 1;
        for (unsigned int i=0; i<indices.size(); ++i, --lastEdge)
        {
            for (sofa::helper::vector<unsigned int>::iterator itt=m_container->m_tetrahedraAroundEdge[lastEdge].begin();
                    itt!=m_container->m_tetrahedraAroundEdge[lastEdge].end(); ++itt)
            {
                unsigned int edgeIndex=m_container->getEdgeIndexInTetrahedron(m_container->m_edgesInTetrahedron[(*itt)],lastEdge);
                m_container->m_edgesInTetrahedron[(*itt)][edgeIndex]=indices[i];
            }

            // updating the edge shell itself (change the old index for the new one)
            m_container->m_tetrahedraAroundEdge[ indices[i] ] = m_container->m_tetrahedraAroundEdge[ lastEdge ];
        }

        m_container->m_tetrahedraAroundEdge.resize( m_container->m_tetrahedraAroundEdge.size() - indices.size() );
    }

    // call the parent's method.
    TriangleSetTopologyModifier::removeEdgesProcess( indices, removeIsolatedItems );
}

void TetrahedronSetTopologyModifier::removeTrianglesProcess( const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedEdges,
        const bool removeIsolatedPoints)
{
    if(!m_container->hasTriangles()) // this method should only be called when triangles exist
        return;

    if (m_container->hasTrianglesInTetrahedron())
    {
        if(!m_container->hasTetrahedraAroundTriangle())
            m_container->createTetrahedraAroundTriangleArray();

        size_t lastTriangle = m_container->m_tetrahedraAroundTriangle.size() - 1;
        for (unsigned int i = 0; i < indices.size(); ++i, --lastTriangle)
        {
            for (sofa::helper::vector<unsigned int>::iterator itt=m_container->m_tetrahedraAroundTriangle[lastTriangle].begin();
                    itt!=m_container->m_tetrahedraAroundTriangle[lastTriangle].end(); ++itt)
            {
                unsigned int triangleIndex=m_container->getTriangleIndexInTetrahedron(m_container->m_trianglesInTetrahedron[(*itt)],lastTriangle);
                m_container->m_trianglesInTetrahedron[(*itt)][triangleIndex] = indices[i];
            }

            // updating the triangle shell itself (change the old index for the new one)
            m_container->m_tetrahedraAroundTriangle[ indices[i] ] = m_container->m_tetrahedraAroundTriangle[ lastTriangle ];
        }
        m_container->m_tetrahedraAroundTriangle.resize( m_container->m_tetrahedraAroundTriangle.size() - indices.size() );
    }

    // call the parent's method.
    TriangleSetTopologyModifier::removeTrianglesProcess( indices, removeIsolatedEdges, removeIsolatedPoints );
}

void TetrahedronSetTopologyModifier::renumberPointsProcess( const sofa::helper::vector<unsigned int> &index,
        const sofa::helper::vector<unsigned int> &inv_index,
        const bool renumberDOF)
{
    if(m_container->hasTetrahedra())
    {
        helper::WriteAccessor< Data< sofa::helper::vector<Tetrahedron> > > m_tetrahedron = m_container->d_tetrahedron;
        if(m_container->hasTetrahedraAroundVertex())
        {
            sofa::helper::vector< sofa::helper::vector< unsigned int > > tetrahedraAroundVertex_cp = m_container->m_tetrahedraAroundVertex;
            for (unsigned int i = 0; i < index.size(); ++i)
            {
                m_container->m_tetrahedraAroundVertex[i] = tetrahedraAroundVertex_cp[ index[i] ];
            }
        }

        for (unsigned int i=0; i<m_tetrahedron.size(); ++i)
        {
            m_tetrahedron[i][0]  = inv_index[ m_tetrahedron[i][0]  ];
            m_tetrahedron[i][1]  = inv_index[ m_tetrahedron[i][1]  ];
            m_tetrahedron[i][2]  = inv_index[ m_tetrahedron[i][2]  ];
            m_tetrahedron[i][3]  = inv_index[ m_tetrahedron[i][3]  ];
        }
    }

    // call the parent's method.
    TriangleSetTopologyModifier::renumberPointsProcess( index, inv_index, renumberDOF );
}

void TetrahedronSetTopologyModifier::removeTetrahedra(const sofa::helper::vector<unsigned int> &tetrahedraIds)
{
    sofa::helper::vector<unsigned int> tetrahedraIds_filtered;
    for (unsigned int i = 0; i < tetrahedraIds.size(); i++)
    {
        if( tetrahedraIds[i] >= m_container->getNumberOfTetrahedra())
            std::cout << "Error: TetrahedronSetTopologyModifier::removeTetrahedra: tetrahedra: "<< tetrahedraIds[i] <<" is out of bound and won't be removed." << std::endl;
        else
            tetrahedraIds_filtered.push_back(tetrahedraIds[i]);
    }

    removeTetrahedraWarning(tetrahedraIds_filtered);

    // inform other objects that the triangles are going to be removed
    propagateTopologicalChanges();

    // now destroy the old tetrahedra.
    removeTetrahedraProcess(tetrahedraIds_filtered ,true);

    m_container->checkTopology();

    m_container->addRemovedTetraIndex(tetrahedraIds_filtered);
}

void TetrahedronSetTopologyModifier::removeItems(const sofa::helper::vector< unsigned int >& items)
{
    removeTetrahedra(items);
}

void TetrahedronSetTopologyModifier::renumberPoints( const sofa::helper::vector<unsigned int> &index,
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


void TetrahedronSetTopologyModifier::propagateTopologicalEngineChanges()
{
    if (m_container->beginChange() == m_container->endChange()) return; // nothing to do if no event is stored

    if (!m_container->isTetrahedronTopologyDirty()) // tetrahedron Data has not been touched
        return TriangleSetTopologyModifier::propagateTopologicalEngineChanges();

    std::list<sofa::core::topology::TopologyEngine *>::iterator it;

    for ( it = m_container->m_enginesList.begin(); it!=m_container->m_enginesList.end(); ++it)
    {
        sofa::core::topology::TopologyEngine* topoEngine = (*it);
        if (topoEngine->isDirty())
        {
#ifndef NDEBUG
            std::cout << "TetrahedronSetTopologyModifier::performing: " << topoEngine->getName() << std::endl;
#endif
            topoEngine->update();
        }
    }

    m_container->cleanTetrahedronTopologyFromDirty();
    TriangleSetTopologyModifier::propagateTopologicalEngineChanges();
}

} // namespace topology

} // namespace component

} // namespace sofa

