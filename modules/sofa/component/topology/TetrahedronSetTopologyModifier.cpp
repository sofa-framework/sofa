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
#include <sofa/component/topology/TetrahedronSetTopologyModifier.h>
#include <sofa/component/topology/TetrahedronSetTopologyChange.h>
#include <sofa/component/topology/TetrahedronSetTopologyContainer.h>
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
using namespace sofa::core::componentmodel::behavior;

const unsigned int tetrahedronEdgeArray[6][2] = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};


void TetrahedronSetTopologyModifier::init()
{
    TriangleSetTopologyModifier::init();

    this->getContext()->get(m_container);
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
    const unsigned int tetrahedronIndex = m_container->m_tetrahedron.size();

    if (m_container->hasTetrahedronTriangles())
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
                addTrianglesWarning( v.size(), v, triangleIndexList);
            }

            m_container->m_tetrahedronTriangle.resize(triangleIndex+1);
            m_container->m_tetrahedronTriangle[tetrahedronIndex][j]= triangleIndex;
        }
    }

    if (m_container->hasTetrahedronEdges())
    {
        for (unsigned int j=0; j<6; ++j)
        {
            int edgeIndex=m_container->getEdgeIndex(tetrahedronEdgeArray[j][0],
                    tetrahedronEdgeArray[j][1]);
            assert(edgeIndex!= -1);

            m_container->m_tetrahedronEdge.resize(edgeIndex+1);
            m_container->m_tetrahedronEdge[tetrahedronIndex][j]= edgeIndex;
        }
    }

    if (m_container->hasTetrahedronVertexShell())
    {
        for (unsigned int j=0; j<4; ++j)
        {
            sofa::helper::vector< unsigned int > &shell = m_container->getTetrahedronVertexShellForModification( t[j] );
            shell.push_back( tetrahedronIndex );
        }
    }

    if (m_container->hasTetrahedronEdgeShell())
    {
        for (unsigned int j=0; j<6; ++j)
        {
            sofa::helper::vector< unsigned int > &shell = m_container->m_tetrahedronEdgeShell[m_container->m_tetrahedronEdge[tetrahedronIndex][j]];
            shell.push_back( tetrahedronIndex );
        }
    }

    if (m_container->hasTetrahedronTriangleShell())
    {
        for (unsigned int j=0; j<4; ++j)
        {
            sofa::helper::vector< unsigned int > &shell = m_container->m_tetrahedronTriangleShell[m_container->m_tetrahedronTriangle[tetrahedronIndex][j]];
            shell.push_back( tetrahedronIndex );
        }
    }

    m_container->m_tetrahedron.push_back(t);
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
    // Warning that tetrahedra just got created
    TetrahedraAdded *e = new TetrahedraAdded(nTetrahedra, tetrahedraList, tetrahedraIndexList, ancestors, baryCoefs);
    addTopologyChange(e);
}


void TetrahedronSetTopologyModifier::removeTetrahedraWarning( sofa::helper::vector<unsigned int> &tetrahedra )
{
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

    bool removeIsolatedVertices = removeIsolatedItems;
    bool removeIsolatedEdges = removeIsolatedItems && m_container->hasEdges();
    bool removeIsolatedTriangles = removeIsolatedItems && m_container->hasTriangles();

    if(removeIsolatedVertices)
    {
        if(!m_container->hasTetrahedronVertexShell())
            m_container->createTetrahedronVertexShellArray();
    }

    if(removeIsolatedEdges)
    {
        if(!m_container->hasTetrahedronEdgeShell())
            m_container->createTetrahedronEdgeShellArray();
    }

    if(removeIsolatedTriangles)
    {
        if(!m_container->hasTetrahedronTriangleShell())
            m_container->createTetrahedronTriangleShellArray();
    }

    sofa::helper::vector<unsigned int> triangleToBeRemoved;
    sofa::helper::vector<unsigned int> edgeToBeRemoved;
    sofa::helper::vector<unsigned int> vertexToBeRemoved;

    unsigned int lastTetrahedron = m_container->getNumberOfTetrahedra() - 1;
    for (unsigned int i=0; i<indices.size(); ++i, --lastTetrahedron)
    {
        Tetrahedron &t = m_container->m_tetrahedron[ indices[i] ];
        Tetrahedron &h = m_container->m_tetrahedron[ lastTetrahedron ];

        if (m_container->hasTetrahedronVertexShell())
        {
            for(unsigned int j=0; j<4; ++j)
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_tetrahedronVertexShell[ t[j] ];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if(removeIsolatedVertices && shell.empty())
                    vertexToBeRemoved.push_back(t[j]);
            }
        }

        if(m_container->hasTetrahedronEdgeShell())
        {
            for(unsigned int j=0; j<6; ++j)
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_tetrahedronEdgeShell[ m_container->m_tetrahedronEdge[indices[i]][j]];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if(removeIsolatedEdges && shell.empty())
                    edgeToBeRemoved.push_back(m_container->m_tetrahedronEdge[indices[i]][j]);
            }
        }

        if(m_container->hasTetrahedronTriangleShell())
        {
            for(unsigned int j=0; j<4; ++j)
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_tetrahedronTriangleShell[ m_container->m_tetrahedronTriangle[indices[i]][j]];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if(removeIsolatedTriangles && shell.empty())
                    triangleToBeRemoved.push_back(m_container->m_tetrahedronTriangle[indices[i]][j]);
            }
        }

        // now updates the shell information of the edge formely at the end of the array
        if ( indices[i] < lastTetrahedron )
        {
            if (m_container->hasTetrahedronVertexShell())
            {
                for(unsigned int j=0; j<4; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell = m_container->m_tetrahedronVertexShell[ h[j] ];
                    replace(shell.begin(), shell.end(), lastTetrahedron, indices[i]);
                }
            }

            if (m_container->hasTetrahedronEdgeShell())
            {
                for(unsigned int j=0; j<6; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell =  m_container->m_tetrahedronEdgeShell[ m_container->m_tetrahedronEdge[lastTetrahedron][j]];
                    replace(shell.begin(), shell.end(), lastTetrahedron, indices[i]);
                }
            }

            if (m_container->hasTetrahedronTriangleShell())
            {
                for(unsigned int j=0; j<4; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell =  m_container->m_tetrahedronTriangleShell[ m_container->m_tetrahedronTriangle[lastTetrahedron][j]];
                    replace(shell.begin(), shell.end(), lastTetrahedron, indices[i]);
                }
            }
        }

        if (m_container->hasTetrahedronTriangles())
        {
            // removes the tetrahedronTriangles from the tetrahedronTriangleArray
            m_container->m_tetrahedronTriangle[ indices[i] ] = m_container->m_tetrahedronTriangle[ lastTetrahedron ]; // overwriting with last valid value.
            m_container->m_tetrahedronTriangle.resize( lastTetrahedron ); // resizing to erase multiple occurence of the tetrahedron.
        }

        if (m_container->hasTetrahedronEdges())
        {
            // removes the tetrahedronEdges from the tetrahedronEdgeArray
            m_container->m_tetrahedronEdge[ indices[i] ] = m_container->m_tetrahedronEdge[ lastTetrahedron ]; // overwriting with last valid value.
            m_container->m_tetrahedronEdge.resize( lastTetrahedron ); // resizing to erase multiple occurence of the tetrahedron.
        }

        // removes the tetrahedron from the tetrahedronArray
        m_container->m_tetrahedron[ indices[i] ] = m_container->m_tetrahedron[ lastTetrahedron ]; // overwriting with last valid value.
        m_container->m_tetrahedron.resize( lastTetrahedron ); // resizing to erase multiple occurence of the tetrahedron.
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

    if(m_container->hasTetrahedronVertexShell())
        m_container->m_tetrahedronVertexShell.resize( m_container->getNbPoints() );
}

void TetrahedronSetTopologyModifier::addEdgesProcess(const sofa::helper::vector< Edge > &edges)
{
    // start by calling the parent's method.
    TriangleSetTopologyModifier::addEdgesProcess( edges );

    if(m_container->hasTetrahedronEdgeShell())
        m_container->m_tetrahedronEdgeShell.resize( m_container->getNumberOfEdges() );
}

void TetrahedronSetTopologyModifier::addTrianglesProcess(const sofa::helper::vector< Triangle > &triangles)
{
    // start by calling the parent's method.
    TriangleSetTopologyModifier::addTrianglesProcess( triangles );

    if(m_container->hasTetrahedronTriangleShell())
        m_container->m_tetrahedronTriangleShell.resize( m_container->getNumberOfTriangles() );
}

void TetrahedronSetTopologyModifier::removePointsProcess( sofa::helper::vector<unsigned int> &indices,
        const bool removeDOF)
{
    if(m_container->hasTetrahedra())
    {
        if(!m_container->hasTetrahedronVertexShell())
        {
            m_container->createTetrahedronVertexShellArray();
        }

        unsigned int lastPoint = m_container->getNbPoints() - 1;
        for (unsigned int i=0; i<indices.size(); ++i, --lastPoint)
        {
            // updating the edges connected to the point replacing the removed one:
            // for all edges connected to the last point
            for (sofa::helper::vector<unsigned int>::iterator itt=m_container->m_tetrahedronVertexShell[lastPoint].begin();
                    itt!=m_container->m_tetrahedronVertexShell[lastPoint].end(); ++itt)
            {
                unsigned int vertexIndex = m_container->getVertexIndexInTetrahedron(m_container->m_tetrahedron[(*itt)],lastPoint);
                m_container->m_tetrahedron[(*itt)][vertexIndex]=indices[i];
            }

            // updating the edge shell itself (change the old index for the new one)
            m_container->m_tetrahedronVertexShell[ indices[i] ] = m_container->m_tetrahedronVertexShell[ lastPoint ];
        }

        m_container->m_tetrahedronVertexShell.resize( m_container->m_tetrahedronVertexShell.size() - indices.size() );
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

    if (m_container->hasTetrahedronEdges())
    {
        if(!m_container->hasTetrahedronEdgeShell())
            m_container->createTetrahedronEdgeShellArray();

        unsigned int lastEdge = m_container->getNumberOfEdges() - 1;
        for (unsigned int i=0; i<indices.size(); ++i, --lastEdge)
        {
            for (sofa::helper::vector<unsigned int>::iterator itt=m_container->m_tetrahedronEdgeShell[lastEdge].begin();
                    itt!=m_container->m_tetrahedronEdgeShell[lastEdge].end(); ++itt)
            {
                unsigned int edgeIndex=m_container->getEdgeIndexInTetrahedron(m_container->m_tetrahedronEdge[(*itt)],lastEdge);
                m_container->m_tetrahedronEdge[(*itt)][edgeIndex]=indices[i];
            }

            // updating the edge shell itself (change the old index for the new one)
            m_container->m_tetrahedronEdgeShell[ indices[i] ] = m_container->m_tetrahedronEdgeShell[ lastEdge ];
        }

        m_container->m_tetrahedronEdgeShell.resize( m_container->m_tetrahedronEdgeShell.size() - indices.size() );
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

    if (m_container->hasTetrahedronTriangles())
    {
        if(!m_container->hasTetrahedronTriangleShell())
            m_container->createTetrahedronTriangleShellArray();

        unsigned int lastTriangle = m_container->m_tetrahedronTriangleShell.size() - 1;
        for (unsigned int i = 0; i < indices.size(); ++i, --lastTriangle)
        {
            for (sofa::helper::vector<unsigned int>::iterator itt=m_container->m_tetrahedronTriangleShell[lastTriangle].begin();
                    itt!=m_container->m_tetrahedronTriangleShell[lastTriangle].end(); ++itt)
            {
                unsigned int triangleIndex=m_container->getTriangleIndexInTetrahedron(m_container->m_tetrahedronTriangle[(*itt)],lastTriangle);
                m_container->m_tetrahedronTriangle[(*itt)][triangleIndex] = indices[i];
            }

            // updating the triangle shell itself (change the old index for the new one)
            m_container->m_tetrahedronTriangleShell[ indices[i] ] = m_container->m_tetrahedronTriangleShell[ lastTriangle ];
        }
        m_container->m_tetrahedronTriangleShell.resize( m_container->m_tetrahedronTriangleShell.size() - indices.size() );
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
        if(m_container->hasTetrahedronVertexShell())
        {
            sofa::helper::vector< sofa::helper::vector< unsigned int > > tetrahedronVertexShell_cp = m_container->m_tetrahedronVertexShell;
            for (unsigned int i = 0; i < index.size(); ++i)
            {
                m_container->m_tetrahedronVertexShell[i] = tetrahedronVertexShell_cp[ index[i] ];
            }
        }

        for (unsigned int i=0; i<m_container->m_tetrahedron.size(); ++i)
        {
            m_container->m_tetrahedron[i][0]  = inv_index[ m_container->m_tetrahedron[i][0]  ];
            m_container->m_tetrahedron[i][1]  = inv_index[ m_container->m_tetrahedron[i][1]  ];
            m_container->m_tetrahedron[i][2]  = inv_index[ m_container->m_tetrahedron[i][2]  ];
            m_container->m_tetrahedron[i][3]  = inv_index[ m_container->m_tetrahedron[i][3]  ];
        }
    }

    // call the parent's method.
    TriangleSetTopologyModifier::renumberPointsProcess( index, inv_index, renumberDOF );
}

void TetrahedronSetTopologyModifier::removeTetrahedra(sofa::helper::vector< unsigned int >& tetrahedra)
{
    removeTetrahedraWarning(tetrahedra);

    // inform other objects that the triangles are going to be removed
    propagateTopologicalChanges();

    // now destroy the old tetrahedra.
    removeTetrahedraProcess(  tetrahedra ,true);

    m_container->checkTopology();
}

void TetrahedronSetTopologyModifier::removeItems(sofa::helper::vector< unsigned int >& items)
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

} // namespace topology

} // namespace component

} // namespace sofa

