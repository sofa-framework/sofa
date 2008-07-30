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


TetrahedronSetTopologyContainer* TetrahedronSetTopologyModifier::getTetrahedronSetTopologyContainer() const
{
    return static_cast<TetrahedronSetTopologyContainer* > (this->m_topologyContainer);
}


void TetrahedronSetTopologyModifier::addTetrahedron(Tetrahedron t)
{
    TetrahedronSetTopologyContainer * container = getTetrahedronSetTopologyContainer();

    // check if the 3 vertices are different
    assert(t[0]!=t[1]);
    assert(t[0]!=t[2]);
    assert(t[0]!=t[3]);
    assert(t[1]!=t[2]);
    assert(t[1]!=t[3]);
    assert(t[2]!=t[3]);

    // check if there already exists a tetrahedron with the same indices
    // assert(container->getTetrahedronIndex(t[0], t[1], t[2], t[3])== -1);

    unsigned int tetrahedronIndex = container->m_tetrahedron.size();

    if (container->hasTetrahedronTriangles())
    {
        for (unsigned int j=0; j<4; ++j)
        {
            int triangleIndex = container->getTriangleIndex(t[(j+1)%4], t[(j+2)%4], t[(j+3)%4]);

            if(triangleIndex == -1)
            {
                // first create the traingle
                sofa::helper::vector< Triangle > v;
                Triangle e1 (t[(j+1)%4],t[(j+2)%4],t[(j+3)%4]);
                v.push_back(e1);

                addTrianglesProcess((const sofa::helper::vector< Triangle > &) v);

                triangleIndex = container->getTriangleIndex(t[(j+1)%4], t[(j+2)%4], t[(j+3)%4]);

                sofa::helper::vector< unsigned int > triangleIndexList;
                triangleIndexList.push_back(triangleIndex);
                this->addTrianglesWarning( v.size(), v, triangleIndexList);
            }

            container->m_tetrahedronTriangle.resize(triangleIndex+1);
            container->m_tetrahedronTriangle[tetrahedronIndex][j]= triangleIndex;
        }
    }

    if (container->hasTetrahedronEdges())
    {
        for (unsigned int j=0; j<6; ++j)
        {
            int edgeIndex=container->getEdgeIndex(tetrahedronEdgeArray[j][0],
                    tetrahedronEdgeArray[j][1]);
            assert(edgeIndex!= -1);

            container->m_tetrahedronEdge.resize(edgeIndex+1);
            container->m_tetrahedronEdge[tetrahedronIndex][j]= edgeIndex;
        }
    }

    if (container->hasTetrahedronVertexShell())
    {
        for (unsigned int j=0; j<4; ++j)
        {
            sofa::helper::vector< unsigned int > &shell = container->getTetrahedronVertexShellForModification( t[j] );
            shell.push_back( tetrahedronIndex );
        }
    }

    if (container->hasTetrahedronEdgeShell())
    {
        for (unsigned int j=0; j<6; ++j)
        {
            sofa::helper::vector< unsigned int > &shell = container->m_tetrahedronEdgeShell[container->m_tetrahedronEdge[tetrahedronIndex][j]];
            shell.push_back( tetrahedronIndex );
        }
    }

    if (container->hasTetrahedronTriangleShell())
    {
        for (unsigned int j=0; j<4; ++j)
        {
            sofa::helper::vector< unsigned int > &shell = container->m_tetrahedronTriangleShell[container->m_tetrahedronTriangle[tetrahedronIndex][j]];
            shell.push_back( tetrahedronIndex );
        }
    }

    container->m_tetrahedron.push_back(t);
}


void TetrahedronSetTopologyModifier::addTetrahedraProcess(const sofa::helper::vector< Tetrahedron > &tetrahedra)
{
    for (unsigned int i = 0; i < tetrahedra.size(); ++i)
    {
        addTetrahedron(tetrahedra[i]);
    }
}


void TetrahedronSetTopologyModifier::addTetrahedraWarning(const unsigned int nTetrahedra,
        const sofa::helper::vector< Tetrahedron >& tetrahedraList,
        const sofa::helper::vector< unsigned int >& tetrahedraIndexList)
{
    // Warning that tetrahedra just got created
    TetrahedraAdded *e = new TetrahedraAdded(nTetrahedra, tetrahedraList, tetrahedraIndexList);
    this->addTopologyChange(e);
}


void TetrahedronSetTopologyModifier::addTetrahedraWarning(const unsigned int nTetrahedra,
        const sofa::helper::vector< Tetrahedron >& tetrahedraList,
        const sofa::helper::vector< unsigned int >& tetrahedraIndexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    // Warning that tetrahedra just got created
    TetrahedraAdded *e = new TetrahedraAdded(nTetrahedra, tetrahedraList, tetrahedraIndexList, ancestors, baryCoefs);
    this->addTopologyChange(e);
}


void TetrahedronSetTopologyModifier::removeTetrahedraWarning( sofa::helper::vector<unsigned int> &tetrahedra )
{
    /// sort vertices to remove in a descendent order
    std::sort( tetrahedra.begin(), tetrahedra.end(), std::greater<unsigned int>() );

    // Warning that these edges will be deleted
    TetrahedraRemoved *e=new TetrahedraRemoved(tetrahedra);
    this->addTopologyChange(e);
}


void TetrahedronSetTopologyModifier::removeTetrahedraProcess( const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedItems)
{
    TetrahedronSetTopologyContainer * container = getTetrahedronSetTopologyContainer();

    bool removeIsolatedVertices = removeIsolatedItems;
    bool removeIsolatedEdges = removeIsolatedItems && container->hasEdges();
    bool removeIsolatedTriangles = removeIsolatedItems && container->hasTriangles();

    if(removeIsolatedVertices)
    {
        if(!container->hasTetrahedronVertexShell())
            container->createTetrahedronVertexShellArray();
    }

    if(removeIsolatedEdges)
    {
        if(!container->hasTetrahedronEdgeShell())
            container->createTetrahedronEdgeShellArray();
    }

    if(removeIsolatedTriangles)
    {
        if(!container->hasTetrahedronTriangleShell())
            container->createTetrahedronTriangleShellArray();
    }

    sofa::helper::vector<unsigned int> triangleToBeRemoved;
    sofa::helper::vector<unsigned int> edgeToBeRemoved;
    sofa::helper::vector<unsigned int> vertexToBeRemoved;

    unsigned int lastTetrahedron = container->getNumberOfTetrahedra() - 1;
    for (unsigned int i=0; i<indices.size(); ++i, --lastTetrahedron)
    {
        Tetrahedron &t = container->m_tetrahedron[ indices[i] ];
        Tetrahedron &h = container->m_tetrahedron[ lastTetrahedron ];

        // first check that the tetrahedron vertex shell array has been initialized
        if (container->hasTetrahedronVertexShell())
        {
            for(unsigned int j=0; j<4; ++j)
            {
                sofa::helper::vector< unsigned int > &shell = container->m_tetrahedronVertexShell[ t[j] ];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if ((removeIsolatedItems) && (shell.empty()))
                    vertexToBeRemoved.push_back(t[j]);
            }
        }

        /** first check that the tetrahedron edge shell array has been initialized */
        if (container->hasTetrahedronEdgeShell())
        {
            for(unsigned int j=0; j<6; ++j)
            {
                sofa::helper::vector< unsigned int > &shell = container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[indices[i]][j]];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if ((removeIsolatedItems) && (shell.empty()))
                    edgeToBeRemoved.push_back(container->m_tetrahedronEdge[indices[i]][j]);
            }
        }

        /** first check that the tetrahedron triangle shell array has been initialized */
        if (container->hasTetrahedronTriangleShell())
        {
            for(unsigned int j=0; j<4; ++j)
            {
                sofa::helper::vector< unsigned int > &shell = container->m_tetrahedronTriangleShell[ container->m_tetrahedronTriangle[indices[i]][j]];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if ((removeIsolatedItems) && (shell.empty()))
                    triangleToBeRemoved.push_back(container->m_tetrahedronTriangle[indices[i]][j]);
            }
        }

        // now updates the shell information of the edge formely at the end of the array
        // first check that the edge shell array has been initialized
        if ( indices[i] < lastTetrahedron )
        {
            if (container->hasTetrahedronVertexShell())
            {
                for(unsigned int j=0; j<4; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell = container->m_tetrahedronVertexShell[ h[j] ];
                    replace(shell.begin(), shell.end(), lastTetrahedron, indices[i]);
                }
            }

            if (container->hasTetrahedronEdgeShell())
            {
                for(unsigned int j=0; j<6; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell =  container->m_tetrahedronEdgeShell[ container->m_tetrahedronEdge[lastTetrahedron][j]];
                    replace(shell.begin(), shell.end(), lastTetrahedron, indices[i]);
                }
            }

            if (container->hasTetrahedronTriangleShell())
            {
                for(unsigned int j=0; j<4; ++j)
                {
                    sofa::helper::vector< unsigned int > &shell =  container->m_tetrahedronTriangleShell[ container->m_tetrahedronTriangle[lastTetrahedron][j]];
                    replace(shell.begin(), shell.end(), lastTetrahedron, indices[i]);
                }
            }
        }

        if (container->hasTetrahedronTriangles())
        {
            // removes the tetrahedronTriangles from the tetrahedronTriangleArray
            container->m_tetrahedronTriangle[ indices[i] ] = container->m_tetrahedronTriangle[ lastTetrahedron ]; // overwriting with last valid value.
            container->m_tetrahedronTriangle.resize( lastTetrahedron ); // resizing to erase multiple occurence of the tetrahedron.
        }

        if (container->hasTetrahedronEdges())
        {
            // removes the tetrahedronEdges from the tetrahedronEdgeArray
            container->m_tetrahedronEdge[ indices[i] ] = container->m_tetrahedronEdge[ lastTetrahedron ]; // overwriting with last valid value.
            container->m_tetrahedronEdge.resize( lastTetrahedron ); // resizing to erase multiple occurence of the tetrahedron.
        }

        // removes the tetrahedron from the tetrahedronArray
        container->m_tetrahedron[ indices[i] ] = container->m_tetrahedron[ lastTetrahedron ]; // overwriting with last valid value.
        container->m_tetrahedron.resize( lastTetrahedron ); // resizing to erase multiple occurence of the tetrahedron.
    }

    if ( (!triangleToBeRemoved.empty()) || (!edgeToBeRemoved.empty()))
    {
        if (!triangleToBeRemoved.empty())
        {
            /// warn that triangles will be deleted
            this->removeTrianglesWarning(triangleToBeRemoved);
        }

        if (!edgeToBeRemoved.empty())
        {
            /// warn that edges will be deleted
            this->removeEdgesWarning(edgeToBeRemoved);
        }

        /// propagate to all components
        container->propagateTopologicalChanges();

        if (!triangleToBeRemoved.empty())
        {
            /// actually remove triangles without looking for isolated vertices
            this->removeTrianglesProcess(triangleToBeRemoved, false, false);

        }

        if (!edgeToBeRemoved.empty())
        {
            /// actually remove edges without looking for isolated vertices
            this->removeEdgesProcess(edgeToBeRemoved, false);
        }
    }

    if (!vertexToBeRemoved.empty())
    {
        this->removePointsWarning(vertexToBeRemoved);
        container->propagateTopologicalChanges();
        this->removePointsProcess(vertexToBeRemoved);
    }
}

void TetrahedronSetTopologyModifier::addPointsProcess(const unsigned int nPoints,
        const bool addDOF)
{
    // start by calling the parent's method.
    // TODO : only if triangles exist, otherwise call the EdgeSet or PointSet method respectively
    TriangleSetTopologyModifier::addPointsProcess( nPoints, addDOF );

    TetrahedronSetTopologyContainer * container = getTetrahedronSetTopologyContainer();

    container->m_tetrahedronVertexShell.resize( container->m_tetrahedronVertexShell.size() + nPoints );
}

void TetrahedronSetTopologyModifier::addPointsProcess(const unsigned int nPoints,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
        const bool addDOF)
{
    // start by calling the parent's method.
    // TODO : only if triangles exist, otherwise call the EdgeSet or PointSet method respectively
    TriangleSetTopologyModifier::addPointsProcess( nPoints, ancestors, baryCoefs, addDOF );

    TetrahedronSetTopologyContainer * container = getTetrahedronSetTopologyContainer();

    container->m_tetrahedronVertexShell.resize( container->m_tetrahedronVertexShell.size() + nPoints );
}

void TetrahedronSetTopologyModifier::addEdgesProcess(const sofa::helper::vector< Edge > &edges)
{
    // start by calling the parent's method.
    // TODO : only if triangles exist, otherwise call the EdgeSet or PointSet method respectively
    TriangleSetTopologyModifier::addEdgesProcess( edges );

    TetrahedronSetTopologyContainer * container = getTetrahedronSetTopologyContainer();

    container->m_tetrahedronEdgeShell.resize( container->m_tetrahedronEdgeShell.size() + edges.size() );
}

void TetrahedronSetTopologyModifier::addTrianglesProcess(const sofa::helper::vector< Triangle > &triangles)
{
    // start by calling the parent's method.
    // TODO : only if triangles exist
    TriangleSetTopologyModifier::addTrianglesProcess( triangles );

    TetrahedronSetTopologyContainer * container = getTetrahedronSetTopologyContainer();

    container->m_tetrahedronTriangleShell.resize( container->m_tetrahedronTriangleShell.size() + triangles.size() );
}

void TetrahedronSetTopologyModifier::removePointsProcess( sofa::helper::vector<unsigned int> &indices,
        const bool removeDOF)
{
    TetrahedronSetTopologyContainer * container = getTetrahedronSetTopologyContainer();

    if(!container->hasTetrahedronVertexShell())
    {
        container->createTetrahedronVertexShellArray();
    }

    unsigned int lastPoint = container->m_tetrahedronVertexShell.size() - 1;
    for (unsigned int i=0; i<indices.size(); ++i, --lastPoint)
    {
        // updating the edges connected to the point replacing the removed one:
        // for all edges connected to the last point
        for (sofa::helper::vector<unsigned int>::iterator itt=container->m_tetrahedronVertexShell[lastPoint].begin();
                itt!=container->m_tetrahedronVertexShell[lastPoint].end(); ++itt)
        {
            int vertexIndex = container->getVertexIndexInTetrahedron(container->m_tetrahedron[(*itt)],lastPoint);
            assert(vertexIndex!= -1);
            container->m_tetrahedron[(*itt)][(unsigned int)vertexIndex]=indices[i];
        }

        // updating the edge shell itself (change the old index for the new one)
        container->m_tetrahedronVertexShell[ indices[i] ] = container->m_tetrahedronVertexShell[ lastPoint ];
    }

    container->m_tetrahedronVertexShell.resize( container->m_tetrahedronVertexShell.size() - indices.size() );

    // Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    // call the parent's method.
    // TODO : only if triangles exist, otherwise call the EdgeSet or PointSet method respectively
    TriangleSetTopologyModifier::removePointsProcess(  indices, removeDOF );
}

void TetrahedronSetTopologyModifier::removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedItems)
{
    TetrahedronSetTopologyContainer * container = getTetrahedronSetTopologyContainer();

    if(!container->hasEdges()) // TODO : this method should only be called when edges exist
        return;

    if (container->hasTetrahedronEdges())
    {
        if(!container->hasTetrahedronEdgeShell())
            container->createTetrahedronEdgeShellArray();

        unsigned int lastEdge = container->getNumberOfEdges() - 1;
        for (unsigned int i=0; i<indices.size(); ++i, --lastEdge)
        {
            for (sofa::helper::vector<unsigned int>::iterator itt=container->m_tetrahedronEdgeShell[lastEdge].begin();
                    itt!=container->m_tetrahedronEdgeShell[lastEdge].end(); ++itt)
            {
                int edgeIndex=container->getEdgeIndexInTetrahedron(container->m_tetrahedronEdge[(*itt)],lastEdge);
                assert((int)edgeIndex!= -1);
                container->m_tetrahedronEdge[(*itt)][(unsigned int) edgeIndex]=indices[i];
            }

            // updating the edge shell itself (change the old index for the new one)
            container->m_tetrahedronEdgeShell[ indices[i] ] = container->m_tetrahedronEdgeShell[ lastEdge ];
        }

        container->m_tetrahedronEdgeShell.resize( container->m_tetrahedronEdgeShell.size() - indices.size() );
    }

    // call the parent's method.
    // TODO : only if triangles exist, otherwise call the EdgeSet or PointSet method respectively
    TriangleSetTopologyModifier::removeEdgesProcess( indices, removeIsolatedItems );
}

void TetrahedronSetTopologyModifier::removeTrianglesProcess( const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedEdges,
        const bool removeIsolatedPoints)
{
    TetrahedronSetTopologyContainer * container = getTetrahedronSetTopologyContainer();

    if(!container->hasTriangles()) // TODO : this method should only be called when triangles exist
        return;

    if (container->hasTetrahedronTriangles())
    {
        if(!container->hasTetrahedronTriangleShell())
            container->createTetrahedronTriangleShellArray();

        unsigned int lastTriangle = container->m_tetrahedronTriangleShell.size() - 1;
        for (unsigned int i = 0; i < indices.size(); ++i, --lastTriangle)
        {
            for (sofa::helper::vector<unsigned int>::iterator itt=container->m_tetrahedronTriangleShell[lastTriangle].begin();
                    itt!=container->m_tetrahedronTriangleShell[lastTriangle].end(); ++itt)
            {
                int triangleIndex=container->getTriangleIndexInTetrahedron(container->m_tetrahedronTriangle[(*itt)],lastTriangle);
                assert((int)triangleIndex!= -1);
                container->m_tetrahedronTriangle[(*itt)][(unsigned int)triangleIndex] = indices[i];
            }

            // updating the triangle shell itself (change the old index for the new one)
            container->m_tetrahedronTriangleShell[ indices[i] ] = container->m_tetrahedronTriangleShell[ lastTriangle ];
        }

        container->m_tetrahedronTriangleShell.resize( container->m_tetrahedronTriangleShell.size() - indices.size() );
    }

    // call the parent's method.
    // TODO : only if triangles exist, otherwise call the EdgeSet or PointSet method respectively
    TriangleSetTopologyModifier::removeTrianglesProcess( indices, removeIsolatedEdges, removeIsolatedPoints );
}

void TetrahedronSetTopologyModifier::renumberPointsProcess( const sofa::helper::vector<unsigned int> &index,
        const sofa::helper::vector<unsigned int> &inv_index,
        const bool renumberDOF)
{
    TetrahedronSetTopologyContainer * container = getTetrahedronSetTopologyContainer();

    if(container->hasTetrahedronVertexShell())
    {
        sofa::helper::vector< sofa::helper::vector< unsigned int > > tetrahedronVertexShell_cp = container->m_tetrahedronVertexShell;
        for (unsigned int i = 0; i < index.size(); ++i)
        {
            container->m_tetrahedronVertexShell[i] = tetrahedronVertexShell_cp[ index[i] ];
        }
    }

    for (unsigned int i=0; i<container->m_tetrahedron.size(); ++i)
    {
        container->m_tetrahedron[i][0]  = inv_index[ container->m_tetrahedron[i][0]  ];
        container->m_tetrahedron[i][1]  = inv_index[ container->m_tetrahedron[i][1]  ];
        container->m_tetrahedron[i][2]  = inv_index[ container->m_tetrahedron[i][2]  ];
        container->m_tetrahedron[i][3]  = inv_index[ container->m_tetrahedron[i][3]  ];
    }

    // call the parent's method.
    // TODO : only if triangles exist, otherwise call the EdgeSet or PointSet method respectively
    TriangleSetTopologyModifier::renumberPointsProcess( index, inv_index, renumberDOF );
}

} // namespace topology

} // namespace component

} // namespace sofa

