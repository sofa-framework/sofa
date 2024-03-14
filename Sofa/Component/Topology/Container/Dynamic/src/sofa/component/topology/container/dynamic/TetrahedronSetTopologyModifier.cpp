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
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyModifier.h>

#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyContainer.h>
#include <sofa/core/topology/TopologyChange.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/core/topology/TopologyHandler.h>

#include <algorithm>
#include <sofa/helper/ScopedAdvancedTimer.h>


namespace sofa::component::topology::container::dynamic
{
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

    if(!m_container)
    {
        msg_error() << "TetrahedronSetTopologyContainer not found in current node: " << this->getContext()->getName();
        d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
}

void TetrahedronSetTopologyModifier::reinit()
{
    TriangleSetTopologyModifier::reinit();
}


void TetrahedronSetTopologyModifier::addTetrahedra(const sofa::type::vector<Tetrahedron> &tetrahedra)
{
    const size_t ntetra = m_container->getNbTetrahedra();

    /// effectively add triangles in the topology container
    addTetrahedraProcess(tetrahedra);

    sofa::type::vector<TetrahedronID> tetrahedraIndex;
    tetrahedraIndex.reserve(tetrahedra.size());

    for (size_t i=0; i<tetrahedra.size(); ++i)
        tetrahedraIndex.push_back(TetrahedronID(ntetra+i));

    // add topology event in the stack of topological events
    addTetrahedraWarning(tetrahedra.size(), tetrahedra, tetrahedraIndex);

    // inform other objects that the edges are already added
    propagateTopologicalChanges();
}


void TetrahedronSetTopologyModifier::addTetrahedra(const sofa::type::vector<Tetrahedron> &tetrahedra,
        const sofa::type::vector<sofa::type::vector<TetrahedronID> > &ancestors,
        const sofa::type::vector<sofa::type::vector<SReal> > &baryCoefs)
{
    const size_t ntetra = m_container->getNbTetrahedra();

    /// effectively add triangles in the topology container
    addTetrahedraProcess(tetrahedra);

    sofa::type::vector<TetrahedronID> tetrahedraIndex;
    tetrahedraIndex.reserve(tetrahedra.size());

    for (size_t i=0; i<tetrahedra.size(); ++i)
        tetrahedraIndex.push_back(TetrahedronID(ntetra+i));

    // add topology event in the stack of topological events
    addTetrahedraWarning(tetrahedra.size(), tetrahedra, tetrahedraIndex, ancestors, baryCoefs);

    // inform other objects that the edges are already added
    propagateTopologicalChanges();
}

void TetrahedronSetTopologyModifier::addTetrahedronProcess(Tetrahedron t)
{
	if (m_container->d_checkTopology.getValue())
	{
		// check if the 3 vertices are different
		assert(t[0] != t[1]);
		assert(t[0] != t[2]);
		assert(t[0] != t[3]);
		assert(t[1] != t[2]);
		assert(t[1] != t[3]);
		assert(t[2] != t[3]);

		// check if there already exists a tetrahedron with the same indices
        assert(m_container->getTetrahedronIndex(t[0], t[1], t[2], t[3]) == sofa::InvalidID);
	}
    helper::WriteAccessor< Data< sofa::type::vector<Tetrahedron> > > m_tetrahedron = m_container->d_tetrahedron;
    const TetrahedronID tetrahedronIndex = (TetrahedronID)m_tetrahedron.size();

    // update nbr point if needed
    unsigned int nbrP = m_container->getNbPoints();
    for(unsigned int i=0; i<4; ++i)
        if (t[i] + 1 > nbrP) // point not well init
        {
            nbrP = t[i] + 1;
            m_container->setNbPoints(nbrP);
        }

    // update m_tetrahedraAroundVertex
    if (m_container->m_tetrahedraAroundVertex.size() < nbrP)
        m_container->m_tetrahedraAroundVertex.resize(nbrP);

    for (PointID j=0; j<4; ++j)
    {
        sofa::type::vector< TetrahedronID > &shell = m_container->m_tetrahedraAroundVertex[t[j]];
        shell.push_back( tetrahedronIndex );
    }

    // update triangle-tetrahedron cross buffers
    if (m_container->m_trianglesInTetrahedron.size() < tetrahedronIndex+1)
        m_container->m_trianglesInTetrahedron.resize(tetrahedronIndex+1);

    for (PointID j=0; j<4; ++j)
    {
        Triangle e1 (t[sofa::core::topology::trianglesOrientationInTetrahedronArray[j][0]],t[sofa::core::topology::trianglesOrientationInTetrahedronArray[j][1]],t[sofa::core::topology::trianglesOrientationInTetrahedronArray[j][2]]);
        TriangleID triangleIndex = m_container->getTriangleIndex(e1[0], e1[1], e1[2]);

        if(triangleIndex == sofa::InvalidID)
        {
            // first create the traingle
            sofa::type::vector< Triangle > v;
            v.push_back(e1);
            addTrianglesProcess((const sofa::type::vector< Triangle > &) v);

            triangleIndex = m_container->getTriangleIndex(e1[0], e1[1], e1[2]);
            assert(triangleIndex != sofa::InvalidID);
            if (triangleIndex == sofa::InvalidID)
            {
                msg_error() << "Triangle creation: " << e1 << " failed in addTetrahedronProcess. Triangle will not be added in buffers.";
                continue;
            }

            sofa::type::vector< TriangleID > triangleIndexList;
            triangleIndexList.push_back(triangleIndex);
            addTrianglesWarning(sofa::Size(v.size()), v, triangleIndexList);
        }

        // update m_trianglesInTetrahedron
        m_container->m_trianglesInTetrahedron[tetrahedronIndex][j]= triangleIndex;

        // update m_tetrahedraAroundTriangle
        if (m_container->m_tetrahedraAroundTriangle.size() < m_container->getNbTriangles())
            m_container->m_tetrahedraAroundTriangle.resize(m_container->getNbTriangles());

        sofa::type::vector< TetrahedronID > &shell = m_container->m_tetrahedraAroundTriangle[triangleIndex];
        shell.push_back( tetrahedronIndex );
    }


    // update edge-tetrahedron cross buffers
    if (m_container->m_edgesInTetrahedron.size() < tetrahedronIndex+1)
        m_container->m_edgesInTetrahedron.resize(tetrahedronIndex+1);

    for (EdgeID j=0; j<6; ++j)
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

        EdgeID edgeIndex=m_container->getEdgeIndex(t[p0],t[p1]);
        // we must create the edge
        if (edgeIndex == sofa::InvalidID)
        {
            sofa::type::vector< Edge > v;
            Edge e1(t[p0],t[p1]);
            v.push_back(e1);

            addEdgesProcess((const sofa::type::vector< Edge > &) v);

            edgeIndex=m_container->getEdgeIndex(t[p0],t[p1]);
            assert(edgeIndex != sofa::InvalidID);
            if (edgeIndex == sofa::InvalidID)
            {
                msg_error() << "Edge creation: " << e1 << " failed in addTetrahedronProcess. Edge will not be added in buffers.";
                continue;
            }

            sofa::type::vector< EdgeID > edgeIndexList;
            edgeIndexList.push_back(edgeIndex);
            addEdgesWarning(sofa::Size(v.size()), v, edgeIndexList);
        }

        // udpate m_edgesInTetrahedron
        m_container->m_edgesInTetrahedron[tetrahedronIndex][j]= edgeIndex;

        // update m_tetrahedraAroundEdge
        if (m_container->m_tetrahedraAroundEdge.size() < m_container->getNbEdges())
            m_container->m_tetrahedraAroundEdge.resize(m_container->getNbEdges());

        sofa::type::vector< TetrahedronID > &shell = m_container->m_tetrahedraAroundEdge[edgeIndex];
        shell.push_back( tetrahedronIndex );
    }

    m_tetrahedron.push_back(t);
}


void TetrahedronSetTopologyModifier::addTetrahedraProcess(const sofa::type::vector< Tetrahedron > &tetrahedra)
{
    for (size_t i = 0; i < tetrahedra.size(); ++i)
    {
        addTetrahedronProcess(tetrahedra[i]);
    }
}


void TetrahedronSetTopologyModifier::addTetrahedraWarning(const size_t nTetrahedra,
        const sofa::type::vector< Tetrahedron >& tetrahedraList,
        const sofa::type::vector< TetrahedronID >& tetrahedraIndexList)
{
    m_container->setTetrahedronTopologyToDirty();
    // Warning that tetrahedra just got created
    const TetrahedraAdded *e = new TetrahedraAdded(nTetrahedra, tetrahedraList, tetrahedraIndexList);
    addTopologyChange(e);
}


void TetrahedronSetTopologyModifier::addTetrahedraWarning(const size_t nTetrahedra,
        const sofa::type::vector< Tetrahedron >& tetrahedraList,
        const sofa::type::vector< TetrahedronID >& tetrahedraIndexList,
        const sofa::type::vector< sofa::type::vector< TetrahedronID > > & ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs)
{
    m_container->setTetrahedronTopologyToDirty();
    // Warning that tetrahedra just got created
    const TetrahedraAdded *e = new TetrahedraAdded(nTetrahedra, tetrahedraList, tetrahedraIndexList, ancestors, baryCoefs);
    addTopologyChange(e);
}


void TetrahedronSetTopologyModifier::removeTetrahedraWarning( sofa::type::vector<TetrahedronID> &tetrahedra )
{
    m_container->setTetrahedronTopologyToDirty();
    /// sort vertices to remove in a descendent order
    std::sort( tetrahedra.begin(), tetrahedra.end(), std::greater<TetrahedronID>() );

    // Warning that these edges will be deleted
    const TetrahedraRemoved *e=new TetrahedraRemoved(tetrahedra);
    addTopologyChange(e);
}

void TetrahedronSetTopologyModifier::removeTetrahedraProcess( const sofa::type::vector<TetrahedronID> &indices,
        const bool removeIsolatedItems)
{
    if(!m_container->hasTetrahedra())
        return;

    const bool removeIsolatedVertices = removeIsolatedItems && removeIsolated.getValue();
    const bool removeIsolatedEdges = removeIsolatedItems && m_container->hasEdges();
    const bool removeIsolatedTriangles = removeIsolatedItems && m_container->hasTriangles();

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

    sofa::type::vector<TriangleID> triangleToBeRemoved;
    sofa::type::vector<EdgeID> edgeToBeRemoved;
    sofa::type::vector<PointID> vertexToBeRemoved;

    helper::WriteAccessor< Data< sofa::type::vector<Tetrahedron> > > m_tetrahedron = m_container->d_tetrahedron;

    TetrahedronID lastTetrahedron = (TetrahedronID)m_container->getNumberOfTetrahedra() - 1;
    for (size_t i=0; i<indices.size(); ++i, --lastTetrahedron)
    {
        const Tetrahedron &t = m_tetrahedron[ indices[i] ];
        const Tetrahedron &h = m_tetrahedron[ lastTetrahedron ];

        if (m_container->hasTetrahedraAroundVertex())
        {
            for(PointID j=0; j<4; ++j)
            {
                sofa::type::vector< TetrahedronID > &shell = m_container->m_tetrahedraAroundVertex[ t[j] ];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if(removeIsolatedVertices && shell.empty())
                {
                    vertexToBeRemoved.push_back(t[j]);
                }
            }
        }

        if(m_container->hasTetrahedraAroundEdge())
        {
            for(EdgeID j=0; j<6; ++j)
            {
                sofa::type::vector< TetrahedronID > &shell = m_container->m_tetrahedraAroundEdge[ m_container->m_edgesInTetrahedron[indices[i]][j]];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if(removeIsolatedEdges && shell.empty())
                    edgeToBeRemoved.push_back(m_container->m_edgesInTetrahedron[indices[i]][j]);
            }
        }

        if(m_container->hasTetrahedraAroundTriangle())
        {
            for(TriangleID j=0; j<4; ++j)
            {
                sofa::type::vector< TetrahedronID > &shell = m_container->m_tetrahedraAroundTriangle[ m_container->m_trianglesInTetrahedron[indices[i]][j]];
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
                for(PointID j=0; j<4; ++j)
                {
                    sofa::type::vector< TetrahedronID > &shell = m_container->m_tetrahedraAroundVertex[ h[j] ];
                    replace(shell.begin(), shell.end(), lastTetrahedron, indices[i]);
                }
            }

            if (m_container->hasTetrahedraAroundEdge())
            {
                for(EdgeID j=0; j<6; ++j)
                {
                    sofa::type::vector< TetrahedronID > &shell =  m_container->m_tetrahedraAroundEdge[ m_container->m_edgesInTetrahedron[lastTetrahedron][j]];
                    replace(shell.begin(), shell.end(), lastTetrahedron, indices[i]);
                }
            }

            if (m_container->hasTetrahedraAroundTriangle())
            {
                for(TriangleID j=0; j<4; ++j)
                {
                    sofa::type::vector< TetrahedronID > &shell =  m_container->m_tetrahedraAroundTriangle[ m_container->m_trianglesInTetrahedron[lastTetrahedron][j]];
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
        removePointsProcess(vertexToBeRemoved, d_propagateToDOF.getValue());
    }
}

void TetrahedronSetTopologyModifier::addPointsProcess(const sofa::Size nPoints)
{
    // start by calling the parent's method.
    TriangleSetTopologyModifier::addPointsProcess( nPoints );

    if(m_container->hasTetrahedraAroundVertex())
        m_container->m_tetrahedraAroundVertex.resize( m_container->getNbPoints() );
}

void TetrahedronSetTopologyModifier::addEdgesProcess(const sofa::type::vector< Edge > &edges)
{
    // start by calling the parent's method.
    TriangleSetTopologyModifier::addEdgesProcess( edges );

    if(m_container->hasTetrahedraAroundEdge())
        m_container->m_tetrahedraAroundEdge.resize( m_container->getNumberOfEdges() );
}

void TetrahedronSetTopologyModifier::addTrianglesProcess(const sofa::type::vector< Triangle > &triangles)
{
    // start by calling the parent's method.
    TriangleSetTopologyModifier::addTrianglesProcess( triangles );
    if(m_container->hasTetrahedraAroundTriangle())
        m_container->m_tetrahedraAroundTriangle.resize( m_container->getNumberOfTriangles() );
}

void TetrahedronSetTopologyModifier::removePointsProcess(const sofa::type::vector<PointID> &indices,
        const bool removeDOF)
{
    if(m_container->hasTetrahedra())
    {
        if(!m_container->hasTetrahedraAroundVertex())
        {
            m_container->createTetrahedraAroundVertexArray();
        }

        helper::WriteAccessor< Data< sofa::type::vector<Tetrahedron> > > m_tetrahedron = m_container->d_tetrahedron;
        PointID lastPoint = (PointID)m_container->getNbPoints() - 1;
        for (size_t i=0; i<indices.size(); ++i, --lastPoint)
        {
            // updating the edges connected to the point replacing the removed one:
            // for all edges connected to the last point
            for (sofa::type::vector<TetrahedronID>::iterator itt=m_container->m_tetrahedraAroundVertex[lastPoint].begin();
                    itt!=m_container->m_tetrahedraAroundVertex[lastPoint].end(); ++itt)
            {
                const PointID vertexIndex = m_container->getVertexIndexInTetrahedron(m_tetrahedron[(*itt)],lastPoint);
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

void TetrahedronSetTopologyModifier::removeEdgesProcess( const sofa::type::vector<EdgeID> &indices,
        const bool removeIsolatedItems)
{
    if(!m_container->hasEdges()) // this method should only be called when edges exist
        return;

    if (m_container->hasEdgesInTetrahedron())
    {
        if(!m_container->hasTetrahedraAroundEdge())
            m_container->createTetrahedraAroundEdgeArray();

        EdgeID lastEdge = (EdgeID)m_container->getNumberOfEdges() - 1;
        for (size_t i=0; i<indices.size(); ++i, --lastEdge)
        {
            for (sofa::type::vector<TetrahedronID>::iterator itt=m_container->m_tetrahedraAroundEdge[lastEdge].begin();
                    itt!=m_container->m_tetrahedraAroundEdge[lastEdge].end(); ++itt)
            {
                const EdgeID edgeIndex=m_container->getEdgeIndexInTetrahedron(m_container->m_edgesInTetrahedron[(*itt)],lastEdge);
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

void TetrahedronSetTopologyModifier::removeTrianglesProcess( const sofa::type::vector<TriangleID> &indices,
        const bool removeIsolatedEdges,
        const bool removeIsolatedPoints)
{
    if(!m_container->hasTriangles()) // this method should only be called when triangles exist
        return;

    if (m_container->hasTrianglesInTetrahedron())
    {
        if(!m_container->hasTetrahedraAroundTriangle())
            m_container->createTetrahedraAroundTriangleArray();

        TriangleID lastTriangle = (TriangleID)m_container->m_tetrahedraAroundTriangle.size() - 1;
        for (size_t i = 0; i < indices.size(); ++i, --lastTriangle)
        {
            for (sofa::type::vector<TetrahedronID>::iterator itt=m_container->m_tetrahedraAroundTriangle[lastTriangle].begin();
                    itt!=m_container->m_tetrahedraAroundTriangle[lastTriangle].end(); ++itt)
            {
                const TriangleID triangleIndex=m_container->getTriangleIndexInTetrahedron(m_container->m_trianglesInTetrahedron[(*itt)],lastTriangle);
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

void TetrahedronSetTopologyModifier::renumberPointsProcess( const sofa::type::vector<PointID> &index,
        const sofa::type::vector<PointID> &inv_index,
        const bool renumberDOF)
{
    if(m_container->hasTetrahedra())
    {
        helper::WriteAccessor< Data< sofa::type::vector<Tetrahedron> > > m_tetrahedron = m_container->d_tetrahedron;
        if(m_container->hasTetrahedraAroundVertex())
        {
            sofa::type::vector< sofa::type::vector< TetrahedronID > > tetrahedraAroundVertex_cp = m_container->m_tetrahedraAroundVertex;
            for (size_t i = 0; i < index.size(); ++i)
            {
                m_container->m_tetrahedraAroundVertex[i] = tetrahedraAroundVertex_cp[ index[i] ];
            }
        }

        for (size_t i=0; i<m_tetrahedron.size(); ++i)
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

void TetrahedronSetTopologyModifier::removeTetrahedra(const sofa::type::vector<TetrahedronID> &tetrahedraIds, const bool removeIsolatedItems)
{
    sofa::type::vector<TetrahedronID> tetrahedraIds_filtered;
    for (size_t i = 0; i < tetrahedraIds.size(); i++)
    {
        if( tetrahedraIds[i] >= m_container->getNumberOfTetrahedra())
            dmsg_warning() << "Tetrahedra: " << tetrahedraIds[i] << " is out of bound and won't be removed.";
        else
            tetrahedraIds_filtered.push_back(tetrahedraIds[i]);
    }

    /// add the topological changes in the queue
    {
        SCOPED_TIMER("removeTetrahedraWarning");
        removeTetrahedraWarning(tetrahedraIds_filtered);
    }

    // inform other objects that the triangles are going to be removed
    {
        SCOPED_TIMER("propagateTopologicalChanges");
        propagateTopologicalChanges();
    }

    // now destroy the old tetrahedra.
    {
        SCOPED_TIMER("removeTetrahedraProcess");
        removeTetrahedraProcess(tetrahedraIds_filtered , removeIsolatedItems);
    }

    m_container->checkTopology();

    m_container->addRemovedTetraIndex(tetrahedraIds_filtered);
}

void TetrahedronSetTopologyModifier::removeItems(const sofa::type::vector< TetrahedronID >& items)
{
    removeTetrahedra(items);
}

void TetrahedronSetTopologyModifier::propagateTopologicalEngineChanges()
{
    if (m_container->beginChange() == m_container->endChange()) return; // nothing to do if no event is stored

    if (!m_container->isTetrahedronTopologyDirty()) // tetrahedron Data has not been touched
        return TriangleSetTopologyModifier::propagateTopologicalEngineChanges();

    auto& tetraTopologyHandlerList = m_container->getTopologyHandlerList(sofa::geometry::ElementType::TETRAHEDRON);
    for (const auto topoHandler : tetraTopologyHandlerList)
    {
        if (topoHandler->isDirty())
        {
            topoHandler->update();
        }
    }

    m_container->cleanTetrahedronTopologyFromDirty();
    TriangleSetTopologyModifier::propagateTopologicalEngineChanges();
}

} //namespace sofa::component::topology::container::dynamic
