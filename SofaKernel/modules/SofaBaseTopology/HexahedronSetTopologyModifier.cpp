/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <SofaBaseTopology/HexahedronSetTopologyModifier.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/TopologyChange.h>
#include <SofaBaseTopology/HexahedronSetTopologyContainer.h>
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
SOFA_DECL_CLASS(HexahedronSetTopologyModifier)
int HexahedronSetTopologyModifierClass = core::RegisterObject("Hexahedron set topology modifier")
        .add< HexahedronSetTopologyModifier >();

using namespace std;
using namespace sofa::defaulttype;
using namespace sofa::core::topology;

const unsigned int edgesInHexahedronArray[12][2]= {{0,1},{0,3},{0,4},{1,2},{1,5},{2,3},{2,6},{3,7},{4,5},{4,7},{5,6},{6,7}};


void HexahedronSetTopologyModifier::init()
{
    QuadSetTopologyModifier::init();
    this->getContext()->get(m_container);
}


void HexahedronSetTopologyModifier::addHexahedra(const sofa::helper::vector<Hexahedron> &hexahedra)
{
    unsigned int nhexa = m_container->getNbHexahedra();

    /// effectively add triangles in the topology container
    addHexahedraProcess(hexahedra);

    sofa::helper::vector<unsigned int> hexahedraIndex;
    hexahedraIndex.reserve(hexahedra.size());

    for (unsigned int i=0; i<hexahedra.size(); ++i)
        hexahedraIndex.push_back(nhexa+i);

    // add topology event in the stack of topological events
    addHexahedraWarning((unsigned int)hexahedra.size(), hexahedra, hexahedraIndex);

    // inform other objects that the edges are already added
    propagateTopologicalChanges();
}


void HexahedronSetTopologyModifier::addHexahedra(const sofa::helper::vector<Hexahedron> &hexahedra,
        const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors,
        const sofa::helper::vector<sofa::helper::vector<double> > &baryCoefs)
{
    unsigned int nhexa = m_container->getNbHexahedra();

    /// effectively add triangles in the topology container
    addHexahedraProcess(hexahedra);

    sofa::helper::vector<unsigned int> hexahedraIndex;
    hexahedraIndex.reserve(hexahedra.size());

    for (unsigned int i=0; i<hexahedra.size(); ++i)
        hexahedraIndex.push_back(nhexa+i);

    // add topology event in the stack of topological events
    addHexahedraWarning((unsigned int)hexahedra.size(), hexahedra, hexahedraIndex, ancestors, baryCoefs);

    // inform other objects that the edges are already added
    propagateTopologicalChanges();
}



void HexahedronSetTopologyModifier::addHexahedronProcess(Hexahedron t)
{
#ifndef NDEBUG
    // check if the 8 vertices are different
    assert(t[0]!=t[1]); assert(t[0]!=t[2]); assert(t[0]!=t[3]); assert(t[0]!=t[4]); assert(t[0]!=t[5]); assert(t[0]!=t[6]); assert(t[0]!=t[7]);
    assert(t[1]!=t[2]); assert(t[1]!=t[3]); assert(t[1]!=t[4]); assert(t[1]!=t[5]); assert(t[1]!=t[6]); assert(t[1]!=t[7]);
    assert(t[2]!=t[3]); assert(t[2]!=t[4]); assert(t[2]!=t[5]); assert(t[2]!=t[6]); assert(t[2]!=t[7]);
    assert(t[3]!=t[4]); assert(t[3]!=t[5]); assert(t[3]!=t[6]); assert(t[3]!=t[7]);
    assert(t[4]!=t[5]); assert(t[4]!=t[6]); assert(t[4]!=t[7]);
    assert(t[5]!=t[6]); assert(t[5]!=t[7]);
    assert(t[6]!=t[7]);

    // check if there already exists a hexahedron with the same indices
    assert(m_container->getHexahedronIndex(t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7])== -1);
#endif
    const unsigned int hexahedronIndex = m_container->getNumberOfHexahedra();
    helper::WriteAccessor< Data< sofa::helper::vector<Hexahedron> > > m_hexahedron = m_container->d_hexahedron;

    if(m_container->hasQuadsInHexahedron())
    {
        int quadIndex;

        // Quad 0 :
        quadIndex=m_container->getQuadIndex(t[0],t[3],t[2],t[1]);
        //assert(quadIndex!= -1);
        if(quadIndex == -1)
        {
            // first create the quad
            sofa::helper::vector< Quad > v;
            Quad e1 (t[0],t[3],t[2],t[1]);
            v.push_back(e1);

            addQuadsProcess((const sofa::helper::vector< Quad > &) v);

            quadIndex=m_container->getQuadIndex(t[0],t[3],t[2],t[1]);
            sofa::helper::vector< unsigned int > quadIndexList;
            quadIndexList.push_back(quadIndex);
            addQuadsWarning((unsigned int)v.size(), v, quadIndexList);
        }
        m_container->m_quadsInHexahedron.resize(hexahedronIndex+1);
        m_container->m_quadsInHexahedron[hexahedronIndex][0]=quadIndex;

        // Quad 1 :
        quadIndex=m_container->getQuadIndex(t[4],t[5],t[6],t[7]);
        //assert(quadIndex!= -1);
        if(quadIndex == -1)
        {
            // first create the quad
            sofa::helper::vector< Quad > v;
            Quad e1 (t[4],t[5],t[6],t[7]);
            v.push_back(e1);

            addQuadsProcess((const sofa::helper::vector< Quad > &) v);

            quadIndex=m_container->getQuadIndex(t[4],t[5],t[6],t[7]);
            sofa::helper::vector< unsigned int > quadIndexList;
            quadIndexList.push_back(quadIndex);
            addQuadsWarning((unsigned int)v.size(), v, quadIndexList);
        }
        m_container->m_quadsInHexahedron.resize(hexahedronIndex+1);
        m_container->m_quadsInHexahedron[hexahedronIndex][1]=quadIndex;

        // Quad 2 :
        quadIndex=m_container->getQuadIndex(t[0],t[1],t[5],t[4]);
        //assert(quadIndex!= -1);
        if(quadIndex == -1)
        {
            // first create the quad
            sofa::helper::vector< Quad > v;
            Quad e1 (t[0],t[1],t[5],t[4]);
            v.push_back(e1);

            addQuadsProcess((const sofa::helper::vector< Quad > &) v);

            quadIndex=m_container->getQuadIndex(t[0],t[1],t[5],t[4]);
            sofa::helper::vector< unsigned int > quadIndexList;
            quadIndexList.push_back(quadIndex);
            addQuadsWarning((unsigned int)v.size(), v, quadIndexList);
        }
        m_container->m_quadsInHexahedron.resize(hexahedronIndex+1);
        m_container->m_quadsInHexahedron[hexahedronIndex][2]=quadIndex;

        // Quad 3 :
        quadIndex=m_container->getQuadIndex(t[1],t[2],t[6],t[5]);
        //assert(quadIndex!= -1);
        if(quadIndex == -1)
        {
            // first create the quad
            sofa::helper::vector< Quad > v;
            Quad e1 (t[1],t[2],t[6],t[5]);
            v.push_back(e1);

            addQuadsProcess((const sofa::helper::vector< Quad > &) v);

            quadIndex=m_container->getQuadIndex(t[1],t[2],t[6],t[5]);
            sofa::helper::vector< unsigned int > quadIndexList;
            quadIndexList.push_back(quadIndex);
            addQuadsWarning((unsigned int)v.size(), v, quadIndexList);
        }
        m_container->m_quadsInHexahedron.resize(hexahedronIndex+1);
        m_container->m_quadsInHexahedron[hexahedronIndex][3]=quadIndex;

        // Quad 4 :
        quadIndex=m_container->getQuadIndex(t[2],t[3],t[7],t[6]);
        //assert(quadIndex!= -1);
        if(quadIndex == -1)
        {
            // first create the quad
            sofa::helper::vector< Quad > v;
            Quad e1 (t[2],t[3],t[7],t[6]);
            v.push_back(e1);

            addQuadsProcess((const sofa::helper::vector< Quad > &) v);

            quadIndex=m_container->getQuadIndex(t[2],t[3],t[7],t[6]);
            sofa::helper::vector< unsigned int > quadIndexList;
            quadIndexList.push_back(quadIndex);
            addQuadsWarning((unsigned int)v.size(), v, quadIndexList);
        }
        m_container->m_quadsInHexahedron.resize(hexahedronIndex+1);
        m_container->m_quadsInHexahedron[hexahedronIndex][4]=quadIndex;

        // Quad 5 :
        quadIndex=m_container->getQuadIndex(t[3],t[0],t[4],t[7]);
        //assert(quadIndex!= -1);
        if(quadIndex == -1)
        {
            // first create the quad
            sofa::helper::vector< Quad > v;
            Quad e1 (t[3],t[0],t[4],t[7]);
            v.push_back(e1);

            addQuadsProcess((const sofa::helper::vector< Quad > &) v);

            quadIndex=m_container->getQuadIndex(t[3],t[0],t[4],t[7]);
            sofa::helper::vector< unsigned int > quadIndexList;
            quadIndexList.push_back(quadIndex);
            addQuadsWarning((unsigned int)v.size(), v, quadIndexList);
        }
        m_container->m_quadsInHexahedron.resize(hexahedronIndex+1);
        m_container->m_quadsInHexahedron[hexahedronIndex][5]=quadIndex;

        if(m_container->hasHexahedraAroundQuad())
        {
            for(unsigned int q=0; q<6; ++q)
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_hexahedraAroundQuad[m_container->m_quadsInHexahedron[hexahedronIndex][q]];
                shell.push_back( hexahedronIndex );
            }
        }
    } // quads

    if(m_container->hasEdgesInHexahedron())
    {
        m_container->m_edgesInHexahedron.resize(hexahedronIndex+1);
        for(unsigned int edgeIdx=0; edgeIdx<12; ++edgeIdx)
        {
            unsigned p0 = edgesInHexahedronArray[edgeIdx][0];
            unsigned p1 = edgesInHexahedronArray[edgeIdx][1];

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

            m_container->m_edgesInHexahedron[hexahedronIndex][edgeIdx]= edgeIndex;
        }

        if(m_container->hasHexahedraAroundEdge())
        {
            for(unsigned int e=0; e<12; ++e)
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_hexahedraAroundEdge[m_container->m_edgesInHexahedron[hexahedronIndex][e]];
                shell.push_back( hexahedronIndex );
            }
        }
    } // edges

    if(m_container->hasHexahedraAroundVertex())
    {
        for(unsigned int v=0; v<8; ++v)
        {
            sofa::helper::vector< unsigned int > &shell = m_container->getHexahedraAroundVertexForModification( t[v] );
            shell.push_back( hexahedronIndex );
        }
    }

    m_hexahedron.push_back(t);
}


void HexahedronSetTopologyModifier::addHexahedraProcess(const sofa::helper::vector< Hexahedron > &hexahedra)
{
    for(unsigned int i = 0; i < hexahedra.size(); ++i)
    {
        addHexahedronProcess(hexahedra[i]);
    }
}


void HexahedronSetTopologyModifier::addHexahedraWarning(const unsigned int nHexahedra,
        const sofa::helper::vector< Hexahedron >& hexahedraList,
        const sofa::helper::vector< unsigned int >& hexahedraIndexList)
{
    m_container->setHexahedronTopologyToDirty();
    // Warning that hexahedra just got created
    HexahedraAdded *e = new HexahedraAdded(nHexahedra, hexahedraList, hexahedraIndexList);
    addTopologyChange(e);
}


void HexahedronSetTopologyModifier::addHexahedraWarning(const unsigned int nHexahedra,
        const sofa::helper::vector< Hexahedron >& hexahedraList,
        const sofa::helper::vector< unsigned int >& hexahedraIndexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    m_container->setHexahedronTopologyToDirty();
    // Warning that hexahedra just got created
    HexahedraAdded *e = new HexahedraAdded(nHexahedra, hexahedraList, hexahedraIndexList, ancestors, baryCoefs);
    addTopologyChange(e);
}


void HexahedronSetTopologyModifier::removeHexahedraWarning( sofa::helper::vector<unsigned int> &hexahedra )
{
    m_container->setHexahedronTopologyToDirty();
    /// sort vertices to remove in a descendent order
    std::sort( hexahedra.begin(), hexahedra.end(), std::greater<unsigned int>() );

    // Warning that these edges will be deleted
    HexahedraRemoved *e = new HexahedraRemoved(hexahedra);
    addTopologyChange(e);
}


void HexahedronSetTopologyModifier::removeHexahedraProcess( const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedItems)
{
    if(!m_container->hasHexahedra())
        return;

    bool removeIsolatedVertices = removeIsolatedItems && removeIsolated.getValue();
    bool removeIsolatedEdges = removeIsolatedItems && m_container->hasEdges();
    bool removeIsolatedQuads = removeIsolatedItems && m_container->hasQuads();

    if(removeIsolatedVertices)
    {
        if(!m_container->hasHexahedraAroundVertex())
            m_container->createHexahedraAroundVertexArray();
    }

    if(removeIsolatedEdges)
    {
        if(!m_container->hasHexahedraAroundEdge())
            m_container->createHexahedraAroundEdgeArray();
    }

    if(removeIsolatedQuads)
    {
        if(!m_container->hasHexahedraAroundQuad())
            m_container->createHexahedraAroundQuadArray();
    }

    sofa::helper::vector<unsigned int> quadToBeRemoved;
    sofa::helper::vector<unsigned int> edgeToBeRemoved;
    sofa::helper::vector<unsigned int> vertexToBeRemoved;

    unsigned int lastHexahedron = m_container->getNumberOfHexahedra() - 1;

    helper::WriteAccessor< Data< sofa::helper::vector<Hexahedron> > > m_hexahedron = m_container->d_hexahedron;

    for(unsigned int i=0; i<indices.size(); ++i, --lastHexahedron)
    {
        const Hexahedron &t = m_hexahedron[ indices[i] ];
        const Hexahedron &h = m_hexahedron[ lastHexahedron ];

        if(m_container->hasHexahedraAroundVertex())
        {
            for(unsigned int v=0; v<8; ++v)
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_hexahedraAroundVertex[ t[v] ];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if(removeIsolatedVertices && shell.empty())
                    vertexToBeRemoved.push_back(t[v]);
            }
        }

        if(m_container->hasHexahedraAroundEdge())
        {
            for(unsigned int e=0; e<12; ++e)
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_hexahedraAroundEdge[ m_container->m_edgesInHexahedron[indices[i]][e]];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if(removeIsolatedEdges && shell.empty())
                    edgeToBeRemoved.push_back(m_container->m_edgesInHexahedron[indices[i]][e]);
            }
        }

        if(m_container->hasHexahedraAroundQuad())
        {
            for(unsigned int q=0; q<6; ++q)
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_hexahedraAroundQuad[ m_container->m_quadsInHexahedron[indices[i]][q]];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if(removeIsolatedQuads && shell.empty())
                    quadToBeRemoved.push_back(m_container->m_quadsInHexahedron[indices[i]][q]);
            }
        }

        // now updates the shell information of the edge formely at the end of the array
        if( indices[i] < lastHexahedron )
        {
            if(m_container->hasHexahedraAroundVertex())
            {
                for(unsigned int v=0; v<8; ++v)
                {
                    sofa::helper::vector< unsigned int > &shell = m_container->m_hexahedraAroundVertex[ h[v] ];
                    replace(shell.begin(), shell.end(), lastHexahedron, indices[i]);
                }
            }

            if(m_container->hasHexahedraAroundEdge())
            {
                for(unsigned int e=0; e<12; ++e)
                {
                    sofa::helper::vector< unsigned int > &shell =  m_container->m_hexahedraAroundEdge[ m_container->m_edgesInHexahedron[lastHexahedron][e]];
                    replace(shell.begin(), shell.end(), lastHexahedron, indices[i]);
                }
            }

            if(m_container->hasHexahedraAroundQuad())
            {
                for(unsigned int q=0; q<6; ++q)
                {
                    sofa::helper::vector< unsigned int > &shell =  m_container->m_hexahedraAroundQuad[ m_container->m_quadsInHexahedron[lastHexahedron][q]];
                    replace(shell.begin(), shell.end(), lastHexahedron, indices[i]);
                }
            }
        }

        if(m_container->hasQuadsInHexahedron())
        {
            // removes the quadsInHexahedrons from the quadsInHexahedronArray
            m_container->m_quadsInHexahedron[ indices[i] ] = m_container->m_quadsInHexahedron[ lastHexahedron ]; // overwriting with last valid value.
            m_container->m_quadsInHexahedron.resize( lastHexahedron ); // resizing to erase multiple occurence of the hexa.
        }

        if(m_container->hasEdgesInHexahedron())
        {
            // removes the edgesInHexahedrons from the edgesInHexahedronArray
            m_container->m_edgesInHexahedron[ indices[i] ] = m_container->m_edgesInHexahedron[ lastHexahedron ]; // overwriting with last valid value.
            m_container->m_edgesInHexahedron.resize( lastHexahedron ); // resizing to erase multiple occurence of the hexa.
        }

        // removes the hexahedron from the hexahedronArray
        m_hexahedron[ indices[i] ] = m_hexahedron[ lastHexahedron ]; // overwriting with last valid value.
        m_hexahedron.resize( lastHexahedron ); // resizing to erase multiple occurence of the hexa.
    }

    if( (!quadToBeRemoved.empty()) || (!edgeToBeRemoved.empty()))
    {
        if(!quadToBeRemoved.empty())
        {
            /// warn that quads will be deleted
            removeQuadsWarning(quadToBeRemoved);
        }

        if(!edgeToBeRemoved.empty())
        {
            /// warn that edges will be deleted
            removeEdgesWarning(edgeToBeRemoved);
        }

        propagateTopologicalChanges();

        if(!quadToBeRemoved.empty())
        {
            /// actually remove quads without looking for isolated vertices
            removeQuadsProcess(quadToBeRemoved, false, false);
        }

        if(!edgeToBeRemoved.empty())
        {
            /// actually remove edges without looking for isolated vertices
            removeEdgesProcess(edgeToBeRemoved, false);
        }
    }

    if(!vertexToBeRemoved.empty())
    {
        removePointsWarning(vertexToBeRemoved);
        propagateTopologicalChanges();
        removePointsProcess(vertexToBeRemoved);
    }
}

void HexahedronSetTopologyModifier::addPointsProcess(const unsigned int nPoints)
{
    // start by calling the parent's method.
    QuadSetTopologyModifier::addPointsProcess( nPoints );

    if(m_container->hasHexahedraAroundVertex())
        m_container->m_hexahedraAroundVertex.resize( m_container->getNbPoints() );
}

void HexahedronSetTopologyModifier::addEdgesProcess(const sofa::helper::vector< Edge > &edges)
{
    // start by calling the parent's method.
    QuadSetTopologyModifier::addEdgesProcess( edges );

    if(m_container->hasHexahedraAroundEdge())
        m_container->m_hexahedraAroundEdge.resize( m_container->getNumberOfEdges() );
}

void HexahedronSetTopologyModifier::addQuadsProcess(const sofa::helper::vector< Quad > &quads)
{
    // start by calling the parent's method.
    QuadSetTopologyModifier::addQuadsProcess( quads );

    if(m_container->hasHexahedraAroundQuad())
        m_container->m_hexahedraAroundQuad.resize( m_container->getNumberOfQuads() );
}

void HexahedronSetTopologyModifier::removePointsProcess(const sofa::helper::vector<unsigned int> &indices,
        const bool removeDOF)
{
    if(m_container->hasHexahedra())
    {
        if(!m_container->hasHexahedraAroundVertex())
        {
            m_container->createHexahedraAroundVertexArray();
        }

        helper::WriteAccessor< Data< sofa::helper::vector<Hexahedron> > > m_hexahedron = m_container->d_hexahedron;

        unsigned int lastPoint = m_container->getNbPoints() - 1;
        for(unsigned int i = 0; i < indices.size(); ++i, --lastPoint)
        {
            // updating the edges connected to the point replacing the removed one:
            // for all edges connected to the last point
            for(sofa::helper::vector<unsigned int>::iterator itt=m_container->m_hexahedraAroundVertex[lastPoint].begin();
                itt!=m_container->m_hexahedraAroundVertex[lastPoint].end(); ++itt)
            {
                unsigned int vertexIndex = m_container->getVertexIndexInHexahedron(m_hexahedron[*itt], lastPoint);
                m_hexahedron[*itt][ vertexIndex] = indices[i];
            }

            // updating the edge shell itself (change the old index for the new one)
            m_container->m_hexahedraAroundVertex[ indices[i] ] = m_container->m_hexahedraAroundVertex[ lastPoint ];
        }

        m_container->m_hexahedraAroundVertex.resize( m_container->m_hexahedraAroundVertex.size() - indices.size() );
    }

    // Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    // call the parent's method.
    QuadSetTopologyModifier::removePointsProcess(  indices, removeDOF );
}

void HexahedronSetTopologyModifier::removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedItems)
{
    if(!m_container->hasEdges()) // this method should only be called when edges exist
        return;

    if(m_container->hasEdgesInHexahedron())
    {
        if(!m_container->hasHexahedraAroundEdge())
            m_container->createHexahedraAroundEdgeArray();

        unsigned int lastEdge = m_container->getNumberOfEdges() - 1;
        for(unsigned int i=0; i<indices.size(); ++i, --lastEdge)
        {
            for(sofa::helper::vector<unsigned int>::iterator itt=m_container->m_hexahedraAroundEdge[lastEdge].begin();
                itt!=m_container->m_hexahedraAroundEdge[lastEdge].end(); ++itt)
            {
                unsigned int edgeIndex = m_container->getEdgeIndexInHexahedron(m_container->m_edgesInHexahedron[*itt], lastEdge);
                m_container->m_edgesInHexahedron[*itt][edgeIndex] = indices[i];
            }

            // updating the edge shell itself (change the old index for the new one)
            m_container->m_hexahedraAroundEdge[ indices[i] ] = m_container->m_hexahedraAroundEdge[ lastEdge ];
        }

        m_container->m_hexahedraAroundEdge.resize( m_container->m_hexahedraAroundEdge.size() - indices.size() );
    }

    // call the parent's method.
    QuadSetTopologyModifier::removeEdgesProcess(  indices, removeIsolatedItems );
}

void HexahedronSetTopologyModifier::removeQuadsProcess( const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedEdges,
        const bool removeIsolatedPoints)
{
    if(!m_container->hasQuads()) // this method should only be called when quads exist
        return;

    if(m_container->hasQuadsInHexahedron())
    {
        if(!m_container->hasHexahedraAroundQuad())
            m_container->createHexahedraAroundQuadArray();

        unsigned int lastQuad = m_container->getNumberOfQuads() - 1;
        for(unsigned int i=0; i<indices.size(); ++i, --lastQuad)
        {
            for(sofa::helper::vector<unsigned int>::iterator itt=m_container->m_hexahedraAroundQuad[lastQuad].begin();
                itt!=m_container->m_hexahedraAroundQuad[lastQuad].end(); ++itt)
            {
                unsigned int quadIndex=m_container->getQuadIndexInHexahedron(m_container->m_quadsInHexahedron[*itt],lastQuad);
                m_container->m_quadsInHexahedron[*itt][quadIndex]=indices[i];
            }

            // updating the quad shell itself (change the old index for the new one)
            m_container->m_hexahedraAroundQuad[ indices[i] ] = m_container->m_hexahedraAroundQuad[ lastQuad ];
        }
        m_container->m_hexahedraAroundQuad.resize( m_container->m_hexahedraAroundQuad.size() - indices.size() );
    }

    // call the parent's method.
    QuadSetTopologyModifier::removeQuadsProcess( indices, removeIsolatedEdges, removeIsolatedPoints );
}

void HexahedronSetTopologyModifier::renumberPointsProcess( const sofa::helper::vector<unsigned int> &index, const sofa::helper::vector<unsigned int> &inv_index, const bool renumberDOF)
{
    if(m_container->hasHexahedra())
    {
        if(m_container->hasHexahedraAroundVertex())
        {
            sofa::helper::vector< sofa::helper::vector< unsigned int > > hexahedraAroundVertex_cp = m_container->m_hexahedraAroundVertex;
            for(unsigned int i=0; i<index.size(); ++i)
            {
                m_container->m_hexahedraAroundVertex[i] = hexahedraAroundVertex_cp[ index[i] ];
            }
        }

        helper::WriteAccessor< Data< sofa::helper::vector<Hexahedron> > > m_hexahedron = m_container->d_hexahedron;

        for(unsigned int i=0; i<m_hexahedron.size(); ++i)
        {
            m_hexahedron[i][0]  = inv_index[ m_hexahedron[i][0]  ];
            m_hexahedron[i][1]  = inv_index[ m_hexahedron[i][1]  ];
            m_hexahedron[i][2]  = inv_index[ m_hexahedron[i][2]  ];
            m_hexahedron[i][3]  = inv_index[ m_hexahedron[i][3]  ];
            m_hexahedron[i][4]  = inv_index[ m_hexahedron[i][4]  ];
            m_hexahedron[i][5]  = inv_index[ m_hexahedron[i][5]  ];
            m_hexahedron[i][6]  = inv_index[ m_hexahedron[i][6]  ];
            m_hexahedron[i][7]  = inv_index[ m_hexahedron[i][7]  ];
        }
    }

    // call the parent's method.
    QuadSetTopologyModifier::renumberPointsProcess( index, inv_index, renumberDOF );
}

void HexahedronSetTopologyModifier::removeHexahedra(const sofa::helper::vector< unsigned int >& hexahedraIds)
{
    sofa::helper::vector<unsigned int> hexahedraIds_filtered;
    for (unsigned int i = 0; i < hexahedraIds.size(); i++)
    {
        if( hexahedraIds[i] >= m_container->getNumberOfHexahedra())
            msg_error() << "Unable to remove the hexahedra: "<< hexahedraIds[i] <<" its index is out of bound." ;
        else
            hexahedraIds_filtered.push_back(hexahedraIds[i]);
    }

    // add the topological changes in the queue
    removeHexahedraWarning(hexahedraIds_filtered);
    // inform other objects that the hexa are going to be removed
    propagateTopologicalChanges();
    // now destroy the old hexahedra.
    removeHexahedraProcess(hexahedraIds_filtered ,true);

    m_container->checkTopology();
}

void HexahedronSetTopologyModifier::removeItems(const sofa::helper::vector< unsigned int >& items)
{
    removeHexahedra(items);
}

void HexahedronSetTopologyModifier::renumberPoints(const sofa::helper::vector<unsigned int> &index,
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

void HexahedronSetTopologyModifier::propagateTopologicalEngineChanges()
{
    if (m_container->beginChange() == m_container->endChange()) return; // nothing to do if no event is stored

    if (!m_container->isHexahedronTopologyDirty()) // hexahedron Data has not been touched
        return QuadSetTopologyModifier::propagateTopologicalEngineChanges();

    std::list<sofa::core::topology::TopologyEngine *>::iterator it;

    for ( it = m_container->m_enginesList.begin(); it!=m_container->m_enginesList.end(); ++it)
    {
        sofa::core::topology::TopologyEngine* topoEngine = (*it);
        if (topoEngine->isDirty())
        {
#ifndef NDEBUG
            msg_info() << "HexahedronSetTopologyModifier::performing: " << topoEngine->getName() ;
#endif
            topoEngine->update();
        }
    }

    m_container->cleanHexahedronTopologyFromDirty();
    QuadSetTopologyModifier::propagateTopologicalEngineChanges();
}


} // namespace topology

} // namespace component

} // namespace sofa

