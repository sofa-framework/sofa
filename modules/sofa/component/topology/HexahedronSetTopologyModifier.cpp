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
#include <sofa/component/topology/HexahedronSetTopologyModifier.h>
#include <sofa/component/topology/HexahedronSetTopologyChange.h>
#include <sofa/component/topology/HexahedronSetTopologyContainer.h>
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
using namespace sofa::core::componentmodel::behavior;

const unsigned int hexahedronEdgeArray[12][2]= {{0,1},{0,3},{0,4},{1,2},{1,5},{2,3},{2,6},{3,7},{4,5},{4,7},{5,6},{6,7}};


HexahedronSetTopologyContainer* HexahedronSetTopologyModifier::getHexahedronSetTopologyContainer() const
{
    return static_cast<HexahedronSetTopologyContainer* > (this->m_topologyContainer);
}


void HexahedronSetTopologyModifier::addHexahedron(Hexahedron t)
{
    HexahedronSetTopologyContainer * container = getHexahedronSetTopologyContainer();

    // check if the 8 vertices are different
    assert(t[0]!=t[1]); assert(t[0]!=t[2]); assert(t[0]!=t[3]); assert(t[0]!=t[4]); assert(t[0]!=t[5]); assert(t[0]!=t[6]); assert(t[0]!=t[7]);
    assert(t[1]!=t[2]); assert(t[1]!=t[3]); assert(t[1]!=t[4]); assert(t[1]!=t[5]); assert(t[1]!=t[6]); assert(t[1]!=t[7]);
    assert(t[2]!=t[3]); assert(t[2]!=t[4]); assert(t[2]!=t[5]); assert(t[2]!=t[6]); assert(t[2]!=t[7]);
    assert(t[3]!=t[4]); assert(t[3]!=t[5]); assert(t[3]!=t[6]); assert(t[3]!=t[7]);
    assert(t[4]!=t[5]); assert(t[4]!=t[6]); assert(t[4]!=t[7]);
    assert(t[5]!=t[6]); assert(t[5]!=t[7]);
    assert(t[6]!=t[7]);

    // check if there already exists a hexahedron with the same indices
    assert(container->getHexahedronIndex(t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7])== -1);

    const unsigned int hexahedronIndex = container->m_hexahedron.size();

    if(container->hasQuads())
    {
        if(container->hasHexahedronQuads())
        {
            int quadIndex;

            // Quad 0 :
            quadIndex=container->getQuadIndex(t[0],t[3],t[2],t[1]);
            //assert(quadIndex!= -1);
            if(quadIndex == -1)
            {
                // first create the quad
                sofa::helper::vector< Quad > v;
                Quad e1 (t[0],t[3],t[2],t[1]);
                v.push_back(e1);

                addQuadsProcess((const sofa::helper::vector< Quad > &) v);

                quadIndex=container->getQuadIndex(t[0],t[3],t[2],t[1]);
                sofa::helper::vector< unsigned int > quadIndexList;
                quadIndexList.push_back(quadIndex);
                this->addQuadsWarning( v.size(), v,quadIndexList);
            }
            container->m_hexahedronQuad.resize(hexahedronIndex+1);
            container->m_hexahedronQuad[hexahedronIndex][0]=quadIndex;

            // Quad 1 :
            quadIndex=container->getQuadIndex(t[4],t[5],t[6],t[7]);
            //assert(quadIndex!= -1);
            if(quadIndex == -1)
            {
                // first create the quad
                sofa::helper::vector< Quad > v;
                Quad e1 (t[4],t[5],t[6],t[7]);
                v.push_back(e1);

                addQuadsProcess((const sofa::helper::vector< Quad > &) v);

                quadIndex=container->getQuadIndex(t[4],t[5],t[6],t[7]);
                sofa::helper::vector< unsigned int > quadIndexList;
                quadIndexList.push_back(quadIndex);
                this->addQuadsWarning( v.size(), v,quadIndexList);
            }
            container->m_hexahedronQuad.resize(hexahedronIndex+1);
            container->m_hexahedronQuad[hexahedronIndex][1]=quadIndex;

            // Quad 2 :
            quadIndex=container->getQuadIndex(t[0],t[1],t[5],t[4]);
            //assert(quadIndex!= -1);
            if(quadIndex == -1)
            {
                // first create the quad
                sofa::helper::vector< Quad > v;
                Quad e1 (t[0],t[1],t[5],t[4]);
                v.push_back(e1);

                addQuadsProcess((const sofa::helper::vector< Quad > &) v);

                quadIndex=container->getQuadIndex(t[0],t[1],t[5],t[4]);
                sofa::helper::vector< unsigned int > quadIndexList;
                quadIndexList.push_back(quadIndex);
                this->addQuadsWarning( v.size(), v,quadIndexList);
            }
            container->m_hexahedronQuad.resize(hexahedronIndex+1);
            container->m_hexahedronQuad[hexahedronIndex][2]=quadIndex;

            // Quad 3 :
            quadIndex=container->getQuadIndex(t[1],t[2],t[6],t[5]);
            //assert(quadIndex!= -1);
            if(quadIndex == -1)
            {
                // first create the quad
                sofa::helper::vector< Quad > v;
                Quad e1 (t[1],t[2],t[6],t[5]);
                v.push_back(e1);

                addQuadsProcess((const sofa::helper::vector< Quad > &) v);

                quadIndex=container->getQuadIndex(t[1],t[2],t[6],t[5]);
                sofa::helper::vector< unsigned int > quadIndexList;
                quadIndexList.push_back(quadIndex);
                this->addQuadsWarning( v.size(), v,quadIndexList);
            }
            container->m_hexahedronQuad.resize(hexahedronIndex+1);
            container->m_hexahedronQuad[hexahedronIndex][3]=quadIndex;

            // Quad 4 :
            quadIndex=container->getQuadIndex(t[2],t[3],t[7],t[6]);
            //assert(quadIndex!= -1);
            if(quadIndex == -1)
            {
                // first create the quad
                sofa::helper::vector< Quad > v;
                Quad e1 (t[2],t[3],t[7],t[6]);
                v.push_back(e1);

                addQuadsProcess((const sofa::helper::vector< Quad > &) v);

                quadIndex=container->getQuadIndex(t[2],t[3],t[7],t[6]);
                sofa::helper::vector< unsigned int > quadIndexList;
                quadIndexList.push_back(quadIndex);
                this->addQuadsWarning( v.size(), v,quadIndexList);
            }
            container->m_hexahedronQuad.resize(hexahedronIndex+1);
            container->m_hexahedronQuad[hexahedronIndex][4]=quadIndex;

            // Quad 5 :
            quadIndex=container->getQuadIndex(t[3],t[0],t[4],t[7]);
            //assert(quadIndex!= -1);
            if(quadIndex == -1)
            {
                // first create the quad
                sofa::helper::vector< Quad > v;
                Quad e1 (t[3],t[0],t[4],t[7]);
                v.push_back(e1);

                addQuadsProcess((const sofa::helper::vector< Quad > &) v);

                quadIndex=container->getQuadIndex(t[3],t[0],t[4],t[7]);
                sofa::helper::vector< unsigned int > quadIndexList;
                quadIndexList.push_back(quadIndex);
                this->addQuadsWarning( v.size(), v,quadIndexList);
            }
            container->m_hexahedronQuad.resize(hexahedronIndex+1);
            container->m_hexahedronQuad[hexahedronIndex][5]=quadIndex;
        } // hexahedronQuads

        if(container->hasHexahedronQuadShell())
        {
            for(unsigned int q=0; q<6; ++q)
            {
                sofa::helper::vector< unsigned int > &shell = container->m_hexahedronQuadShell[container->m_hexahedronQuad[hexahedronIndex][q]];
                shell.push_back( hexahedronIndex );
            }
        }
    } // quads

    if(container->hasEdges())
    {
        if(container->hasHexahedronEdges())
        {
            container->m_hexahedronEdge.resize(hexahedronIndex+1);
            for(unsigned int edgeIdx=0; edgeIdx<12; ++edgeIdx)
            {
                const int edgeIndex=container->getEdgeIndex(t[hexahedronEdgeArray[edgeIdx][0]],
                        t[hexahedronEdgeArray[edgeIdx][1]]);
                assert(edgeIndex!= -1);
                container->m_hexahedronEdge[hexahedronIndex][edgeIdx]= edgeIndex;
            }
        }

        if(container->hasHexahedronEdgeShell())
        {
            for(unsigned int e=0; e<12; ++e)
            {
                sofa::helper::vector< unsigned int > &shell = container->m_hexahedronEdgeShell[container->m_hexahedronEdge[hexahedronIndex][e]];
                shell.push_back( hexahedronIndex );
            }
        }
    } // edges

    if(container->hasHexahedronVertexShell())
    {
        for(unsigned int v=0; v<8; ++v)
        {
            sofa::helper::vector< unsigned int > &shell = container->getHexahedronVertexShellForModification( t[v] );
            shell.push_back( hexahedronIndex );
        }
    }

    container->m_hexahedron.push_back(t);
}


void HexahedronSetTopologyModifier::addHexahedraProcess(const sofa::helper::vector< Hexahedron > &hexahedra)
{
    for(unsigned int i = 0; i < hexahedra.size(); ++i)
    {
        addHexahedron(hexahedra[i]);
    }
}


void HexahedronSetTopologyModifier::addHexahedraWarning(const unsigned int nHexahedra,
        const sofa::helper::vector< Hexahedron >& hexahedraList,
        const sofa::helper::vector< unsigned int >& hexahedraIndexList)
{
    // Warning that hexahedra just got created
    HexahedraAdded *e = new HexahedraAdded(nHexahedra, hexahedraList, hexahedraIndexList);
    this->addTopologyChange(e);
}


void HexahedronSetTopologyModifier::addHexahedraWarning(const unsigned int nHexahedra,
        const sofa::helper::vector< Hexahedron >& hexahedraList,
        const sofa::helper::vector< unsigned int >& hexahedraIndexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    // Warning that hexahedra just got created
    HexahedraAdded *e = new HexahedraAdded(nHexahedra, hexahedraList, hexahedraIndexList, ancestors, baryCoefs);
    this->addTopologyChange(e);
}


void HexahedronSetTopologyModifier::removeHexahedraWarning( sofa::helper::vector<unsigned int> &hexahedra )
{
    /// sort vertices to remove in a descendent order
    std::sort( hexahedra.begin(), hexahedra.end(), std::greater<unsigned int>() );

    // Warning that these edges will be deleted
    HexahedraRemoved *e = new HexahedraRemoved(hexahedra);
    this->addTopologyChange(e);
}


void HexahedronSetTopologyModifier::removeHexahedraProcess( const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedItems)
{
    HexahedronSetTopologyContainer * container = getHexahedronSetTopologyContainer();

    bool removeIsolatedVertices = removeIsolatedItems;
    bool removeIsolatedEdges = removeIsolatedItems && container->hasEdges();
    bool removeIsolatedQuads = removeIsolatedItems && container->hasQuads();

    if(removeIsolatedVertices)
    {
        if(!container->hasHexahedronVertexShell())
            container->createHexahedronVertexShellArray();
    }

    if(removeIsolatedEdges)
    {
        if(!container->hasHexahedronEdgeShell())
            container->createHexahedronEdgeShellArray();
    }

    if(removeIsolatedQuads)
    {
        if(!container->hasHexahedronQuadShell())
            container->createHexahedronQuadShellArray();
    }

    sofa::helper::vector<unsigned int> quadToBeRemoved;
    sofa::helper::vector<unsigned int> edgeToBeRemoved;
    sofa::helper::vector<unsigned int> vertexToBeRemoved;

    for(unsigned int i=0; i<indices.size(); ++i)
    {
        const unsigned int lastHexahedron = container->m_hexahedron.size() - 1;
        Hexahedron &t = container->m_hexahedron[ indices[i] ];
        Hexahedron &h = container->m_hexahedron[ lastHexahedron ];

        if(container->hasHexahedronVertexShell())
        {
            for(unsigned int v=0; v<8; ++v)
            {
                sofa::helper::vector< unsigned int > &shell = container->m_hexahedronVertexShell[ t[v] ];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if((removeIsolatedVertices) && (shell.empty()))
                    vertexToBeRemoved.push_back(t[v]);
            }
        }

        if(container->hasHexahedronEdgeShell())
        {
            for(unsigned int e=0; e<12; ++e)
            {
                sofa::helper::vector< unsigned int > &shell = container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[indices[i]][e]];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if((removeIsolatedItems) && (shell.empty()))
                    edgeToBeRemoved.push_back(container->m_hexahedronEdge[indices[i]][e]);
            }
        }

        if(container->hasHexahedronQuadShell())
        {
            for(unsigned int q=0; q<6; ++q)
            {
                sofa::helper::vector< unsigned int > &shell = container->m_hexahedronQuadShell[ container->m_hexahedronQuad[indices[i]][q]];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if((removeIsolatedItems) && (shell.empty()))
                    quadToBeRemoved.push_back(container->m_hexahedronQuad[indices[i]][q]);
            }
        }

        // now updates the shell information of the edge formely at the end of the array
        if( indices[i] < lastHexahedron )
        {
            if(container->hasHexahedronVertexShell())
            {
                for(unsigned int v=0; v<8; ++v)
                {
                    sofa::helper::vector< unsigned int > &shell = container->m_hexahedronVertexShell[ h[v] ];
                    replace(shell.begin(), shell.end(), lastHexahedron, indices[i]);
                }
            }

            if(container->hasHexahedronEdgeShell())
            {
                for(unsigned int e=0; e<12; ++e)
                {
                    sofa::helper::vector< unsigned int > &shell =  container->m_hexahedronEdgeShell[ container->m_hexahedronEdge[lastHexahedron][e]];
                    replace(shell.begin(), shell.end(), lastHexahedron, indices[i]);
                }
            }

            if(!container->m_hexahedronQuadShell.empty())
            {
                for(unsigned int q=0; q<6; ++q)
                {
                    sofa::helper::vector< unsigned int > &shell =  container->m_hexahedronQuadShell[ container->m_hexahedronQuad[lastHexahedron][q]];
                    replace(shell.begin(), shell.end(), lastHexahedron, indices[i]);
                }
            }
        }

        if(container->hasHexahedronQuads())
        {
            // removes the hexahedronQuads from the hexahedronQuadArray
            container->m_hexahedronQuad[ indices[i] ] = container->m_hexahedronQuad[ lastHexahedron ]; // overwriting with last valid value.
            container->m_hexahedronQuad.resize( lastHexahedron ); // resizing to erase multiple occurence of the hexa.
        }

        if(container->hasHexahedronEdges())
        {
            // removes the hexahedronEdges from the hexahedronEdgeArray
            container->m_hexahedronEdge[ indices[i] ] = container->m_hexahedronEdge[ lastHexahedron ]; // overwriting with last valid value.
            container->m_hexahedronEdge.resize( lastHexahedron ); // resizing to erase multiple occurence of the hexa.
        }

        // removes the hexahedron from the hexahedronArray
        container->m_hexahedron[ indices[i] ] = container->m_hexahedron[ lastHexahedron ]; // overwriting with last valid value.
        container->m_hexahedron.resize( lastHexahedron ); // resizing to erase multiple occurence of the hexa.
    }

    if( (!quadToBeRemoved.empty()) || (!edgeToBeRemoved.empty()))
    {
        if(!quadToBeRemoved.empty())
        {
            /// warn that quads will be deleted
            this->removeQuadsWarning(quadToBeRemoved);
        }

        if(!edgeToBeRemoved.empty())
        {
            /// warn that edges will be deleted
            this->removeEdgesWarning(edgeToBeRemoved);
        }

        container->propagateTopologicalChanges();

        if(!quadToBeRemoved.empty())
        {
            /// actually remove quads without looking for isolated vertices
            this->removeQuadsProcess(quadToBeRemoved, false, false);
        }

        if(!edgeToBeRemoved.empty())
        {
            /// actually remove edges without looking for isolated vertices
            this->removeEdgesProcess(edgeToBeRemoved, false);
        }
    }

    if(!vertexToBeRemoved.empty())
    {
        this->removePointsWarning(vertexToBeRemoved);
        container->propagateTopologicalChanges();
        this->removePointsProcess(vertexToBeRemoved);
    }
}

void HexahedronSetTopologyModifier::addPointsProcess(const unsigned int nPoints,
        const bool addDOF)
{
    // start by calling the parent's method.
    // TODO : only if quads exist, otherwise call the EdgeSet or PointSet method respectively
    QuadSetTopologyModifier::addPointsProcess( nPoints, addDOF );

    HexahedronSetTopologyContainer * container = getHexahedronSetTopologyContainer();

    if(container->hasHexahedronVertexShell())
        container->m_hexahedronVertexShell.resize( container->m_hexahedronVertexShell.size() + nPoints );
}

void HexahedronSetTopologyModifier::addPointsProcess(const unsigned int nPoints,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
        const bool addDOF)
{
    // start by calling the parent's method.
    // TODO : only if quads exist, otherwise call the EdgeSet or PointSet method respectively
    QuadSetTopologyModifier::addPointsProcess( nPoints, ancestors, baryCoefs, addDOF );

    HexahedronSetTopologyContainer * container = getHexahedronSetTopologyContainer();

    if(container->hasHexahedronVertexShell())
        container->m_hexahedronVertexShell.resize( container->m_hexahedronVertexShell.size() + nPoints );
}

void HexahedronSetTopologyModifier::addEdgesProcess(const sofa::helper::vector< Edge > &edges)
{
    // start by calling the parent's method.
    // TODO : only if quads exist, otherwise call the EdgeSet or PointSet method respectively
    QuadSetTopologyModifier::addEdgesProcess( edges );

    HexahedronSetTopologyContainer * container = getHexahedronSetTopologyContainer();

    if(container->hasHexahedronEdgeShell())
        container->m_hexahedronEdgeShell.resize( container->m_hexahedronEdgeShell.size() + edges.size() );
}

void HexahedronSetTopologyModifier::addQuadsProcess(const sofa::helper::vector< Quad > &quads)
{
    // start by calling the parent's method.
    // TODO : only if quads exist
    QuadSetTopologyModifier::addQuadsProcess( quads );

    HexahedronSetTopologyContainer * container = getHexahedronSetTopologyContainer();

    if(!container->m_hexahedronQuadShell.empty())
        container->m_hexahedronQuadShell.resize( container->m_hexahedronQuadShell.size() + quads.size() );
}

void HexahedronSetTopologyModifier::removePointsProcess( sofa::helper::vector<unsigned int> &indices,
        const bool removeDOF)
{
    HexahedronSetTopologyContainer * container = getHexahedronSetTopologyContainer();

    if(!container->hasHexahedronVertexShell())
    {
        container->createHexahedronVertexShellArray();
    }

    unsigned int lastPoint = container->getNbPoints() - 1;
    for(unsigned int i = 0; i < indices.size(); ++i, --lastPoint)
    {
        // updating the edges connected to the point replacing the removed one:
        // for all edges connected to the last point
        for(sofa::helper::vector<unsigned int>::iterator itt=container->m_hexahedronVertexShell[lastPoint].begin();
            itt!=container->m_hexahedronVertexShell[lastPoint].end(); ++itt)
        {
            int vertexIndex = container->getVertexIndexInHexahedron(container->m_hexahedron[(*itt)], lastPoint);
            assert(vertexIndex!= -1);
            container->m_hexahedron[(*itt)][(unsigned int) vertexIndex] = indices[i];
        }

        // updating the edge shell itself (change the old index for the new one)
        container->m_hexahedronVertexShell[ indices[i] ] = container->m_hexahedronVertexShell[ lastPoint ];
    }

    container->m_hexahedronVertexShell.resize( container->m_hexahedronVertexShell.size() - indices.size() );

    // Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    // call the parent's method.
    // TODO : only if quads exist, otherwise call EdgeSet or PointSet method respectively
    QuadSetTopologyModifier::removePointsProcess(  indices, removeDOF );
}

void HexahedronSetTopologyModifier::removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedItems)
{
    HexahedronSetTopologyContainer * container = getHexahedronSetTopologyContainer();

    if(!container->hasEdges()) // TODO : this method should only be called when edges exist
        return;

    if(container->hasHexahedronEdges())
    {
        if(!container->hasHexahedronEdgeShell())
            container->createHexahedronEdgeShellArray();

        unsigned int lastEdge = container->getNumberOfEdges() - 1;
        for(unsigned int i=0; i<indices.size(); ++i, --lastEdge)
        {
            for(sofa::helper::vector<unsigned int>::iterator itt=container->m_hexahedronEdgeShell[lastEdge].begin();
                itt!=container->m_hexahedronEdgeShell[lastEdge].end(); ++itt)
            {
                int edgeIndex = container->getEdgeIndexInHexahedron(container->m_hexahedronEdge[(*itt)], lastEdge);
                assert((int)edgeIndex!= -1);
                container->m_hexahedronEdge[(*itt)][(unsigned int) edgeIndex] = indices[i];
            }

            // updating the edge shell itself (change the old index for the new one)
            container->m_hexahedronEdgeShell[ indices[i] ] = container->m_hexahedronEdgeShell[ lastEdge ];
        }

        container->m_hexahedronEdgeShell.resize( container->m_hexahedronEdgeShell.size() - indices.size() );
    }

    // call the parent's method.
    // TODO : only if quads exist, otherwise call EdgeSet or PointSet method respectively
    QuadSetTopologyModifier::removeEdgesProcess(  indices, removeIsolatedItems );
}

void HexahedronSetTopologyModifier::removeQuadsProcess( const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedEdges,
        const bool removeIsolatedPoints)
{
    HexahedronSetTopologyContainer * container = getHexahedronSetTopologyContainer();

    if(!container->hasQuads()) // TODO : this method should only be called when quads exist
        return;

    if(container->hasHexahedronQuads())
    {
        if(!container->hasHexahedronQuadShell())
            container->createHexahedronQuadShellArray();

        unsigned int lastQuad = container->getNumberOfQuads() - 1;
        for(unsigned int i=0; i<indices.size(); ++i, --lastQuad)
        {
            for(sofa::helper::vector<unsigned int>::iterator itt=container->m_hexahedronQuadShell[lastQuad].begin();
                itt!=container->m_hexahedronQuadShell[lastQuad].end(); ++itt)
            {
                int quadIndex=container->getQuadIndexInHexahedron(container->m_hexahedronQuad[(*itt)],lastQuad);
                assert((int)quadIndex!= -1);
                container->m_hexahedronQuad[(*itt)][(unsigned int)quadIndex]=indices[i];
            }

            // updating the quad shell itself (change the old index for the new one)
            container->m_hexahedronQuadShell[ indices[i] ] = container->m_hexahedronQuadShell[ lastQuad ];
        }
        container->m_hexahedronQuadShell.resize( container->m_hexahedronQuadShell.size() - indices.size() );
    }

    // call the parent's method.
    QuadSetTopologyModifier::removeQuadsProcess( indices, removeIsolatedEdges, removeIsolatedPoints );
}

void HexahedronSetTopologyModifier::renumberPointsProcess( const sofa::helper::vector<unsigned int> &index, const sofa::helper::vector<unsigned int> &inv_index, const bool renumberDOF)
{
    HexahedronSetTopologyContainer * container = getHexahedronSetTopologyContainer();

    if(container->hasHexahedronVertexShell())
    {
        sofa::helper::vector< sofa::helper::vector< unsigned int > > hexahedronVertexShell_cp = container->m_hexahedronVertexShell;
        for(unsigned int i=0; i<index.size(); ++i)
        {
            container->m_hexahedronVertexShell[i] = hexahedronVertexShell_cp[ index[i] ];
        }
    }

    for(unsigned int i=0; i<container->m_hexahedron.size(); ++i)
    {
        container->m_hexahedron[i][0]  = inv_index[ container->m_hexahedron[i][0]  ];
        container->m_hexahedron[i][1]  = inv_index[ container->m_hexahedron[i][1]  ];
        container->m_hexahedron[i][2]  = inv_index[ container->m_hexahedron[i][2]  ];
        container->m_hexahedron[i][3]  = inv_index[ container->m_hexahedron[i][3]  ];
        container->m_hexahedron[i][4]  = inv_index[ container->m_hexahedron[i][4]  ];
        container->m_hexahedron[i][5]  = inv_index[ container->m_hexahedron[i][5]  ];
        container->m_hexahedron[i][6]  = inv_index[ container->m_hexahedron[i][6]  ];
        container->m_hexahedron[i][7]  = inv_index[ container->m_hexahedron[i][7]  ];
    }

    // call the parent's method.
    // TODO : only if quads exist, otherwise call EdgeSet or PointSet method respectively
    QuadSetTopologyModifier::renumberPointsProcess( index, inv_index, renumberDOF );
}

} // namespace topology

} // namespace component

} // namespace sofa

