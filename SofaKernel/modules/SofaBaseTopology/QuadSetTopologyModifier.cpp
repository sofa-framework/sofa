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
#include <SofaBaseTopology/QuadSetTopologyModifier.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/topology/TopologyChange.h>
#include <SofaBaseTopology/QuadSetTopologyContainer.h>
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
SOFA_DECL_CLASS(QuadSetTopologyModifier)
int QuadSetTopologyModifierClass = core::RegisterObject("Quad set topology modifier")
        .add< QuadSetTopologyModifier >();


using namespace std;
using namespace sofa::defaulttype;
using namespace sofa::core::topology;


void QuadSetTopologyModifier::init()
{
    EdgeSetTopologyModifier::init();
    this->getContext()->get(m_container);
}


void QuadSetTopologyModifier::addQuads(const sofa::helper::vector<Quad> &quads)
{
    unsigned int nQuads = m_container->getNbQuads();

    /// effectively add triangles in the topology container
    addQuadsProcess(quads);

    sofa::helper::vector<unsigned int> quadsIndex;
    quadsIndex.reserve(quads.size());

    for (unsigned int i=0; i<quads.size(); ++i)
        quadsIndex.push_back(nQuads+i);

    // add topology event in the stack of topological events
    addQuadsWarning((unsigned int)quads.size(), quads, quadsIndex);

    // inform other objects that the edges are already added
    propagateTopologicalChanges();
}



void QuadSetTopologyModifier::addQuads(const sofa::helper::vector<Quad> &quads,
        const sofa::helper::vector<sofa::helper::vector<unsigned int> > &ancestors,
        const sofa::helper::vector<sofa::helper::vector<double> > &baryCoefs)
{
    unsigned int nQuads = m_container->getNbQuads();

    /// effectively add triangles in the topology container
    addQuadsProcess(quads);

    sofa::helper::vector<unsigned int> quadsIndex;
    quadsIndex.reserve(quads.size());

    for (unsigned int i=0; i<quads.size(); ++i)
        quadsIndex.push_back(nQuads+i);

    // add topology event in the stack of topological events
    addQuadsWarning((unsigned int)quads.size(), quads, quadsIndex, ancestors, baryCoefs);

    // inform other objects that the edges are already added
    propagateTopologicalChanges();
}


void QuadSetTopologyModifier::addQuadProcess(Quad t)
{
#ifndef NDEBUG
    // check if the 4 vertices are different
    if((t[0]==t[1]) || (t[0]==t[2]) || (t[0]==t[3])
       || (t[1]==t[2]) || (t[1]==t[3]) || (t[2]==t[3]))
    {
        sout << "Error: [QuadSetTopologyModifier::addQuad] : invalid quad: "
                << t[0] << ", " << t[1] << ", " << t[2] <<  ", " << t[3] <<  endl;

        return;
    }

    // check if there already exists a quad with the same indices
    // Important: getEdgeIndex creates the quad vertex shell array
    if(m_container->hasQuadsAroundVertex())
    {
        if(m_container->getQuadIndex(t[0],t[1],t[2],t[3]) != -1)
        {
            sout << "Error: [QuadSetTopologyModifier::addQuad] : Quad "
                    << t[0] << ", " << t[1] << ", " << t[2] <<  ", " << t[3] << " already exists." << endl;
            return;
        }
    }
#endif

    const unsigned int quadIndex = m_container->getNumberOfQuads();

    if(m_container->hasQuadsAroundVertex())
    {
        for(unsigned int j=0; j<4; ++j)
        {
            sofa::helper::vector< unsigned int > &shell = m_container->getQuadsAroundVertexForModification( t[j] );
            shell.push_back( quadIndex );
        }
    }

    helper::WriteAccessor< Data< sofa::helper::vector<Quad> > > m_quad = m_container->d_quad;

    if(m_container->hasEdges())
    {
        for(unsigned int j=0; j<4; ++j)
        {
            int edgeIndex = m_container->getEdgeIndex(t[(j+1)%4], t[(j+2)%4]);

            if(edgeIndex == -1)
            {
                // first create the edges
                sofa::helper::vector< Edge > v(1);
                Edge e1 (t[(j+1)%4], t[(j+2)%4]);
                v[0] = e1;

                addEdgesProcess((const sofa::helper::vector< Edge > &) v);

                edgeIndex = m_container->getEdgeIndex(t[(j+1)%4],t[(j+2)%4]);
                sofa::helper::vector< unsigned int > edgeIndexList;
                edgeIndexList.push_back((unsigned int) edgeIndex);
                addEdgesWarning((unsigned int)v.size(), v, edgeIndexList);
            }

            if(m_container->hasEdgesInQuad())
            {
                m_container->m_edgesInQuad.resize(quadIndex+1);
                m_container->m_edgesInQuad[quadIndex][j]= edgeIndex;
            }

            if(m_container->hasQuadsAroundEdge())
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_quadsAroundEdge[m_container->m_edgesInQuad[quadIndex][j]];
                shell.push_back( quadIndex );
            }
        }
    }

    m_quad.push_back(t);
}


void QuadSetTopologyModifier::addQuadsProcess(const sofa::helper::vector< Quad > &quads)
{
    helper::WriteAccessor< Data< sofa::helper::vector<Quad> > > m_quad = m_container->d_quad;
    m_quad.reserve(m_quad.size() + quads.size());

    for(unsigned int i=0; i<quads.size(); ++i)
    {
        addQuadProcess(quads[i]);
    }
}


void QuadSetTopologyModifier::addQuadsWarning(const unsigned int nQuads,
        const sofa::helper::vector< Quad >& quadsList,
        const sofa::helper::vector< unsigned int >& quadsIndexList)
{
    m_container->setQuadTopologyToDirty();
    // Warning that quads just got created
    QuadsAdded *e = new QuadsAdded(nQuads, quadsList, quadsIndexList);
    addTopologyChange(e);
}


void QuadSetTopologyModifier::addQuadsWarning(const unsigned int nQuads,
        const sofa::helper::vector< Quad >& quadsList,
        const sofa::helper::vector< unsigned int >& quadsIndexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    m_container->setQuadTopologyToDirty();
    // Warning that quads just got created
    QuadsAdded *e=new QuadsAdded(nQuads, quadsList,quadsIndexList,ancestors,baryCoefs);
    addTopologyChange(e);
}


void QuadSetTopologyModifier::removeQuadsWarning( sofa::helper::vector<unsigned int> &quads)
{
    m_container->setQuadTopologyToDirty();
    /// sort vertices to remove in a descendent order
    std::sort( quads.begin(), quads.end(), std::greater<unsigned int>() );

    // Warning that these quads will be deleted
    QuadsRemoved *e=new QuadsRemoved(quads);
    addTopologyChange(e);
}


void QuadSetTopologyModifier::removeQuadsProcess(const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedEdges,
        const bool removeIsolatedPoints)
{
    if(!m_container->hasQuads()) // this method should only be called when quads exist
    {
#ifndef NDEBUG
        sout << "Error. [QuadSetTopologyModifier::removeQuadsProcess] quad array is empty." << sendl;
#endif
        return;
    }

    if(m_container->hasEdges() && removeIsolatedEdges)
    {
        if(!m_container->hasEdgesInQuad())
            m_container->createEdgesInQuadArray();

        if(!m_container->hasQuadsAroundEdge())
            m_container->createQuadsAroundEdgeArray();
    }

    if(removeIsolatedPoints)
    {
        if(!m_container->hasQuadsAroundVertex())
            m_container->createQuadsAroundVertexArray();
    }

    sofa::helper::vector<unsigned int> edgeToBeRemoved;
    sofa::helper::vector<unsigned int> vertexToBeRemoved;
    helper::WriteAccessor< Data< sofa::helper::vector<Quad> > > m_quad = m_container->d_quad;

    unsigned int lastQuad = m_container->getNumberOfQuads() - 1;
    for(unsigned int i=0; i<indices.size(); ++i, --lastQuad)
    {
        const Quad &t = m_quad[ indices[i] ];
        const Quad &q = m_quad[ lastQuad ];

        // first check that the quad vertex shell array has been initialized
        if(m_container->hasQuadsAroundVertex())
        {
            for(unsigned int v=0; v<4; ++v)
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_quadsAroundVertex[ t[v] ];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if(removeIsolatedPoints && shell.empty())
                    vertexToBeRemoved.push_back(t[v]);
            }
        }

        if(m_container->hasQuadsAroundEdge())
        {
            for(unsigned int e=0; e<4; ++e)
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_quadsAroundEdge[ m_container->m_edgesInQuad[indices[i]][e]];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if(removeIsolatedEdges && shell.empty())
                    edgeToBeRemoved.push_back(m_container->m_edgesInQuad[indices[i]][e]);
            }
        }

        if(indices[i] < lastQuad)
        {
            // now updates the shell information of the quad at the end of the array
            if(m_container->hasQuadsAroundVertex())
            {
                for(unsigned int v=0; v<4; ++v)
                {
                    sofa::helper::vector< unsigned int > &shell = m_container->m_quadsAroundVertex[ q[v] ];
                    replace(shell.begin(), shell.end(), lastQuad, indices[i]);
                }
            }

            if(m_container->hasQuadsAroundEdge())
            {
                for(unsigned int e=0; e<4; ++e)
                {
                    sofa::helper::vector< unsigned int > &shell =  m_container->m_quadsAroundEdge[ m_container->m_edgesInQuad[lastQuad][e]];
                    replace(shell.begin(), shell.end(), lastQuad, indices[i]);
                }
            }
        }

        // removes the edgesInQuads from the edgesInQuadsArray
        if(m_container->hasEdgesInQuad())
        {
            m_container->m_edgesInQuad[ indices[i] ] = m_container->m_edgesInQuad[ lastQuad ]; // overwriting with last valid value.
            m_container->m_edgesInQuad.resize( lastQuad ); // resizing to erase multiple occurence of the quad.
        }

        // removes the quad from the quadArray
        m_quad[ indices[i] ] = m_quad[ lastQuad ]; // overwriting with last valid value.
        m_quad.resize( lastQuad ); // resizing to erase multiple occurence of the quad.
    }

    if(!edgeToBeRemoved.empty())
    {
        /// warn that edges will be deleted
        removeEdgesWarning(edgeToBeRemoved);
        propagateTopologicalChanges();
        /// actually remove edges without looking for isolated vertices
        removeEdgesProcess(edgeToBeRemoved,false);
    }

    if(!vertexToBeRemoved.empty())
    {
        removePointsWarning(vertexToBeRemoved);
        /// propagate to all components
        propagateTopologicalChanges();
        removePointsProcess(vertexToBeRemoved);
    }
}

void QuadSetTopologyModifier::addPointsProcess(const unsigned int nPoints)
{
    // start by calling the parent's method.
    EdgeSetTopologyModifier::addPointsProcess( nPoints );

    // now update the local container structures.
    if(m_container->hasQuadsAroundVertex())
        m_container->m_quadsAroundVertex.resize( m_container->getNbPoints() );
}

void QuadSetTopologyModifier::addEdgesProcess(const sofa::helper::vector< Edge > &edges)
{
    if(!m_container->hasEdges())
    {
        m_container->createEdgeSetArray();
    }

    // start by calling the parent's method.
    EdgeSetTopologyModifier::addEdgesProcess( edges );

    if(m_container->hasQuadsAroundEdge())
        m_container->m_quadsAroundEdge.resize( m_container->m_quadsAroundEdge.size() + edges.size() );
}

void QuadSetTopologyModifier::removePointsProcess(const sofa::helper::vector<unsigned int> &indices,
        const bool removeDOF)
{
    if(m_container->hasQuads())
    {
        if(!m_container->hasQuadsAroundVertex())
            m_container->createQuadsAroundVertexArray();

        unsigned int lastPoint = m_container->getNbPoints() - 1;
        helper::WriteAccessor< Data< sofa::helper::vector<Quad> > > m_quad = m_container->d_quad;

        for(unsigned int i=0; i<indices.size(); ++i, --lastPoint)
        {
            // updating the quads connected to the point replacing the removed one:
            // for all quads connected to the last point

            sofa::helper::vector<unsigned int> &shell = m_container->m_quadsAroundVertex[lastPoint];
            for(unsigned int j=0; j<shell.size(); ++j)
            {
                const unsigned int q = shell[j];
                for(unsigned int k=0; k<4; ++k)
                {
                    if(m_quad[q][k] == lastPoint)
                        m_quad[q][k] = indices[i];
                }
            }

            // updating the edge shell itself (change the old index for the new one)
            m_container->m_quadsAroundVertex[ indices[i] ] = m_container->m_quadsAroundVertex[ lastPoint ];
        }

        m_container->m_quadsAroundVertex.resize( m_container->m_quadsAroundVertex.size() - indices.size() );
    }

    // Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    // call the parent's method.
    EdgeSetTopologyModifier::removePointsProcess( indices, removeDOF );
}

void QuadSetTopologyModifier::removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedItems)
{
    // Note: this does not check if an edge is removed from an existing quad (it should never happen)

    if(m_container->hasEdgesInQuad()) // this method should only be called when edges exist
    {
        if(!m_container->hasQuadsAroundEdge())
            m_container->createQuadsAroundEdgeArray();

        unsigned int lastEdge = m_container->getNumberOfEdges() - 1;
        for(unsigned int i = 0; i < indices.size(); ++i, --lastEdge)
        {
            // updating the quads connected to the edge replacing the removed one:
            // for all quads connected to the last point
            for(sofa::helper::vector<unsigned int>::iterator itt = m_container->m_quadsAroundEdge[lastEdge].begin();
                itt != m_container->m_quadsAroundEdge[lastEdge].end(); ++itt)
            {
                unsigned int edgeIndex = m_container->getEdgeIndexInQuad(m_container->m_edgesInQuad[*itt], lastEdge);
                m_container->m_edgesInQuad[*itt][edgeIndex] = indices[i];
            }

            // updating the edge shell itself (change the old index for the new one)
            m_container->m_quadsAroundEdge[ indices[i] ] = m_container->m_quadsAroundEdge[ lastEdge ];
        }

        m_container->m_quadsAroundEdge.resize( m_container->getNumberOfEdges() - indices.size() );
    }

    // call the parent's method.
    EdgeSetTopologyModifier::removeEdgesProcess(indices, removeIsolatedItems);
}

void QuadSetTopologyModifier::renumberPointsProcess( const sofa::helper::vector<unsigned int> &index,
        const sofa::helper::vector<unsigned int> &inv_index,
        const bool renumberDOF)
{
    if(m_container->hasQuads())
    {
        helper::WriteAccessor< Data< sofa::helper::vector<Quad> > > m_quad = m_container->d_quad;

        if(m_container->hasQuadsAroundVertex())
        {
            sofa::helper::vector< sofa::helper::vector< unsigned int > > quadsAroundVertex_cp = m_container->m_quadsAroundVertex;
            for(unsigned int i=0; i<index.size(); ++i)
            {
                m_container->m_quadsAroundVertex[i] = quadsAroundVertex_cp[ index[i] ];
            }
        }

        for(unsigned int i=0; i<m_quad.size(); ++i)
        {
            m_quad[i][0]  = inv_index[ m_quad[i][0]  ];
            m_quad[i][1]  = inv_index[ m_quad[i][1]  ];
            m_quad[i][2]  = inv_index[ m_quad[i][2]  ];
            m_quad[i][3]  = inv_index[ m_quad[i][3]  ];
        }
    }

    // call the parent's method
    EdgeSetTopologyModifier::renumberPointsProcess( index, inv_index, renumberDOF );
}

void QuadSetTopologyModifier::removeQuads(const sofa::helper::vector< unsigned int >& quadIds,
        const bool removeIsolatedEdges,
        const bool removeIsolatedPoints)
{
    sofa::helper::vector<unsigned int> quadIds_filtered;
    for (unsigned int i = 0; i < quadIds.size(); i++)
    {
        if( quadIds[i] >= m_container->getNumberOfQuads())
            std::cout << "Error: QuadSetTopologyModifier::removeQuads: quad: "<< quadIds[i] <<" is out of bound and won't be removed." << std::endl;
        else
            quadIds_filtered.push_back(quadIds[i]);
    }

    /// add the topological changes in the queue
    removeQuadsWarning(quadIds_filtered);
    // inform other objects that the quads are going to be removed
    propagateTopologicalChanges();
    // now destroy the old quads.
    removeQuadsProcess( quadIds_filtered, removeIsolatedEdges, removeIsolatedPoints);

    m_container->checkTopology();
}

void QuadSetTopologyModifier::removeItems(const sofa::helper::vector<unsigned int> &items)
{
    removeQuads(items, true, true);
}

void QuadSetTopologyModifier::renumberPoints( const sofa::helper::vector<unsigned int> &index,
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


void QuadSetTopologyModifier::propagateTopologicalEngineChanges()
{
    if (m_container->beginChange() == m_container->endChange()) return; // nothing to do if no event is stored

    if (!m_container->isQuadTopologyDirty()) // quad Data has not been touched
        return EdgeSetTopologyModifier::propagateTopologicalEngineChanges();

    std::list<sofa::core::topology::TopologyEngine *>::iterator it;

    for ( it = m_container->m_enginesList.begin(); it!=m_container->m_enginesList.end(); ++it)
    {
        sofa::core::topology::TopologyEngine* topoEngine = (*it);
        if (topoEngine->isDirty())
        {
#ifndef NDEBUG
            std::cout << "QuadSetTopologyModifier::performing: " << topoEngine->getName() << std::endl;
#endif
            topoEngine->update();
        }
    }

    m_container->cleanQuadTopologyFromDirty();
    EdgeSetTopologyModifier::propagateTopologicalEngineChanges();
}


} // namespace topology

} // namespace component

} // namespace sofa

