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
#include <sofa/component/topology/container/dynamic/QuadSetTopologyModifier.h>

#include <sofa/component/topology/container/dynamic/QuadSetTopologyContainer.h>
#include <sofa/core/topology/TopologyChange.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/topology/TopologyHandler.h>

#include <algorithm>

namespace sofa::component::topology::container::dynamic
{
int QuadSetTopologyModifierClass = core::RegisterObject("Quad set topology modifier")
        .add< QuadSetTopologyModifier >();


using namespace std;
using namespace sofa::defaulttype;
using namespace sofa::core::topology;


void QuadSetTopologyModifier::init()
{
    EdgeSetTopologyModifier::init();
    this->getContext()->get(m_container);

    if(!m_container)
    {
        msg_error() << "QuadSetTopologyContainer not found in current node: " << this->getContext()->getName();
        d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
}


void QuadSetTopologyModifier::addQuads(const sofa::type::vector<Quad> &quads)
{
    const size_t nQuads = m_container->getNbQuads();

    /// effectively add triangles in the topology container
    addQuadsProcess(quads);

    sofa::type::vector<QuadID> quadsIndex;
    quadsIndex.reserve(quads.size());

    for (size_t i=0; i<quads.size(); ++i)
        quadsIndex.push_back(QuadID(nQuads+i));

    // add topology event in the stack of topological events
    addQuadsWarning(quads.size(), quads, quadsIndex);

    // inform other objects that the edges are already added
    propagateTopologicalChanges();
}



void QuadSetTopologyModifier::addQuads(const sofa::type::vector<Quad> &quads,
        const sofa::type::vector<sofa::type::vector<QuadID> > &ancestors,
        const sofa::type::vector<sofa::type::vector<SReal> > &baryCoefs)
{
    const size_t nQuads = m_container->getNbQuads();

    /// effectively add triangles in the topology container
    addQuadsProcess(quads);

    sofa::type::vector<QuadID> quadsIndex;
    quadsIndex.reserve(quads.size());

    for (size_t i=0; i<quads.size(); ++i)
        quadsIndex.push_back(QuadID(nQuads+i));

    // add topology event in the stack of topological events
    addQuadsWarning(quads.size(), quads, quadsIndex, ancestors, baryCoefs);

    // inform other objects that the edges are already added
    propagateTopologicalChanges();
}


void QuadSetTopologyModifier::addQuadProcess(Quad t)
{
	if (m_container->d_checkTopology.getValue())
	{
		// check if the 4 vertices are different
		if ((t[0] == t[1]) || (t[0] == t[2]) || (t[0] == t[3])
			|| (t[1] == t[2]) || (t[1] == t[3]) || (t[2] == t[3]))
		{
			msg_error() << "Invalid quad: "	<< t[0] << ", " << t[1] << ", " << t[2] << ", " << t[3];

			return;
		}

		// check if there already exists a quad with the same indices
		// Important: getEdgeIndex creates the quad vertex shell array
		if (m_container->hasQuadsAroundVertex())
		{
            if (m_container->getQuadIndex(t[0], t[1], t[2], t[3]) != sofa::InvalidID)
			{
				msg_error() << "Quad " << t[0] << ", " << t[1] << ", " << t[2] << ", " << t[3] << " already exists.";
				return;
			}
		}
	}

    const QuadID quadIndex = (QuadID)m_container->getNumberOfQuads();
    helper::WriteAccessor< Data< sofa::type::vector<Quad> > > m_quad = m_container->d_quad;

    // update nbr point if needed
    unsigned int nbrP = m_container->getNbPoints();
    for(unsigned int i=0; i<4; ++i)
        if (t[i] + 1 > nbrP) // point not well init
        {
            nbrP = t[i] + 1;
            m_container->setNbPoints(nbrP);
        }

    // update m_quadsAroundVertex
    if (m_container->m_quadsAroundVertex.size() < nbrP)
        m_container->m_quadsAroundVertex.resize(nbrP);

    for(PointID j=0; j<4; ++j)
    {
        sofa::type::vector< QuadID > &shell = m_container->m_quadsAroundVertex[t[j]];
        shell.push_back( quadIndex );
    }


    // update edge-quad cross buffers
    if (m_container->m_edgesInQuad.size() < quadIndex+1)
        m_container->m_edgesInQuad.resize(quadIndex+1);

    for(PointID j=0; j<4; ++j)
    {
        EdgeID edgeIndex = m_container->getEdgeIndex(t[(j+1)%4], t[(j+2)%4]);

        if(edgeIndex == sofa::InvalidID)
        {
            // first create the edges
            sofa::type::vector< Edge > v(1);
            Edge e1 (t[(j+1)%4], t[(j+2)%4]);
            v[0] = e1;

            addEdgesProcess((const sofa::type::vector< Edge > &) v);

            edgeIndex = m_container->getEdgeIndex(t[(j+1)%4],t[(j+2)%4]);
            assert(edgeIndex != sofa::InvalidID);
            if (edgeIndex == sofa::InvalidID)
            {
                msg_error() << "Edge creation: " << e1 << " failed in addQuadProcess. Edge will not be added in buffers.";
                continue;
            }

            sofa::type::vector< EdgeID > edgeIndexList;
            edgeIndexList.push_back((EdgeID) edgeIndex);
            addEdgesWarning(sofa::Size(v.size()), v, edgeIndexList);
        }

        // update m_edgesInQuad
        m_container->m_edgesInQuad[quadIndex][j]= edgeIndex;

        // update m_quadsAroundEdge
        if(m_container->m_quadsAroundEdge.size() < m_container->getNbEdges())
            m_container->m_quadsAroundEdge.resize(m_container->getNbEdges());

        sofa::type::vector< QuadID > &shell = m_container->m_quadsAroundEdge[edgeIndex];
        shell.push_back( quadIndex );
    }

    m_quad.push_back(t);
}


void QuadSetTopologyModifier::addQuadsProcess(const sofa::type::vector< Quad > &quads)
{
    helper::WriteAccessor< Data< sofa::type::vector<Quad> > > m_quad = m_container->d_quad;
    m_quad.reserve(m_quad.size() + quads.size());

    for(size_t i=0; i<quads.size(); ++i)
    {
        addQuadProcess(quads[i]);
    }
}


void QuadSetTopologyModifier::addQuadsWarning(const size_t nQuads,
        const sofa::type::vector< Quad >& quadsList,
        const sofa::type::vector< QuadID >& quadsIndexList)
{
    m_container->setQuadTopologyToDirty();
    // Warning that quads just got created
    const QuadsAdded *e = new QuadsAdded(nQuads, quadsList, quadsIndexList);
    addTopologyChange(e);
}


void QuadSetTopologyModifier::addQuadsWarning(const size_t nQuads,
        const sofa::type::vector< Quad >& quadsList,
        const sofa::type::vector< QuadID >& quadsIndexList,
        const sofa::type::vector< sofa::type::vector< QuadID > > & ancestors,
        const sofa::type::vector< sofa::type::vector< SReal > >& baryCoefs)
{
    m_container->setQuadTopologyToDirty();
    // Warning that quads just got created
    const QuadsAdded *e=new QuadsAdded(nQuads, quadsList,quadsIndexList,ancestors,baryCoefs);
    addTopologyChange(e);
}


void QuadSetTopologyModifier::removeQuadsWarning( sofa::type::vector<QuadID> &quads)
{
    m_container->setQuadTopologyToDirty();
    /// sort vertices to remove in a descendent order
    std::sort( quads.begin(), quads.end(), std::greater<QuadID>() );

    // Warning that these quads will be deleted
    const QuadsRemoved *e=new QuadsRemoved(quads);
    addTopologyChange(e);
}


void QuadSetTopologyModifier::removeQuadsProcess(const sofa::type::vector<QuadID> &indices,
        const bool removeIsolatedEdges,
        const bool removeIsolatedPoints)
{
    if(!m_container->hasQuads()) // this method should only be called when quads exist
    {
        msg_error() << "Quad array is empty.";
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

    sofa::type::vector<EdgeID> edgeToBeRemoved;
    sofa::type::vector<PointID> vertexToBeRemoved;
    helper::WriteAccessor< Data< sofa::type::vector<Quad> > > m_quad = m_container->d_quad;

    size_t lastQuad = m_container->getNumberOfQuads() - 1;
    for(size_t i=0; i<indices.size(); ++i, --lastQuad)
    {
        const Quad &t = m_quad[ indices[i] ];
        const Quad &q = m_quad[ lastQuad ];

        // first check that the quad vertex shell array has been initialized
        if(m_container->hasQuadsAroundVertex())
        {
            for(PointID v=0; v<4; ++v)
            {
                sofa::type::vector< QuadID > &shell = m_container->m_quadsAroundVertex[ t[v] ];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if(removeIsolatedPoints && shell.empty())
                    vertexToBeRemoved.push_back(t[v]);
            }
        }

        if(m_container->hasQuadsAroundEdge())
        {
            for(EdgeID e=0; e<4; ++e)
            {
                sofa::type::vector< QuadID > &shell = m_container->m_quadsAroundEdge[ m_container->m_edgesInQuad[indices[i]][e]];
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
                for(PointID v=0; v<4; ++v)
                {
                    sofa::type::vector< QuadID > &shell = m_container->m_quadsAroundVertex[ q[v] ];
                    replace(shell.begin(), shell.end(), (QuadID)lastQuad, indices[i]);
                }
            }

            if(m_container->hasQuadsAroundEdge())
            {
                for(EdgeID e=0; e<4; ++e)
                {
                    sofa::type::vector< QuadID > &shell =  m_container->m_quadsAroundEdge[ m_container->m_edgesInQuad[lastQuad][e]];
                    replace(shell.begin(), shell.end(), (QuadID)lastQuad, indices[i]);
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
        removePointsProcess(vertexToBeRemoved, d_propagateToDOF.getValue());
    }
}

void QuadSetTopologyModifier::addPointsProcess(const sofa::Size nPoints)
{
    // start by calling the parent's method.
    EdgeSetTopologyModifier::addPointsProcess( nPoints );

    // now update the local container structures.
    if(m_container->hasQuadsAroundVertex())
        m_container->m_quadsAroundVertex.resize( m_container->getNbPoints() );
}

void QuadSetTopologyModifier::addEdgesProcess(const sofa::type::vector< Edge > &edges)
{
    // start by calling the parent's method.
    EdgeSetTopologyModifier::addEdgesProcess( edges );

    if(m_container->hasQuadsAroundEdge())
        m_container->m_quadsAroundEdge.resize( m_container->m_quadsAroundEdge.size() + edges.size() );
}

void QuadSetTopologyModifier::removePointsProcess(const sofa::type::vector<PointID> &indices,
        const bool removeDOF)
{
    if(m_container->hasQuads())
    {
        if(!m_container->hasQuadsAroundVertex())
            m_container->createQuadsAroundVertexArray();

        size_t lastPoint = m_container->getNbPoints() - 1;
        helper::WriteAccessor< Data< sofa::type::vector<Quad> > > m_quad = m_container->d_quad;

        for(size_t i=0; i<indices.size(); ++i, --lastPoint)
        {
            // updating the quads connected to the point replacing the removed one:
            // for all quads connected to the last point

            sofa::type::vector<QuadID> &shell = m_container->m_quadsAroundVertex[lastPoint];
            for(size_t j=0; j<shell.size(); ++j)
            {
                const QuadID q = shell[j];
                for(PointID k=0; k<4; ++k)
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

void QuadSetTopologyModifier::removeEdgesProcess( const sofa::type::vector<EdgeID> &indices,
        const bool removeIsolatedItems)
{
    // Note: this does not check if an edge is removed from an existing quad (it should never happen)

    if(m_container->hasEdgesInQuad()) // this method should only be called when edges exist
    {
        if(!m_container->hasQuadsAroundEdge())
            m_container->createQuadsAroundEdgeArray();

        size_t lastEdge = m_container->getNumberOfEdges() - 1;
        for(size_t i = 0; i < indices.size(); ++i, --lastEdge)
        {
            // updating the quads connected to the edge replacing the removed one:
            // for all quads connected to the last point
            for(sofa::type::vector<QuadID>::iterator itt = m_container->m_quadsAroundEdge[lastEdge].begin();
                itt != m_container->m_quadsAroundEdge[lastEdge].end(); ++itt)
            {
                const EdgeID edgeIndex = m_container->getEdgeIndexInQuad(m_container->m_edgesInQuad[*itt], (EdgeID)lastEdge);
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

void QuadSetTopologyModifier::renumberPointsProcess( const sofa::type::vector<PointID> &index,
        const sofa::type::vector<PointID> &inv_index,
        const bool renumberDOF)
{
    if(m_container->hasQuads())
    {
        helper::WriteAccessor< Data< sofa::type::vector<Quad> > > m_quad = m_container->d_quad;

        if(m_container->hasQuadsAroundVertex())
        {
            sofa::type::vector< sofa::type::vector< QuadID > > quadsAroundVertex_cp = m_container->m_quadsAroundVertex;
            for(size_t i=0; i<index.size(); ++i)
            {
                m_container->m_quadsAroundVertex[i] = quadsAroundVertex_cp[ index[i] ];
            }
        }

        for(size_t i=0; i<m_quad.size(); ++i)
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

void QuadSetTopologyModifier::removeQuads(const sofa::type::vector< QuadID >& quadIds,
        const bool removeIsolatedEdges,
        const bool removeIsolatedPoints)
{
    sofa::type::vector<QuadID> quadIds_filtered;
    for (size_t i = 0; i < quadIds.size(); i++)
    {
        if( quadIds[i] >= m_container->getNumberOfQuads())
            dmsg_warning() << "Quad: "<< quadIds[i] <<" is out of bound and won't be removed.";
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

void QuadSetTopologyModifier::removeItems(const sofa::type::vector<QuadID> &items)
{
    removeQuads(items, true, true);
}


void QuadSetTopologyModifier::propagateTopologicalEngineChanges()
{
    if (m_container->beginChange() == m_container->endChange()) return; // nothing to do if no event is stored

    if (!m_container->isQuadTopologyDirty()) // quad Data has not been touched
        return EdgeSetTopologyModifier::propagateTopologicalEngineChanges();

    auto& quadTopologyHandlerList = m_container->getTopologyHandlerList(sofa::geometry::ElementType::QUAD);
    for (const auto topoHandler : quadTopologyHandlerList)
    {
        if (topoHandler->isDirty())
        {
            topoHandler->update();
        }
    }

    m_container->cleanQuadTopologyFromDirty();
    EdgeSetTopologyModifier::propagateTopologicalEngineChanges();
}


} //namespace sofa::component::topology::container::dynamic
