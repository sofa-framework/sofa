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
#include <sofa/component/topology/QuadSetTopologyModifier.h>
#include <sofa/component/topology/QuadSetTopologyChange.h>
#include <sofa/component/topology/QuadSetTopologyContainer.h>
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


void QuadSetTopologyModifier::init()
{
    EdgeSetTopologyModifier::init();
    this->getContext()->get(m_container);
}


void QuadSetTopologyModifier::addQuad(Quad t)
{
#ifndef NDEBUG
    // check if the 4 vertices are different
    if((t[0]==t[1]) || (t[0]==t[2]) || (t[0]==t[3])
       || (t[1]==t[2]) || (t[1]==t[3]) || (t[2]==t[3]))
    {
        cout << "Error: [QuadSetTopologyModifier::addQuad] : invalid quad: "
                << t[0] << ", " << t[1] << ", " << t[2] <<  ", " << t[3] <<  endl;

        return;
    }

    // check if there already exists a quad with the same indices
    // Important: getEdgeIndex creates the quad vertex shell array
    if(m_container->hasQuadVertexShell())
    {
        if(m_container->getQuadIndex(t[0],t[1],t[2],t[3]) != -1)
        {
            cout << "Error: [QuadSetTopologyModifier::addQuad] : Quad "
                    << t[0] << ", " << t[1] << ", " << t[2] <<  ", " << t[3] << " already exists." << endl;
            return;
        }
    }
#endif

    const unsigned int quadIndex = m_container->m_quad.size();

    if(m_container->hasQuadVertexShell())
    {
        for(unsigned int j=0; j<4; ++j)
        {
            sofa::helper::vector< unsigned int > &shell = m_container->getQuadVertexShellForModification( t[j] );
            shell.push_back( quadIndex );
        }
    }

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
                this->addEdgesWarning( v.size(), v, edgeIndexList);
            }

            if(m_container->hasQuadEdges())
            {
                m_container->m_quadEdge.resize(quadIndex+1);
                m_container->m_quadEdge[quadIndex][j]= edgeIndex;
            }

            if(m_container->hasQuadEdgeShell())
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_quadEdgeShell[m_container->m_quadEdge[quadIndex][j]];
                shell.push_back( quadIndex );
            }
        }
    }

    m_container->m_quad.push_back(t);
}


void QuadSetTopologyModifier::addQuadsProcess(const sofa::helper::vector< Quad > &quads)
{
    m_container->m_quad.reserve(m_container->m_quad.size() + quads.size());

    for(unsigned int i=0; i<quads.size(); ++i)
    {
        addQuad(quads[i]);
    }
}


void QuadSetTopologyModifier::addQuadsWarning(const unsigned int nQuads,
        const sofa::helper::vector< Quad >& quadsList,
        const sofa::helper::vector< unsigned int >& quadsIndexList)
{
    // Warning that quads just got created
    QuadsAdded *e = new QuadsAdded(nQuads, quadsList, quadsIndexList);
    this->addTopologyChange(e);
}


void QuadSetTopologyModifier::addQuadsWarning(const unsigned int nQuads,
        const sofa::helper::vector< Quad >& quadsList,
        const sofa::helper::vector< unsigned int >& quadsIndexList,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > > & ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs)
{
    // Warning that quads just got created
    QuadsAdded *e=new QuadsAdded(nQuads, quadsList,quadsIndexList,ancestors,baryCoefs);
    this->addTopologyChange(e);
}


void QuadSetTopologyModifier::removeQuadsWarning( sofa::helper::vector<unsigned int> &quads)
{
    /// sort vertices to remove in a descendent order
    std::sort( quads.begin(), quads.end(), std::greater<unsigned int>() );

    // Warning that these quads will be deleted
    QuadsRemoved *e=new QuadsRemoved(quads);
    this->addTopologyChange(e);
}


void QuadSetTopologyModifier::removeQuadsProcess(const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedEdges,
        const bool removeIsolatedPoints)
{
    if(!m_container->hasQuads()) // TODO : this method should only be called when quads exist
    {
#ifndef NDEBUG
        cout << "Error. [QuadSetTopologyModifier::removeQuadsProcess] quad array is empty." << endl;
#endif
        m_container->createQuadSetArray();
    }

    if(m_container->hasEdges() && removeIsolatedEdges)
    {
        if(!m_container->hasQuadEdges())
            m_container->createQuadEdgeArray();

        if(!m_container->hasQuadEdgeShell())
            m_container->createQuadEdgeShellArray();
    }

    if(removeIsolatedPoints)
    {
        if(!m_container->hasQuadVertexShell())
            m_container->createQuadVertexShellArray();
    }

    sofa::helper::vector<unsigned int> edgeToBeRemoved;
    sofa::helper::vector<unsigned int> vertexToBeRemoved;

    for(unsigned int i = 0; i<indices.size(); ++i)
    {
        const unsigned int lastQuad = m_container->m_quad.size() - 1;
        Quad &t = m_container->m_quad[ indices[i] ];
        Quad &q = m_container->m_quad[ lastQuad ];

        // first check that the quad vertex shell array has been initialized
        if(m_container->hasQuadVertexShell())
        {
            for(unsigned int v=0; v<4; ++v)
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_quadVertexShell[ t[v] ];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if((removeIsolatedPoints) && shell.empty())
                    vertexToBeRemoved.push_back(t[v]);
            }
        }

        /** first check that the quad edge shell array has been initialized */
        if(m_container->hasQuadEdgeShell())
        {
            for(unsigned int e=0; e<4; ++e)
            {
                sofa::helper::vector< unsigned int > &shell = m_container->m_quadEdgeShell[ m_container->m_quadEdge[indices[i]][e]];
                shell.erase(remove(shell.begin(), shell.end(), indices[i]), shell.end());
                if((removeIsolatedEdges) && shell.empty())
                    edgeToBeRemoved.push_back(m_container->m_quadEdge[indices[i]][e]);
            }
        }

        if(indices[i] < lastQuad)
        {
            // now updates the shell information of the quad at the end of the array
            if(m_container->hasQuadVertexShell())
            {
                for(unsigned int v=0; v<4; ++v)
                {
                    sofa::helper::vector< unsigned int > &shell = m_container->m_quadVertexShell[ q[v] ];
                    replace(shell.begin(), shell.end(), lastQuad, indices[i]);
                }
            }

            if(m_container->hasQuadEdgeShell())
            {
                for(unsigned int e=0; e<4; ++e)
                {
                    sofa::helper::vector< unsigned int > &shell =  m_container->m_quadEdgeShell[ m_container->m_quadEdge[lastQuad][e]];
                    replace(shell.begin(), shell.end(), lastQuad, indices[i]);
                }
            }
        }

        // removes the quadEdges from the quadEdgesArray
        if(m_container->hasQuadEdges())
        {
            m_container->m_quadEdge[ indices[i] ] = m_container->m_quadEdge[ lastQuad ]; // overwriting with last valid value.
            m_container->m_quadEdge.resize( lastQuad ); // resizing to erase multiple occurence of the quad.
        }

        // removes the quad from the quadArray
        m_container->m_quad[ indices[i] ] = m_container->m_quad[ lastQuad ]; // overwriting with last valid value.
        m_container->m_quad.resize( lastQuad ); // resizing to erase multiple occurence of the quad.
    }

    if(!edgeToBeRemoved.empty())
    {
        /// warn that edges will be deleted
        this->removeEdgesWarning(edgeToBeRemoved);
        m_container->propagateTopologicalChanges();
        /// actually remove edges without looking for isolated vertices
        this->removeEdgesProcess(edgeToBeRemoved,false);
    }

    if(!vertexToBeRemoved.empty())
    {
        this->removePointsWarning(vertexToBeRemoved);
        /// propagate to all components
        m_container->propagateTopologicalChanges();
        this->removePointsProcess(vertexToBeRemoved);
    }
}

void QuadSetTopologyModifier::addPointsProcess(const unsigned int nPoints, const bool addDOF)
{
    // start by calling the parent's method.
    // TODO : only if edges exist, otherwise call the PointSet method
    EdgeSetTopologyModifier::addPointsProcess( nPoints, addDOF );

    // now update the local container structures.
    if(m_container->hasQuadVertexShell())
        m_container->m_quadVertexShell.resize( m_container->m_quadVertexShell.size() + nPoints );
}

void QuadSetTopologyModifier::addPointsProcess(const unsigned int nPoints,
        const sofa::helper::vector< sofa::helper::vector< unsigned int > >& ancestors,
        const sofa::helper::vector< sofa::helper::vector< double > >& baryCoefs,
        const bool addDOF)
{
    // start by calling the parent's method.
    // TODO : only if edges exist, otherwise call the PointSet method
    EdgeSetTopologyModifier::addPointsProcess( nPoints, ancestors, baryCoefs, addDOF );

    // now update the local container structures.
    if(m_container->hasQuadVertexShell())
        m_container->m_quadVertexShell.resize( m_container->m_quadVertexShell.size() + nPoints );
}

void QuadSetTopologyModifier::addEdgesProcess(const sofa::helper::vector< Edge > &edges)
{
    // now update the local container structures.
    if(!m_container->hasEdges())
    {
        m_container->createEdgeSetArray();
    }

    // start by calling the parent's method.
    EdgeSetTopologyModifier::addEdgesProcess( edges );

    if(m_container->hasQuadEdgeShell())
        m_container->m_quadEdgeShell.resize( m_container->m_quadEdgeShell.size() + edges.size() );
}

void QuadSetTopologyModifier::removePointsProcess( sofa::helper::vector<unsigned int> &indices,
        const bool removeDOF)
{
    // now update the local container structures

    // force the creation of the quad vertex shell array before any point is deleted
    if(!m_container->hasQuadVertexShell())
        m_container->createQuadVertexShellArray();

    unsigned int lastPoint = m_container->getNbPoints() - 1;
    for(unsigned int i=0; i<indices.size(); ++i, --lastPoint)
    {
        // updating the quads connected to the point replacing the removed one:
        // for all quads connected to the last point

        sofa::helper::vector<unsigned int> &shell = m_container->m_quadVertexShell[lastPoint];
        for(unsigned int j=0; j<shell.size(); ++j)
        {
            const unsigned int q = shell[j];
            for(unsigned int k=0; k<4; ++k)
            {
                if(m_container->m_quad[q][k] == lastPoint)
                    m_container->m_quad[q][k] = indices[i];
            }
        }

        // updating the edge shell itself (change the old index for the new one)
        m_container->m_quadVertexShell[ indices[i] ] = m_container->m_quadVertexShell[ lastPoint ];
    }

    m_container->m_quadVertexShell.resize( m_container->m_quadVertexShell.size() - indices.size() );

    // Important : the points are actually deleted from the mechanical object's state vectors iff (removeDOF == true)
    // call the parent's method.
    // TODO : only if edges exist, otherwise call PointSetMethod
    EdgeSetTopologyModifier::removePointsProcess( indices, removeDOF );
}

void QuadSetTopologyModifier::removeEdgesProcess( const sofa::helper::vector<unsigned int> &indices,
        const bool removeIsolatedItems)
{
    // TODO : clarify what exactly has to happen here (what if an edge is removed from an existing quad?)

    // now update the local container structures
    if(!m_container->hasEdges()) // TODO : this method should only be called when edges exist
    {
#ifndef NDEBUG
        cout << "Warning. [QuadSetTopologyModifier::removeEdgesProcess] edge array is empty." << endl;
#endif
        m_container->createEdgeSetArray();
    }

    if(!m_container->hasQuadEdgeShell())
        m_container->createQuadEdgeShellArray();

    if(!m_container->hasQuadEdges())
        m_container->createQuadEdgeArray();

    unsigned int edgeIndex;
    unsigned int lastEdge = m_container->getNumberOfEdges() - 1;
    for(unsigned int i = 0; i < indices.size(); ++i, --lastEdge)
    {
        // updating the quads connected to the edge replacing the removed one:
        // for all quads connected to the last point
        for(sofa::helper::vector<unsigned int>::iterator itt = m_container->m_quadEdgeShell[lastEdge].begin();
            itt != m_container->m_quadEdgeShell[lastEdge].end(); ++itt)
        {
            edgeIndex = m_container->getEdgeIndexInQuad(m_container->m_quadEdge[(*itt)], lastEdge);
            assert((int)edgeIndex!= -1);
            m_container->m_quadEdge[(*itt)][(unsigned int)edgeIndex] = indices[i];
        }

        // updating the edge shell itself (change the old index for the new one)
        m_container->m_quadEdgeShell[ indices[i] ] = m_container->m_quadEdgeShell[ lastEdge ];
    }

    m_container->m_quadEdgeShell.resize( m_container->m_quadEdgeShell.size() - indices.size() );

    // call the parent's method.
    EdgeSetTopologyModifier::removeEdgesProcess(indices, removeIsolatedItems);
}

void QuadSetTopologyModifier::renumberPointsProcess( const sofa::helper::vector<unsigned int> &index,
        const sofa::helper::vector<unsigned int> &inv_index,
        const bool renumberDOF)
{
    // now update the local container structures.
    if(!m_container->hasQuads()) // TODO : this method should only be called when quads exist
    {
#ifndef NDEBUG
        cout << "Error. [QuadSetTopologyModifier::renumberPointsProcess] quad array is empty." << endl;
#endif
        m_container->createQuadSetArray();
    }

    if(m_container->hasQuadVertexShell())
    {
        sofa::helper::vector< sofa::helper::vector< unsigned int > > quadVertexShell_cp = m_container->m_quadVertexShell;
        for(unsigned int i=0; i<index.size(); ++i)
        {
            m_container->m_quadVertexShell[i] = quadVertexShell_cp[ index[i] ];
        }
    }

    for(unsigned int i=0; i<m_container->m_quad.size(); ++i)
    {
        m_container->m_quad[i][0]  = inv_index[ m_container->m_quad[i][0]  ];
        m_container->m_quad[i][1]  = inv_index[ m_container->m_quad[i][1]  ];
        m_container->m_quad[i][2]  = inv_index[ m_container->m_quad[i][2]  ];
        m_container->m_quad[i][3]  = inv_index[ m_container->m_quad[i][3]  ];
    }

    // call the parent's method
    if(m_container->hasEdges())
        EdgeSetTopologyModifier::renumberPointsProcess( index, inv_index, renumberDOF );
    else
        PointSetTopologyModifier::renumberPointsProcess( index, inv_index, renumberDOF );
}

} // namespace topology

} // namespace component

} // namespace sofa

