/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "ManifoldEdgeSetTopologyContainer.h"

#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>
// Use BOOST GRAPH LIBRARY :

#include <boost/config.hpp>
#include <iostream>
#include <vector>
#include <utility>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/bandwidth.hpp>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace std;
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(ManifoldEdgeSetTopologyContainer)
int ManifoldEdgeSetTopologyContainerClass = core::RegisterObject("ManifoldEdge set topology container")
        .add< ManifoldEdgeSetTopologyContainer >()
        ;

ManifoldEdgeSetTopologyContainer::ManifoldEdgeSetTopologyContainer()
    : EdgeSetTopologyContainer( )
{}


void ManifoldEdgeSetTopologyContainer::init()
{
    // load edges
    EdgeSetTopologyContainer::init();

    // the edgesAroundVertex is needed to recognize if the edgeSet is manifold
    createEdgesAroundVertexArray();

    computeConnectedComponent();
    checkTopology();
}

void ManifoldEdgeSetTopologyContainer::createEdgesAroundVertexArray()
{
    if(!hasEdges())	//  this method should only be called when edges exist
    {
#ifndef NDEBUG
        sout << "Warning. [ManifoldEdgeSetTopologyContainer::createEdgesAroundVertexArray] edge array is empty." << endl;
#endif
        createEdgeSetArray();
    }

    if(hasEdgesAroundVertex())
    {
        clearEdgesAroundVertex();
    }

    m_edgesAroundVertex.resize( getNbPoints() );

    helper::ReadAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;
    for (unsigned int edge = 0; edge < m_edge.size(); ++edge)
    {
        // check to how many edges is the end vertex of each edge connnected to
        unsigned int size1 = m_edgesAroundVertex[ m_edge[edge][1] ].size();

        // adding edge i in the edge shell of both points, while respecting the manifold orientation
        // (ie : the edge will be added in second position for its first extremity point, and in first position for its second extremity point)

        m_edgesAroundVertex[ m_edge[edge][0] ].push_back( edge );

        if(size1==0)
        {
            m_edgesAroundVertex[ m_edge[edge][1] ].push_back( edge );
        }
        else if(size1==1)
        {
            unsigned int nextEdge = m_edgesAroundVertex[ m_edge[edge][1] ][0];
            m_edgesAroundVertex[ m_edge[edge][1] ][0] = edge;
            m_edgesAroundVertex[ m_edge[edge][1] ].push_back( nextEdge );
        }
        else
        {
            // not manifold !!!
            m_edgesAroundVertex[ m_edge[edge][1] ].push_back( edge );

            sout << "Error. [ManifoldEdgeSetTopologyContainer::createEdgesAroundVertexArray] The given EdgeSet is not manifold." << endl;
        }
    }
}

// Return the number of connected components
int ManifoldEdgeSetTopologyContainer::getNumberConnectedComponents(sofa::helper::vector<unsigned int>& components) // const
{
    computeConnectedComponent();

    components = m_ComponentVertexArray;
    return m_ConnectedComponentArray.size();
}

void ManifoldEdgeSetTopologyContainer::computeConnectedComponent()
{
    using namespace boost;
    typedef adjacency_list <vecS, vecS, undirectedS> Graph;

    if(isvoid_ConnectedComponent())
    {

        Graph G;

        const sofa::helper::vector<Edge> &ea=getEdgeArray();
        for (unsigned int k=0; k<ea.size(); ++k)
        {
            add_edge(ea[k][0], ea[k][1], G);
        }

        m_ComponentVertexArray.resize(num_vertices(G));
        int num = connected_components(G, &m_ComponentVertexArray[0]);

        std::vector< std::vector<int> > components(num);
        for(int i=0; i<num; i++)
        {
            components[i].resize(4);
            components[i][0]=0;
            components[i][1]=-1;
            components[i][2]=-1;
            components[i][3]=-1;
        }

        for(unsigned int j=0; j<m_ComponentVertexArray.size(); j++)
        {

            components[m_ComponentVertexArray[j]][0]+=1;
            components[m_ComponentVertexArray[j]][1]=j;

            if((getEdgesAroundVertex(j)).size()==1)
            {

                if((getEdge((getEdgesAroundVertex(j))[0]))[0]==j)
                {
                    components[m_ComponentVertexArray[j]][2]=j;
                }
                else   // (getEdge((getEdgesAroundVertex(j))[0]))[1]==j
                {
                    components[m_ComponentVertexArray[j]][3]=j;
                }
            }
        }

        for(int i=0; i<num; i++)
        {

            bool is_closed = (components[i][2]==-1 && components[i][3]==-1);
            if(is_closed)
            {
                components[i][2]=components[i][1];
            }
            ConnectedComponent cc= ConnectedComponent(components[i][2], components[i][3], components[i][0], i);
            m_ConnectedComponentArray.push_back(cc);
        }

    }
    else
    {
        return;
    }
}

bool ManifoldEdgeSetTopologyContainer::checkTopology() const
{
#ifndef NDEBUG
    bool ret = true;

    if (hasEdgesAroundVertex())
    {
        for (unsigned int i=0; i<m_edgesAroundVertex.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &es = m_edgesAroundVertex[i];

            if(es.size() != 1 && es.size() != 2)
            {
                //serr << "ERROR: ManifoldEdgeSetTopologyContainer::checkTopology() fails ."<<sendl;
                std::cout << "*** CHECK FAILED : check_manifold_edge_vertex_shell, i = " << i << std::endl;
                ret = false;
            }
        }
    }

    return ret &&  EdgeSetTopologyContainer::checkTopology();
#else
    return true;
#endif
}

void ManifoldEdgeSetTopologyContainer::clear()
{
    m_ComponentVertexArray.clear();
    m_ConnectedComponentArray.clear();

    EdgeSetTopologyContainer::clear();
}

} // namespace topology

} // namespace component

} // namespace sofa

