/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/component/topology/EdgeSetTopologyContainer.h>

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

#include <sofa/component/container/MeshLoader.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace std;
using namespace sofa::defaulttype;
SOFA_DECL_CLASS(EdgeSetTopologyContainer)
int EdgeSetTopologyContainerClass = core::RegisterObject("Edge set topology container")
        .add< EdgeSetTopologyContainer >()
        ;

EdgeSetTopologyContainer::EdgeSetTopologyContainer()
    : PointSetTopologyContainer( )
    , d_edge(initData(&d_edge, "edges", "List of edge indices"))
{
}


void EdgeSetTopologyContainer::init()
{
    d_edge.updateIfDirty(); // make sure m_edge is up to date

    helper::ReadAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;
    if (!m_edge.empty())
    {
        for (unsigned int i=0; i<m_edge.size(); ++i)
        {
            for(unsigned int j=0; j<2; ++j)
            {
                int a = m_edge[i][j];
                if (a >= getNbPoints()) nbPoints.setValue(a+1);
            }
        }
    }
    // std::cout << "coords: " << getPX(m_edge[1][0]) << " " << getPY(m_edge[1][0]) << " " << getPZ(m_edge[1][0]) << std::endl;
    PointSetTopologyContainer::init();
}

void EdgeSetTopologyContainer::loadFromMeshLoader(sofa::component::container::MeshLoader* loader)
{
    // load points
    helper::ReadAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;
    if (!m_edge.empty()) return;
    PointSetTopologyContainer::loadFromMeshLoader(loader);
    loader->getEdges(*(d_edge.beginEdit()));
    d_edge.endEdit();
}

void EdgeSetTopologyContainer::addEdge(int a, int b)
{
    helper::WriteAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;
    m_edge.push_back(Edge(a,b));
    if (a >= getNbPoints()) nbPoints.setValue(a+1);
    if (b >= getNbPoints()) nbPoints.setValue(b+1);
}

void EdgeSetTopologyContainer::createEdgesAroundVertexArray()
{
    if(!hasEdges())	// this method should only be called when edges exist
    {
#ifndef NDEBUG
        sout << "Warning. [EdgeSetTopologyContainer::createEdgesAroundVertexArray] edge array is empty." << endl;
#endif
        createEdgeSetArray();
    }

    if(hasEdgesAroundVertex())
    {
        clearEdgesAroundVertex();
    }

    helper::ReadAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;
    m_edgesAroundVertex.resize( getNbPoints() );
    for (unsigned int edge=0; edge<m_edge.size(); ++edge)
    {
        // adding edge in the edge shell of both points
        m_edgesAroundVertex[ m_edge[edge][0] ].push_back(edge);
        m_edgesAroundVertex[ m_edge[edge][1] ].push_back(edge);
    }
}

void EdgeSetTopologyContainer::createEdgeSetArray()
{
#ifndef NDEBUG
    sout << "Error. [EdgeSetTopologyContainer::createEdgeSetArray] This method must be implemented by a child topology." << endl;
#endif
}

const sofa::helper::vector<Edge> &EdgeSetTopologyContainer::getEdgeArray()
{
    if(!hasEdges() && getNbPoints()>0)
    {
#ifndef NDEBUG
        sout << "Warning. [EdgeSetTopologyContainer::getEdgeArray] creating edge array." << endl;
#endif
        createEdgeSetArray();
    }

    return d_edge.getValue();
}

int EdgeSetTopologyContainer::getEdgeIndex(PointID v1, PointID v2)
{
    if(!hasEdges()) // this method should only be called when edges exist
    {
#ifndef NDEBUG
        sout << "Warning. [EdgeSetTopologyContainer::getEdgeIndex] edge array is empty." << endl;
#endif
        createEdgeSetArray();
    }

    if(!hasEdgesAroundVertex())
        createEdgesAroundVertexArray();

    const sofa::helper::vector< unsigned int > &es1 = getEdgesAroundVertex(v1) ;
    helper::ReadAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;

    int result = -1;
    for(unsigned int i=0; (i < es1.size()) && (result == -1); ++i)
    {
        const Edge &e = m_edge[ es1[i] ];
        if ((e[0] == v2) || (e[1] == v2))
            result = (int) es1[i];
    }
    return result;
}

const Edge EdgeSetTopologyContainer::getEdge (EdgeID i)
{
    if(!hasEdges())
        createEdgeSetArray();

    return (d_edge.getValue())[i];
}


// Return the number of connected components from the graph containing all edges and give, for each vertex, which component it belongs to  (use BOOST GRAPH LIBRAIRY)
int EdgeSetTopologyContainer::getNumberConnectedComponents(sofa::helper::vector<unsigned int>& components)
{
    using namespace boost;

    typedef adjacency_list <vecS, vecS, undirectedS> Graph;

    Graph G;
    helper::ReadAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;

    for (unsigned int k=0; k<m_edge.size(); ++k)
    {
        add_edge(m_edge[k][0], m_edge[k][1], G);
    }

    components.resize(num_vertices(G));
    int num = (int) connected_components(G, &components[0]);

    return num;
}

bool EdgeSetTopologyContainer::checkTopology() const
{
#ifndef NDEBUG
    bool ret = true;

    if(hasEdgesAroundVertex())
    {
        helper::ReadAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;
        std::set<int> edgeSet;
        std::set<int>::iterator it;

        for (unsigned int i=0; i<m_edgesAroundVertex.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &es = m_edgesAroundVertex[i];

            for (unsigned int j=0; j<es.size(); ++j)
            {
                bool check_edge_vertex_shell = (m_edge[ es[j] ][0] == i) ||  (m_edge[ es[j] ][1] == i);
                if(! check_edge_vertex_shell)
                {
                    std::cout << "*** CHECK FAILED : check_edge_vertex_shell, i = " << i << " , j = " << j << std::endl;
                    ret = false;
                }

                it=edgeSet.find(es[j]);
                if (it == edgeSet.end())
                {
                    edgeSet.insert (es[j]);
                }
            }
        }

        if (edgeSet.size() != m_edge.size())
        {
            std::cout << "*** CHECK FAILED : check_edge_vertex_shell, edge are missing in m_edgesAroundVertex" << std::endl;
            ret = false;
        }
    }

    return ret &&  PointSetTopologyContainer::checkTopology();
#else
    return true;
#endif
}


bool EdgeSetTopologyContainer::checkConnexity()
{
    if(!hasEdgesAroundVertex())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        sout << "Warning. [EdgeSetTopologyContainer::checkConnexity] EdgesAroundVertex shell array is empty." << endl;
#endif
        createEdgesAroundVertexArray();
    }

    bool end = false;
    int cpt = 0;

    sofa::helper::vector <EdgeID> edge_T0;
    sofa::helper::vector <EdgeID> edge_T1;
    sofa::helper::vector <EdgeID> edge_T2;
    sofa::helper::vector <EdgeID> edge_Tmp;

    edge_T1.push_back(0);
    cpt = 1;

    while (!end && cpt < this->getNbEdges())
    {
        // First Step - Create new region
        for (unsigned int i = 0; i<edge_T1.size(); ++i)
        {
            EdgeID edgeIndex = edge_T1[i];
            Edge edge = this->getEdge(edgeIndex);

            for (unsigned int j = 0; j<2; ++j)
            {
                sofa::helper::vector<unsigned int> edgeAVertex = m_edgesAroundVertex[ edge[j] ];
                sofa::helper::vector<EdgeID> nextEdges;

                if (edgeAVertex.size() == 1) // reach border
                    continue;
                else
                {
                    for (unsigned int k = 0; k<edgeAVertex.size(); ++k)
                    {
                        if (edgeAVertex[k] != edgeIndex) //not himself
                            nextEdges.push_back(edgeAVertex[k]);
                    }
                }

                // avoid redundancy
                for (unsigned int k = 0; k<nextEdges.size(); ++k)
                {
                    bool edgeFound = false;
                    EdgeID elem = nextEdges[k];

                    for (unsigned int l = 0; l<edge_Tmp.size(); ++l)
                        if ( elem == edge_Tmp[l])
                        {
                            edgeFound = true;
                            break;
                        }

                    if (!edgeFound)
                        edge_Tmp.push_back (elem);
                }
            }
        }

        // Second Step - Avoid backward direction
        for (unsigned int i = 0; i<edge_Tmp.size(); ++i)
        {
            bool edgeFound = false;
            EdgeID elem = edge_Tmp[i];

            for (unsigned int j = 0; j<edge_T0.size(); ++j)
                if (edge_T0[j] == elem)
                {
                    edgeFound = true;
                    break;
                }

            if (!edgeFound)
            {
                for (unsigned int j = 0; j<edge_T1.size(); ++j)
                    if (edge_T1[j] == elem)
                    {
                        edgeFound = true;
                        break;
                    }
            }

            if (!edgeFound)
                edge_T2.push_back(elem);
        }

        // cpt for connexity
        cpt +=edge_T2.size();

        if (edge_T2.size() == 0) // reach end
        {
            end = true;
#ifndef NDEBUG
            sout << "Loop for computing connexity has reach end." << sendl;
#endif
        }

        // iterate
        edge_T0 = edge_T1;
        edge_T1 = edge_T2;
        edge_T2.clear();
        edge_Tmp.clear();
    }

    if (cpt != this->getNbEdges())
    {
        serr << "Warning: in computing connexity, edges are missings. There is more than one connexe component." << sendl;
        return false;
    }

    return true;
}


unsigned int EdgeSetTopologyContainer::getNumberOfEdges() const
{
    d_edge.updateIfDirty();
    return (d_edge.getValue()).size();
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &EdgeSetTopologyContainer::getEdgesAroundVertexArray()
{
    if(!hasEdgesAroundVertex())
    {
#ifndef NDEBUG
        sout << "Warning. [EdgeSetTopologyContainer::getEdgesAroundVertexArray] edge vertex shell array is empty." << endl;
#endif
        createEdgesAroundVertexArray();
    }

    return m_edgesAroundVertex;
}

const EdgesAroundVertex& EdgeSetTopologyContainer::getEdgesAroundVertex(PointID i)
{
    if(!hasEdgesAroundVertex())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        sout << "Warning. [EdgeSetTopologyContainer::getEdgesAroundVertex] edge vertex shell array is empty." << endl;
#endif
        createEdgesAroundVertexArray();
    }

#ifndef NDEBUG
    if(m_edgesAroundVertex.size() <= i)
        sout << "Error. [EdgeSetTopologyContainer::getEdgesAroundVertex] edge vertex shell array out of bounds: "
                << i << " >= " << m_edgesAroundVertex.size() << endl;
#endif

    return m_edgesAroundVertex[i];
}

sofa::helper::vector< unsigned int > &EdgeSetTopologyContainer::getEdgesAroundVertexForModification(const unsigned int i)
{
    if(!hasEdgesAroundVertex())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        sout << "Warning. [EdgeSetTopologyContainer::getEdgesAroundVertexForModification] edge vertex shell array is empty." << endl;
#endif
        createEdgesAroundVertexArray();
    }

    return m_edgesAroundVertex[i];
}



bool EdgeSetTopologyContainer::hasEdges() const
{
    d_edge.updateIfDirty();
    return !(d_edge.getValue()).empty();
}

bool EdgeSetTopologyContainer::hasEdgesAroundVertex() const
{
    return !m_edgesAroundVertex.empty();
}

void EdgeSetTopologyContainer::clearEdges()
{
    helper::WriteAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;
    m_edge.clear();
}

void EdgeSetTopologyContainer::clearEdgesAroundVertex()
{
    m_edgesAroundVertex.clear();
}

void EdgeSetTopologyContainer::clear()
{
    clearEdges();
    clearEdgesAroundVertex();

    PointSetTopologyContainer::clear();
}


} // namespace topology

} // namespace component

} // namespace sofa

