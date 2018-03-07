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
#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
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
SOFA_DECL_CLASS(EdgeSetTopologyContainer)
int EdgeSetTopologyContainerClass = core::RegisterObject("Edge set topology container")
        .add< EdgeSetTopologyContainer >()
        ;

EdgeSetTopologyContainer::EdgeSetTopologyContainer()
    : PointSetTopologyContainer( )
    , d_edge(initData(&d_edge, "edges", "List of edge indices"))
    , m_checkConnexity(initData(&m_checkConnexity, false, "checkConnexity", "It true, will check the connexity of the mesh."))
{
}


void EdgeSetTopologyContainer::init()
{
    d_edge.updateIfDirty(); // make sure m_edge is up to date

    helper::ReadAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;
    if (!m_edge.empty() && !d_initPoints.isSet()) // if d_initPoints is set, we don't overwrite it.
    {
        for (size_t i=0; i<m_edge.size(); ++i)
        {
            for(size_t j=0; j<2; ++j)
            {
                int a = m_edge[i][j];
                if (a >= getNbPoints()) setNbPoints(a+1);
            }
        }
    }

    PointSetTopologyContainer::init();
}

void EdgeSetTopologyContainer::addEdge(int a, int b)
{
    helper::WriteAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;
    m_edge.push_back(Edge(a,b));
    if (a >= getNbPoints()) setNbPoints(a+1);
    if (b >= getNbPoints()) setNbPoints(b+1);
}

void EdgeSetTopologyContainer::createEdgesAroundVertexArray()
{
    if(!hasEdges())	// this method should only be called when edges exist
    {
#ifndef NDEBUG
        sout << "Warning. [EdgeSetTopologyContainer::createEdgesAroundVertexArray] edge array is empty." << sendl;
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

    if (m_checkConnexity.getValue())
        this->checkConnexity();
}

void EdgeSetTopologyContainer::reinit()
{
    PointSetTopologyContainer::reinit();

    if (m_checkConnexity.getValue())
        this->checkConnexity();
}

void EdgeSetTopologyContainer::createEdgeSetArray()
{
#ifndef NDEBUG
    sout << "Error. [EdgeSetTopologyContainer::createEdgeSetArray] This method must be implemented by a child topology." << sendl;
#endif
}

const sofa::helper::vector<EdgeSetTopologyContainer::Edge> &EdgeSetTopologyContainer::getEdgeArray()
{
    if(!hasEdges() && getNbPoints()>0)
    {
#ifndef NDEBUG
        sout << "Warning. [EdgeSetTopologyContainer::getEdgeArray] creating edge array." << sendl;
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
        sout << "Warning. [EdgeSetTopologyContainer::getEdgeIndex] edge array is empty." << sendl;
#endif
        createEdgeSetArray();
    }

    if(!hasEdgesAroundVertex())
        createEdgesAroundVertexArray();

    const sofa::helper::vector< unsigned int > &es1 = getEdgesAroundVertex(v1) ;
    helper::ReadAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;

    int result = -1;
    for(size_t i=0; (i < es1.size()) && (result == -1); ++i)
    {
        const Edge &e = m_edge[ es1[i] ];
        if ((e[0] == v2) || (e[1] == v2))
            result = (int) es1[i];
    }
    return result;
}

const EdgeSetTopologyContainer::Edge EdgeSetTopologyContainer::getEdge (EdgeID i)
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

    for (size_t k=0; k<m_edge.size(); ++k)
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

        for (size_t i=0; i<m_edgesAroundVertex.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &es = m_edgesAroundVertex[i];

            for (size_t j=0; j<es.size(); ++j)
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


/// Get information about connexity of the mesh
/// @{
bool EdgeSetTopologyContainer::checkConnexity()
{

    size_t nbr = this->getNbEdges();

    if (nbr == 0)
    {
#ifndef NDEBUG
        serr << "Warning. [EdgeSetTopologyContainer::checkConnexity] Can't compute connexity as there are no edges" << sendl;
#endif
        return false;
    }

    VecEdgeID elemAll = this->getConnectedElement(0);

    if (elemAll.size() != nbr)
    {
        serr << "Warning: in computing connexity, edges are missings. There is more than one connexe component." << sendl;
        return false;
    }

    return true;
}


unsigned int EdgeSetTopologyContainer::getNumberOfConnectedComponent()
{
    size_t nbr = this->getNbEdges();

    if (nbr == 0)
    {
#ifndef NDEBUG
        serr << "Warning. [EdgeSetTopologyContainer::getNumberOfConnectedComponent] Can't compute connexity as there are no edges" << sendl;
#endif
        return 0;
    }

    VecEdgeID elemAll = this->getConnectedElement(0);
    unsigned int cpt = 1;

    while (elemAll.size() < nbr)
    {
        std::sort(elemAll.begin(), elemAll.end());
        EdgeID other_edgeID = elemAll.size();

        for (EdgeID i = 0; i<(EdgeID)elemAll.size(); ++i)
            if (elemAll[i] != i)
            {
                other_edgeID = i;
                break;
            }

        VecEdgeID elemTmp = this->getConnectedElement(other_edgeID);
        cpt++;

        elemAll.insert(elemAll.begin(), elemTmp.begin(), elemTmp.end());
    }

    return cpt;
}


const EdgeSetTopologyContainer::VecEdgeID EdgeSetTopologyContainer::getConnectedElement(EdgeID elem)
{
    if(!hasEdgesAroundVertex())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        serr << "Warning. [EdgeSetTopologyContainer::getConnectedElement] EdgesAroundVertex shell array is empty." << sendl;
#endif
        createEdgesAroundVertexArray();
    }

    VecEdgeID elemAll;
    VecEdgeID elemOnFront, elemPreviousFront, elemNextFront;
    bool end = false;
    size_t cpt = 0;
    size_t nbr = this->getNbEdges();

    // init algo
    elemAll.push_back(elem);
    elemOnFront.push_back(elem);
    elemPreviousFront.clear();
    cpt++;

    while (!end && cpt < nbr)
    {
        // First Step - Create new region
        elemNextFront = this->getElementAroundElements(elemOnFront); // for each edgeId on the propagation front

        // Second Step - Avoid backward direction
        for (size_t i = 0; i<elemNextFront.size(); ++i)
        {
            bool find = false;
            EdgeID id = elemNextFront[i];

            for (size_t j = 0; j<elemAll.size(); ++j)
                if (id == elemAll[j])
                {
                    find = true;
                    break;
                }

            if (!find)
            {
                elemAll.push_back(id);
                elemPreviousFront.push_back(id);
            }
        }

        // cpt for connexity
        cpt +=elemPreviousFront.size();

        if (elemPreviousFront.empty())
        {
            end = true;
#ifndef NDEBUG
            serr << "Loop for computing connexity has reach end." << sendl;
#endif
        }

        // iterate
        elemOnFront = elemPreviousFront;
        elemPreviousFront.clear();
    }

    return elemAll;
}


const EdgeSetTopologyContainer::VecEdgeID EdgeSetTopologyContainer::getElementAroundElement(EdgeID elem)
{
    VecEdgeID elems;

    if (!hasEdgesAroundVertex())
    {
#ifndef NDEBUG
        serr << "Warning. [EdgeSetTopologyContainer::getElementAroundElement] edge vertex shell array is empty." << sendl;
#endif
        createEdgesAroundVertexArray();
    }

    Edge the_edge = this->getEdge(elem);

    for(size_t i = 0; i<2; ++i) // for each node of the edge
    {
        EdgesAroundVertex edgeAV = this->getEdgesAroundVertex(the_edge[i]);

        for (size_t j = 0; j<edgeAV.size(); ++j) // for each edge around the node
        {
            bool find = false;
            EdgeID id = edgeAV[j];

            if (id == elem)
                continue;

            for (size_t k = 0; k<elems.size(); ++k) // check no redundancy
                if (id == elems[k])
                {
                    find = true;
                    break;
                }

            if (!find)
                elems.push_back(id);
        }
    }

    return elems;
}


const EdgeSetTopologyContainer::VecEdgeID EdgeSetTopologyContainer::getElementAroundElements(VecEdgeID elems)
{
    VecEdgeID elemAll;
    VecEdgeID elemTmp;

    if (!hasEdgesAroundVertex())
    {
#ifndef NDEBUG
        serr << "Warning. [EdgeSetTopologyContainer::getElementAroundElements] edge vertex shell array is empty." << sendl;
#endif
        createEdgesAroundVertexArray();
    }

    for (size_t i = 0; i <elems.size(); ++i) // for each edgeId of input vector
    {
        VecEdgeID elemTmp2 = this->getElementAroundElement(elems[i]);

        elemTmp.insert(elemTmp.end(), elemTmp2.begin(), elemTmp2.end());
    }

    for (size_t i = 0; i<elemTmp.size(); ++i) // for each edge Id found
    {
        bool find = false;
        EdgeID id = elemTmp[i];

        for (size_t j = 0; j<elems.size(); ++j) // check no redundancy with input vector
            if (id == elems[j])
            {
                find = true;
                break;
            }

        if (!find)
        {
            for (size_t j = 0; j<elemAll.size(); ++j) // check no redundancy in output vector
                if (id == elemAll[j])
                {
                    find = true;
                    break;
                }
        }

        if (!find)
            elemAll.push_back(id);
    }


    return elemAll;
}

/// @}




unsigned int EdgeSetTopologyContainer::getNumberOfEdges() const
{
    d_edge.updateIfDirty();
    return (unsigned int)d_edge.getValue().size();
}

unsigned int EdgeSetTopologyContainer::getNumberOfElements() const
{
    return this->getNumberOfEdges();
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &EdgeSetTopologyContainer::getEdgesAroundVertexArray()
{
    if(!hasEdgesAroundVertex())
    {
#ifndef NDEBUG
        sout << "Warning. [EdgeSetTopologyContainer::getEdgesAroundVertexArray] edge vertex shell array is empty." << sendl;
#endif
        createEdgesAroundVertexArray();
    }

    return m_edgesAroundVertex;
}

const EdgeSetTopologyContainer::EdgesAroundVertex& EdgeSetTopologyContainer::getEdgesAroundVertex(PointID i)
{
    if(!hasEdgesAroundVertex())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        sout << "Warning. [EdgeSetTopologyContainer::getEdgesAroundVertex] edge vertex shell array is empty." << sendl;
#endif
        createEdgesAroundVertexArray();
    }

#ifndef NDEBUG
    if(m_edgesAroundVertex.size() <= i)
        sout << "Error. [EdgeSetTopologyContainer::getEdgesAroundVertex] edge vertex shell array out of bounds: "
                << i << " >= " << m_edgesAroundVertex.size() << sendl;
#endif

    return m_edgesAroundVertex[i];
}

sofa::helper::vector< unsigned int > &EdgeSetTopologyContainer::getEdgesAroundVertexForModification(const unsigned int i)
{
    if(!hasEdgesAroundVertex())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        sout << "Warning. [EdgeSetTopologyContainer::getEdgesAroundVertexForModification] edge vertex shell array is empty." << sendl;
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
    // Do not set to 0 the number of points as it prevents the  creation of topological items (edgeArray in tetrahedra for instance)
//    PointSetTopologyContainer::clear();
}



void EdgeSetTopologyContainer::updateTopologyEngineGraph()
{
    this->updateDataEngineGraph(this->d_edge, this->m_enginesList);

    // will concatenate with points one:
    PointSetTopologyContainer::updateTopologyEngineGraph();
}


} // namespace topology

} // namespace component

} // namespace sofa

