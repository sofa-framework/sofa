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
#include <sofa/component/topology/container/dynamic/EdgeSetTopologyContainer.h>
#include <sofa/core/topology/TopologyHandler.h>

#include <sofa/core/ObjectFactory.h>

#include <vector>
#include <list>
#include <set>
#include <utility>

// Use BOOST GRAPH LIBRARY :
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

namespace sofa::component::topology::container::dynamic
{

using namespace std;
using namespace sofa::defaulttype;
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
    const helper::ReadAccessor< Data< sofa::type::vector<Edge> > > m_edge = d_edge;
    
    if (d_initPoints.isSet())
    {
        setNbPoints(Size(d_initPoints.getValue().size()));
    }
    else if (!m_edge.empty())
    {
        for (size_t i=0; i<m_edge.size(); ++i)
        {
            for(sofa::Index j=0; j<2; ++j)
            {
                const EdgeID a = m_edge[i][j];
                if (a >= getNbPoints()) setNbPoints(a+1);
            }
        }
    }

    PointSetTopologyContainer::init();

    // only init if edges are present at init.
    if (!m_edge.empty())
        initTopology();
}

void EdgeSetTopologyContainer::initTopology()
{
    // force computation of neighborhood elements
    createEdgesAroundVertexArray();
}

void EdgeSetTopologyContainer::addEdge(Index a, Index b)
{
    helper::WriteAccessor< Data< sofa::type::vector<Edge> > > m_edge = d_edge;
    m_edge.push_back(Edge(a,b));
    if (a >= getNbPoints()) setNbPoints(a+1);
    if (b >= getNbPoints()) setNbPoints(b+1);
}

void EdgeSetTopologyContainer::createEdgesAroundVertexArray()
{
    // first clear potential previous buffer
    clearEdgesAroundVertex();

    if(!hasEdges())
        createEdgeSetArray();

    if (hasEdgesAroundVertex()) // created by upper topology
        return;

    const helper::ReadAccessor< Data< sofa::type::vector<Edge> > > edges = d_edge;

    if (edges.empty())
    {
        msg_warning() << "EdgesAroundVertex buffer can't be created as no edges are present in this topology.";
        return;
    }

    const std::size_t nbPoints = getNbPoints();
    if (nbPoints == 0) // in case only Data have been copied and not going thourgh AddTriangle methods.
        this->setNbPoints(sofa::Size(d_initPoints.getValue().size()));

    m_edgesAroundVertex.resize(getNbPoints());
    for (unsigned int edgeId=0; edgeId<edges.size(); ++edgeId)
    {
        const Edge& edge = edges[edgeId];
        // adding edge in the edge shell of both points
        
        if (edge[0] >= unsigned(nbPoints) || edge[1] >= unsigned(nbPoints))
        {
            msg_warning() << "EdgesAroundVertex creation failed, Edge buffer is not consistent with number of points, Edge: " << edge << " for: " << nbPoints << " points.";
            continue;
        }

        m_edgesAroundVertex[ edge[0] ].push_back(edgeId);
        m_edgesAroundVertex[ edge[1] ].push_back(edgeId);
    }

    if (m_checkConnexity.getValue())
        this->checkConnexity();
}

void EdgeSetTopologyContainer::reinit()
{
    PointSetTopologyContainer::reinit();

    createEdgesAroundVertexArray();

    if (m_checkConnexity.getValue())
        this->checkConnexity();
}

void EdgeSetTopologyContainer::createEdgeSetArray()
{
    msg_error() << "createEdgeSetArray method must be implemented by an higher level topology.";
}


const sofa::type::vector<EdgeSetTopologyContainer::Edge> &EdgeSetTopologyContainer::getEdges()
{
    return getEdgeArray();
}

const sofa::type::vector<EdgeSetTopologyContainer::Edge> &EdgeSetTopologyContainer::getEdgeArray()
{
    return d_edge.getValue();
}

EdgeSetTopologyContainer::EdgeID EdgeSetTopologyContainer::getEdgeIndex(PointID v1, PointID v2)
{
    if (!hasEdges())
    {
        return InvalidID;
    }

    const sofa::type::vector< EdgeID >& es1 = getEdgesAroundVertex(v1);
    if (es1.empty())
        return InvalidID;

    const helper::ReadAccessor< Data< sofa::type::vector<Edge> > > m_edge = d_edge;

    EdgeID result = InvalidID;
    for(size_t i=0; (i < es1.size()) && (result == InvalidID); ++i)
    {
        const Edge &e = m_edge[ es1[i] ];
        if ((e[0] == v2) || (e[1] == v2))
            result = es1[i];
    }

    return result;
}

const EdgeSetTopologyContainer::Edge EdgeSetTopologyContainer::getEdge (EdgeID i)
{
    if ((size_t)i >= getNbEdges())
        return InvalidEdge;
    else
        return (d_edge.getValue())[i];
}


// Return the number of connected components from the graph containing all edges and give, for each vertex, which component it belongs to  (use BOOST GRAPH LIBRAIRY)
int EdgeSetTopologyContainer::getNumberConnectedComponents(sofa::type::vector<EdgeID>& components)
{
    using namespace boost;

    typedef adjacency_list <vecS, vecS, undirectedS> Graph;

    Graph G;
    const helper::ReadAccessor< Data< sofa::type::vector<Edge> > > m_edge = d_edge;

    for (size_t k=0; k<m_edge.size(); ++k)
    {
        add_edge(m_edge[k][0], m_edge[k][1], G);
    }

    components.resize(num_vertices(G));
    const int num = (int) connected_components(G, &components[0]);

    return num;
}

bool EdgeSetTopologyContainer::checkTopology() const
{
    if (!d_checkTopology.getValue())
        return true;

    bool ret = true;

    if (hasEdgesAroundVertex())
    {
        const helper::ReadAccessor< Data< sofa::type::vector<Edge> > > m_edge = d_edge;
        std::set<int> edgeSet;

        // loop on all edges around vertex
        for (size_t i = 0; i < m_edgesAroundVertex.size(); ++i)
        {
            const sofa::type::vector<EdgeID> &es = m_edgesAroundVertex[i];
            for (size_t j = 0; j < es.size(); ++j)
            {
                const Edge& edge = m_edge[es[j]];
                if (!(edge[0] == i || edge[1] == i))
                {
                    msg_error() << "EdgeSetTopologyContainer::checkTopology() failed: edge " << es[j] << ": [" << edge << "] not around vertex: " << i;
                    ret = false;
                }

                // count number of edge
                edgeSet.insert(es[j]);
            }
        }

        if (edgeSet.size() != m_edge.size())
        {
            msg_error() << "EdgeSetTopologyContainer::checkTopology() failed: found " << edgeSet.size() << " edges in m_edgesAroundVertex out of " << m_edge.size();
            ret = false;
        }
    }

    return ret && PointSetTopologyContainer::checkTopology();
}


/// Get information about connexity of the mesh
/// @{
bool EdgeSetTopologyContainer::checkConnexity()
{
    const size_t nbr = this->getNbEdges();

    if (nbr == 0)
    {
        msg_error() << "CheckConnexity: Can't compute connexity as there are no edges";
        return false;
    }

    const VecEdgeID elemAll = this->getConnectedElement(0);

    if (elemAll.size() != nbr)
    {
		msg_warning() << "CheckConnexity: Edges are missings. There is more than one connexe component.";
        return false;
    }

    return true;
}


Size EdgeSetTopologyContainer::getNumberOfConnectedComponent()
{
    const auto nbr = this->getNbEdges();

    if (nbr == 0)
    {
        msg_error() << "Can't getNumberOfConnectedComponent as there are no edges";
        return 0;
    }

    VecEdgeID elemAll = this->getConnectedElement(0);
    Size cpt = 1;

    while (elemAll.size() < nbr)
    {
        std::sort(elemAll.begin(), elemAll.end());
        size_t other_edgeID = elemAll.size();

        for (EdgeID i = 0; i<(EdgeID)elemAll.size(); ++i)
            if (elemAll[i] != i)
            {
                other_edgeID = i;
                break;
            }

        VecEdgeID elemTmp = this->getConnectedElement((EdgeID)other_edgeID);
        cpt++;

        elemAll.insert(elemAll.begin(), elemTmp.begin(), elemTmp.end());
    }

    return cpt;
}


const EdgeSetTopologyContainer::VecEdgeID EdgeSetTopologyContainer::getConnectedElement(EdgeID elem)
{
    VecEdgeID elemAll;
    if(!hasEdgesAroundVertex())	// this method should only be called when the shell array exists
    {
        dmsg_warning() << "getConnectedElement: EdgesAroundVertex is empty. Be sure to call createEdgesAroundVertexArray first.";
        return elemAll;
    }
    
    VecEdgeID elemOnFront, elemPreviousFront, elemNextFront;
    bool end = false;
    size_t cpt = 0;
    const size_t nbr = this->getNbEdges();

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
            msg_warning() << "Loop to compute connexity has reach end before processing all elements. This means the mesh is not fully connected.";
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
    	dmsg_warning() << "getElementAroundElement: Edge vertex shell array is empty. Be sure to call createEdgesAroundVertexArray first.";
        return elems;
    }

    Edge the_edge = this->getEdge(elem);

    for(sofa::Index i = 0; i<2; ++i) // for each node of the edge
    {
        const EdgesAroundVertex& edgeAV = this->getEdgesAroundVertex(the_edge[i]);
        if (edgeAV.empty()) {
            msg_error() << "No edge found aroud of vertex id: " << the_edge[i] << ". Should at least found edge id: " << elem;
            continue;
        }

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
    if (!hasEdgesAroundVertex())
    {
        dmsg_warning() << "getElementAroundElements: EdgesAroundVertexArray is empty. Be sure to call createEdgesAroundVertexArray first.";
        return elemAll;
    }

    VecEdgeID elemTmp;
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




Size EdgeSetTopologyContainer::getNumberOfEdges() const
{
    d_edge.updateIfDirty();
    return Size(d_edge.getValue().size());
}

Size EdgeSetTopologyContainer::getNumberOfElements() const
{
    return this->getNumberOfEdges();
}

const sofa::type::vector< sofa::type::vector<EdgeSetTopologyContainer::EdgeID> > &EdgeSetTopologyContainer::getEdgesAroundVertexArray()
{
    return m_edgesAroundVertex;
}

const EdgeSetTopologyContainer::EdgesAroundVertex& EdgeSetTopologyContainer::getEdgesAroundVertex(PointID id)
{
    if(id < m_edgesAroundVertex.size())
        return m_edgesAroundVertex[id];

    return InvalidSet;
}

sofa::type::vector< EdgeSetTopologyContainer::EdgeID > &EdgeSetTopologyContainer::getEdgesAroundVertexForModification(const PointID i)
{
    if(!hasEdgesAroundVertex())	// this method should only be called when the shell array exists
    {
        dmsg_warning() << "getEdgesAroundVertexForModification: EdgesAroundVertexArray is empty. Be sure to call createEdgesAroundVertexArray first.";
        createEdgesAroundVertexArray();
    }

    //TODO epernod (2020-04): this method should be removed as it can create a seg fault.
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
    helper::WriteAccessor< Data< sofa::type::vector<Edge> > > m_edge = d_edge;
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

void EdgeSetTopologyContainer::setEdgeTopologyToDirty()
{
    // set this container to dirty
    m_edgeTopologyDirty = true;

    auto& edgeTopologyHandlerList = getTopologyHandlerList(sofa::geometry::ElementType::EDGE);
    for (const auto topoHandler : edgeTopologyHandlerList)
    {
        topoHandler->setDirtyValue();
        msg_info() << "Edge Topology Set dirty engine: " << topoHandler->getName();
    }

}

void EdgeSetTopologyContainer::cleanEdgeTopologyFromDirty()
{
    m_edgeTopologyDirty = false;

    // security, clean all engines to avoid loops
    auto& edgeTopologyHandlerList = getTopologyHandlerList(sofa::geometry::ElementType::EDGE);
    for (const auto topoHandler : edgeTopologyHandlerList)
    {
        if (topoHandler->isDirty())
        {
            msg_warning() << "Edge Topology update did not clean engine: " << topoHandler->getName();
            topoHandler->cleanDirty();
        }
    }
}

bool EdgeSetTopologyContainer::linkTopologyHandlerToData(core::topology::TopologyHandler* topologyHandler, sofa::geometry::ElementType elementType)
{
    if (elementType == sofa::geometry::ElementType::EDGE)
    {
        d_edge.addOutput(topologyHandler);
        return true;
    }
    else
    {
        return PointSetTopologyContainer::linkTopologyHandlerToData(topologyHandler, elementType);
    }
}

bool EdgeSetTopologyContainer::unlinkTopologyHandlerToData(core::topology::TopologyHandler* topologyHandler, sofa::geometry::ElementType elementType)
{
    if (elementType == sofa::geometry::ElementType::EDGE)
    {
        d_edge.delOutput(topologyHandler);
        return true;
    }
    else
    {
        return PointSetTopologyContainer::unlinkTopologyHandlerToData(topologyHandler, elementType);
    }
}

} //namespace sofa::component::topology::container::dynamic
