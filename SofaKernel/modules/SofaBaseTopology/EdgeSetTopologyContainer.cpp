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
		if(CHECK_TOPOLOGY)
			msg_warning() << "Edge array is empty.";
		
        createEdgeSetArray();
    }

    if(hasEdgesAroundVertex())
    {
        clearEdgesAroundVertex();
    }

    helper::ReadAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;

    int nbPoints = getNbPoints();
    if (nbPoints == 0) // in case only Data have been copied and not going thourgh AddTriangle methods.
        this->setNbPoints(d_initPoints.getValue().size());

    m_edgesAroundVertex.resize(getNbPoints());

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
	if(CHECK_TOPOLOGY)
        msg_error() << "createEdgeSetArray method must be implemented by a child topology.";

}

const sofa::helper::vector<EdgeSetTopologyContainer::Edge> &EdgeSetTopologyContainer::getEdgeArray()
{
    if(!hasEdges() && getNbPoints()>0)
    {
		if(CHECK_TOPOLOGY)
			msg_info() << "Creating edge array.";

        createEdgeSetArray();
    }

    return d_edge.getValue();
}

EdgeSetTopologyContainer::EdgeID EdgeSetTopologyContainer::getEdgeIndex(PointID v1, PointID v2)
{
    if(!hasEdges()) // this method should only be called when edges exist
    {
		if(CHECK_TOPOLOGY)
			msg_warning() << "Edge array is empty.";

        createEdgeSetArray();
    }

    if(!hasEdgesAroundVertex())
        createEdgesAroundVertexArray();

    const sofa::helper::vector< EdgeID > &es1 = getEdgesAroundVertex(v1) ;
    helper::ReadAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;

    EdgeID result = InvalidID;
    for(size_t i=0; (i < es1.size()) && (result == InvalidID); ++i)
    {
        const Edge &e = m_edge[ es1[i] ];
        if ((e[0] == v2) || (e[1] == v2))
            result = es1[i];
    }

    if (result == InvalidID)
        msg_warning() << "Edge with indices: [" << v1 << "; " << v2 << "] not found.";

    return result;
}

const EdgeSetTopologyContainer::Edge EdgeSetTopologyContainer::getEdge (EdgeID i)
{
    if(!hasEdges())
        createEdgeSetArray();

    if ((size_t)i >= getNbEdges())
        return Edge(InvalidID, InvalidID);
    else
        return (d_edge.getValue())[i];
}


// Return the number of connected components from the graph containing all edges and give, for each vertex, which component it belongs to  (use BOOST GRAPH LIBRAIRY)
int EdgeSetTopologyContainer::getNumberConnectedComponents(sofa::helper::vector<EdgeID>& components)
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
	if (CHECK_TOPOLOGY)
	{
		bool ret = true;

		if (hasEdgesAroundVertex())
		{
			helper::ReadAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;
			std::set<int> edgeSet;
			std::set<int>::iterator it;

			for (size_t i = 0; i < m_edgesAroundVertex.size(); ++i)
			{
				const sofa::helper::vector<EdgeID> &es = m_edgesAroundVertex[i];

				for (size_t j = 0; j < es.size(); ++j)
				{
					bool check_edge_vertex_shell = (m_edge[es[j]][0] == i) || (m_edge[es[j]][1] == i);
					if (!check_edge_vertex_shell)
					{
						msg_warning() << "*** CHECK FAILED : check_edge_vertex_shell, i = " << i << " , j = " << j;
						ret = false;
					}

					it = edgeSet.find(es[j]);
					if (it == edgeSet.end())
					{
						edgeSet.insert(es[j]);
					}
				}
			}

			if (edgeSet.size() != m_edge.size())
			{
				msg_warning() << "*** CHECK FAILED : check_edge_vertex_shell, edge are missing in m_edgesAroundVertex";
				ret = false;
			}
		}

		return ret && PointSetTopologyContainer::checkTopology();
	}

    return true;
}


/// Get information about connexity of the mesh
/// @{
bool EdgeSetTopologyContainer::checkConnexity()
{

    size_t nbr = this->getNbEdges();

    if (nbr == 0)
    {
		if(CHECK_TOPOLOGY)
			msg_warning() << "Can't compute connexity as there are no edges";

        return false;
    }

    VecEdgeID elemAll = this->getConnectedElement(0);

    if (elemAll.size() != nbr)
    {
		msg_warning() << "Warning: in computing connexity, edges are missings. There is more than one connexe component.";
        return false;
    }

    return true;
}


size_t EdgeSetTopologyContainer::getNumberOfConnectedComponent()
{
    size_t nbr = this->getNbEdges();

    if (nbr == 0)
    {
		if(CHECK_TOPOLOGY)
			msg_warning() << "Can't compute connexity as there are no edges";

        return 0;
    }

    VecEdgeID elemAll = this->getConnectedElement(0);
    size_t cpt = 1;

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
    if(!hasEdgesAroundVertex())	// this method should only be called when the shell array exists
    {
		if(CHECK_TOPOLOGY)
			msg_warning() << "EdgesAroundVertex shell array is empty.";

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
			if(CHECK_TOPOLOGY)
				msg_error() << "Loop for computing connexity has reach end.";

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
		if(CHECK_TOPOLOGY)
			msg_warning() << "Edge vertex shell array is empty.";

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
		if(CHECK_TOPOLOGY)
			msg_warning() << "Edge vertex shell array is empty.";

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




size_t EdgeSetTopologyContainer::getNumberOfEdges() const
{
    d_edge.updateIfDirty();
    return d_edge.getValue().size();
}

size_t EdgeSetTopologyContainer::getNumberOfElements() const
{
    return this->getNumberOfEdges();
}

const sofa::helper::vector< sofa::helper::vector<EdgeSetTopologyContainer::EdgeID> > &EdgeSetTopologyContainer::getEdgesAroundVertexArray()
{
    if(!hasEdgesAroundVertex())
    {
		if(CHECK_TOPOLOGY)
			msg_warning() << "Edge vertex shell array is empty.";

        createEdgesAroundVertexArray();
    }

    return m_edgesAroundVertex;
}

const EdgeSetTopologyContainer::EdgesAroundVertex& EdgeSetTopologyContainer::getEdgesAroundVertex(PointID i)
{
    if(!hasEdgesAroundVertex())	// this method should only be called when the shell array exists
    {
		if(CHECK_TOPOLOGY)
			msg_warning() << "Edge vertex shell array is empty.";

        createEdgesAroundVertexArray();
    }

		if(CHECK_TOPOLOGY)
			if(m_edgesAroundVertex.size() <= i)
				msg_error() << "Edge vertex shell array out of bounds: "
						<< i << " >= " << m_edgesAroundVertex.size();


    return m_edgesAroundVertex[i];
}

sofa::helper::vector< EdgeSetTopologyContainer::EdgeID > &EdgeSetTopologyContainer::getEdgesAroundVertexForModification(const PointID i)
{
    if(!hasEdgesAroundVertex())	// this method should only be called when the shell array exists
    {
		if(CHECK_TOPOLOGY)
			msg_warning() << "Edge vertex shell array is empty.";

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

void EdgeSetTopologyContainer::setEdgeTopologyToDirty()
{
    // set this container to dirty
    m_edgeTopologyDirty = true;

    std::list<sofa::core::topology::TopologyEngine *>::iterator it;
    for (it = m_enginesList.begin(); it!=m_enginesList.end(); ++it)
    {
        sofa::core::topology::TopologyEngine* topoEngine = (*it);
        topoEngine->setDirtyValue();
        if (CHECK_TOPOLOGY)
            msg_info() << "Edge Topology Set dirty engine: " << topoEngine->name;
    }
}

void EdgeSetTopologyContainer::cleanEdgeTopologyFromDirty()
{
    m_edgeTopologyDirty = false;

    // security, clean all engines to avoid loops
    std::list<sofa::core::topology::TopologyEngine *>::iterator it;
    for ( it = m_enginesList.begin(); it!=m_enginesList.end(); ++it)
    {
        if ((*it)->isDirty())
        {
            if (CHECK_TOPOLOGY)
                msg_warning() << "Edge Topology update did not clean engine: " << (*it)->name;
            (*it)->cleanDirty();
        }
    }
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

