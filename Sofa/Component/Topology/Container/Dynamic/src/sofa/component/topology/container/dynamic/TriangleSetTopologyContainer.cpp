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
#include <sofa/component/topology/container/dynamic/TriangleSetTopologyContainer.h>
#include <sofa/core/topology/TopologyHandler.h>

#include <sofa/core/ObjectFactory.h>

namespace sofa::component::topology::container::dynamic
{
using namespace std;
using namespace sofa::defaulttype;


int TriangleSetTopologyContainerClass = core::RegisterObject("Triangle set topology container")
        .add< TriangleSetTopologyContainer >()
        ;

TriangleSetTopologyContainer::TriangleSetTopologyContainer()
    : EdgeSetTopologyContainer()
    , d_triangle(initData(&d_triangle, "triangles", "List of triangle indices"))
{

}


void TriangleSetTopologyContainer::addTriangle(Index a, Index b, Index c )
{
    helper::WriteAccessor< Data< sofa::type::vector<Triangle> > > m_triangle = d_triangle;
    m_triangle.push_back(Triangle(a,b,c));
    if (a >= getNbPoints()) setNbPoints(a+1);
    if (b >= getNbPoints()) setNbPoints(b+1);
    if (c >= getNbPoints()) setNbPoints(c+1);
}

void TriangleSetTopologyContainer::init()
{
    const helper::ReadAccessor< Data< sofa::type::vector<Triangle> > > m_triangle = d_triangle;

    if (d_initPoints.isSet())
    {
        setNbPoints(Size(d_initPoints.getValue().size()));
    }
    else if (!m_triangle.empty())
    {
        // Todo (epernod 2019-03-12): optimise by removing this loop or at least create AroundVertex buffer at the same time.
        for (size_t i=0; i<m_triangle.size(); ++i)
        {
            for(PointID j=0; j<3; ++j)
            {
                const Index a = m_triangle[i][j];
                if (a >= getNbPoints()) setNbPoints(a+1);
            }
        }
    }

    // only init if triangles are present at init.
    if (!m_triangle.empty())
        initTopology();
}

void TriangleSetTopologyContainer::initTopology()
{
    // Force creation of Edge Neighboordhood buffers.
    EdgeSetTopologyContainer::initTopology();

    // Create triangle cross element buffers.
    createEdgesInTriangleArray();
    createTrianglesAroundVertexArray();
    createTrianglesAroundEdgeArray();
}

void TriangleSetTopologyContainer::reinit()
{
    EdgeSetTopologyContainer::reinit();
}


void TriangleSetTopologyContainer::createTriangleSetArray()
{
    msg_error() << "createTriangleSetArray method must be implemented by a child topology.";
}

void TriangleSetTopologyContainer::createTrianglesAroundVertexArray()
{
    // first clear potential previous buffer
    clearTrianglesAroundVertex();

    if(!hasTriangles()) // this method should only be called when triangles exist
        createTriangleSetArray();

    if (hasTrianglesAroundVertex()) // created by upper topology
        return;

    const helper::ReadAccessor< Data< sofa::type::vector<Triangle> > > m_triangle = d_triangle;

    if (m_triangle.empty())
    {
        msg_warning() << "TrianglesAroundVertex buffer can't be created as no triangles are present in this topology.";
        return;
    }

    const int nbPoints = getNbPoints();
    if (nbPoints == 0) // in case only Data have been copied and not going thourgh AddTriangle methods.
        this->setNbPoints(sofa::Size(d_initPoints.getValue().size()));

    m_trianglesAroundVertex.resize(getNbPoints());
    for (size_t i = 0; i < m_triangle.size(); ++i)
    {
        if (m_triangle[i][0] >= getNbPoints() || m_triangle[i][1] >= getNbPoints() || m_triangle[i][2] >= getNbPoints())
        {
            msg_warning() << "trianglesAroundVertex creation failed, Triangle buffer is not consistent with number of points, Triangle: " << m_triangle[i] << " for: " << getNbPoints() << " points.";
            continue;
        }


        // adding edge i in the edge shell of both points
        for (unsigned int j=0; j<3; ++j)
            m_trianglesAroundVertex[ m_triangle[i][j]  ].push_back( (TriangleID)i );
    }
}

void TriangleSetTopologyContainer::createTrianglesAroundEdgeArray ()
{
    // first clear potential previous buffer
    clearTrianglesAroundEdge();

    if(!hasTriangles()) // this method should only be called when triangles exist
        createTriangleSetArray();

    if (hasTrianglesAroundEdge()) // created by upper topology
        return;

    const auto numTriangles = getNumberOfTriangles();
    if (numTriangles == 0)
    {
        msg_warning() << "TrianglesAroundEdge buffer can't be created as no triangles are present in this topology.";
        return;
    }

    if(!hasEdges()) // this method should only be called when edges exist
        createEdgeSetArray();

    const auto numEdges = getNumberOfEdges();
    if (numEdges == 0)
    {
        msg_warning() << "TrianglesAroundEdge buffer can't be created as no edges are present in this topology.";
        return;
    }

    if(!hasEdgesInTriangle())
        createEdgesInTriangleArray();

    if (m_edgesInTriangle.empty())
    {
        msg_warning() << "TrianglesAroundEdge buffer can't be created as EdgesInTriangle buffer creation failed.";
        return;
    }

    m_trianglesAroundEdge.resize( numEdges );
    for (size_t i = 0; i < numTriangles; ++i)
    {
        const Triangle &t = getTriangle((TriangleID)i);
        // adding triangle i in the triangle shell of all edges
        for (unsigned int j=0; j<3; ++j)
        {
            if (d_edge.getValue()[m_edgesInTriangle[i][j]][0] == t[(j + 1) % 3])
                m_trianglesAroundEdge[m_edgesInTriangle[i][j]].insert(m_trianglesAroundEdge[m_edgesInTriangle[i][j]].begin(), (TriangleID)i); // triangle is on the left of the edge
            else
                m_trianglesAroundEdge[m_edgesInTriangle[i][j]].push_back((TriangleID)i); // triangle is on the right of the edge
        }
    }
}

void TriangleSetTopologyContainer::createEdgeSetArray()
{
    if(!hasTriangles()) // this method should only be called when triangles exist
        createTriangleSetArray();

    if(hasEdges())
    {
        // clear edges and all shells that depend on edges
        EdgeSetTopologyContainer::clear();

        if(hasEdgesInTriangle())
            clearEdgesInTriangle();

        if(hasTrianglesAroundEdge())
            clearTrianglesAroundEdge();
    }

    // create a temporary map to find redundant edges
    std::map<Edge, EdgeID> edgeMap;
    helper::WriteAccessor< Data< sofa::type::vector<Edge> > > m_edge = d_edge;
    const helper::ReadAccessor< Data< sofa::type::vector<Triangle> > > m_triangle = d_triangle;

    for (size_t i=0; i<m_triangle.size(); ++i)
    {
        const Triangle &t = m_triangle[i];
        for(unsigned int j=0; j<3; ++j)
        {
            const PointID v1 = t[(j+1)%3];
            const PointID v2 = t[(j+2)%3];

            // sort vertices in lexicographic order
            const Edge e = ((v1<v2) ? Edge(v1,v2) : Edge(v2,v1));

            if(edgeMap.find(e) == edgeMap.end())
            {
                // edge not in edgeMap so create a new one
                const size_t edgeIndex = edgeMap.size();
                edgeMap[e] = (EdgeID)edgeIndex;
                //m_edge.push_back(e); Changed to have oriented edges on the border of the triangulation
                m_edge.push_back(Edge(v1,v2));
            }
        }
    }
}

void TriangleSetTopologyContainer::createEdgesInTriangleArray()
{
    // first clear potential previous buffer
    clearEdgesInTriangle();

    if(!hasTriangles()) // this method should only be called when triangles exist
        createTriangleSetArray();

    if (hasEdgesInTriangle()) // created by upper topology
        return;

    const helper::ReadAccessor< Data< sofa::type::vector<Triangle> > > m_triangle = d_triangle;


    bool foundEdge = true;

    if (hasEdges())
    {
        /// there are already existing edges : must use an inefficient method. Parse all triangles and find the edge that match each triangle edge
        const helper::ReadAccessor< Data< sofa::type::vector<Edge> > > m_edge = d_edge;
        const size_t numTriangles = getNumberOfTriangles();
        const size_t numEdges = getNumberOfEdges();

        m_edgesInTriangle.resize(numTriangles);
        /// create a multi map where the key is a vertex index and the content is the indices of edges adjacent to that vertex.
        std::multimap<PointID, EdgeID> edgesAroundVertexMap;
        std::multimap<PointID, EdgeID>::iterator it;

        for (size_t edge=0; edge<numEdges; ++edge)  //Todo: check if not better using multimap <PointID ,TriangleID> and for each edge, push each triangle present in both shell
        {
            edgesAroundVertexMap.insert(std::pair<PointID, EdgeID> (m_edge[edge][0], (EdgeID)edge));
            edgesAroundVertexMap.insert(std::pair<PointID, EdgeID> (m_edge[edge][1], (EdgeID)edge));
        }
        for ( size_t i = 0 ; (i < numTriangles) && (foundEdge == true) ; ++i )
        {
            const Triangle &t = m_triangle[i];
            // adding edge i in the edge shell of both points
            for ( unsigned int j = 0 ; (j < 3) && (foundEdge == true) ; ++j )
            {
                //finding edge i in edge array
                std::pair<std::multimap<PointID, EdgeID>::iterator, std::multimap<PointID, EdgeID>::iterator > itPair=edgesAroundVertexMap.equal_range(t[(j+1)%3]);

                foundEdge=false;
                for(it=itPair.first; (it!=itPair.second) && (foundEdge==false); ++it)
                {
                    const EdgeID edge = (*it).second;
                    if ( (m_edge[edge][0] == t[(j+1)%3] && m_edge[edge][1] == t[(j+2)%3]) || (m_edge[edge][0] == t[(j+2)%3] && m_edge[edge][1] == t[(j+1)%3]))
                    {
                        m_edgesInTriangle[i][j] = edge;
                        foundEdge=true;
                    }
                }

                if (!foundEdge)
                {
                    msg_error() << "Cannot find edge " << j
                        << " [" << t[(j + 1) % 3] << ", " << t[(j + 2) % 3] << "]"
                        << " in triangle " << i << " [" << t << "]" << " in the provided edge list ("
                        << this->d_edge.getLinkPath() << "). It shows an inconsistency between the edge list ("
                        << this->d_edge.getLinkPath() << ") and the triangle list (" << this->d_triangle.getLinkPath()
                        << "). Either fix the topology (probably in a mesh file), or provide only the triangle list to '"
                        << this->getPathName() << "' and not the edges. In the latter case, the edge list will be "
                        "computed from triangles.";
                    m_edgesInTriangle.clear();
                    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
                    return;
                }
            }
        }
    }
    if(!hasEdges() || foundEdge == false) // To optimize, this method should be called without creating edgesArray before.
    {
        /// create edge array and triangle edge array at the same time
        const size_t numTriangles = getNumberOfTriangles();
        m_edgesInTriangle.resize(numTriangles);


        // create a temporary map to find redundant edges
        std::map<Edge, EdgeID> edgeMap;
        helper::WriteAccessor< Data< sofa::type::vector<Edge> > > m_edge = d_edge;

        for (size_t i=0; i<m_triangle.size(); ++i)
        {
            const Triangle &t = m_triangle[i];
            for(unsigned int j=0; j<3; ++j)
            {
                const PointID v1 = t[(j+1)%3];
                const PointID v2 = t[(j+2)%3];

                // sort vertices in lexicographic order
                const Edge e = ((v1<v2) ? Edge(v1,v2) : Edge(v2,v1));

                if(edgeMap.find(e) == edgeMap.end())
                {
                    // edge not in edgeMap so create a new one
                    const size_t edgeIndex = edgeMap.size();
                    /// add new edge
                    edgeMap[e] = (EdgeID)edgeIndex;
//			  m_edge.push_back(e);
                    m_edge.push_back(Edge(v1,v2));

                }

                m_edgesInTriangle[i][j] = edgeMap[e];
            }
        }
    }

}



void TriangleSetTopologyContainer::createElementsOnBorder()
{

    if(!hasTrianglesAroundEdge())	// Use the trianglesAroundEdgeArray. Should check if it is consistent
    {
        createTrianglesAroundEdgeArray();
    }

    if(!m_trianglesOnBorder.empty())
        m_trianglesOnBorder.clear();

    if(!m_edgesOnBorder.empty())
        m_edgesOnBorder.clear();

    if(!m_pointsOnBorder.empty())
        m_pointsOnBorder.clear();

    const size_t nbrEdges = getNumberOfEdges();
    bool newTriangle = true;
    bool newEdge = true;
    bool newPoint = true;

    const helper::ReadAccessor< Data< sofa::type::vector<Edge> > > m_edge = d_edge;
    for (size_t i = 0; i < nbrEdges; i++)
    {
        if (m_trianglesAroundEdge[i].size() == 1) // I.e this edge is on a border
        {

            // --- Triangle case ---
            for (size_t j = 0; j < m_trianglesOnBorder.size(); j++) // Loop to avoid duplicated indices
            {
                if (m_trianglesOnBorder[j] == m_trianglesAroundEdge[i][0])
                {
                    newTriangle = false;
                    break;
                }
            }

            if(newTriangle) // If index doesn't already exist, add it to the list of triangles On border.
            {
                m_trianglesOnBorder.push_back (m_trianglesAroundEdge[i][0]);
            }


            // --- Edge case ---
            for (size_t j = 0; j < m_edgesOnBorder.size(); j++) // Loop to avoid duplicated indices
            {
                if (m_edgesOnBorder[j] == i)
                {
                    newEdge = false;
                    break;
                }
            }

            if(newEdge) // If index doesn't already exist, add it to the list of edges On border.
            {
                m_edgesOnBorder.push_back ((EdgeID)i);
            }


            // --- Point case ---
            PointID firstVertex = m_edge[i][0];
            for (size_t j = 0; j < m_pointsOnBorder.size(); j++) // Loop to avoid duplicated indices
            {
                if (m_pointsOnBorder[j] == firstVertex)
                {
                    newPoint = false;
                    break;
                }
            }

            if(newPoint) // If index doesn't already exist, add it to the list of points On border.
            {
                m_pointsOnBorder.push_back (firstVertex);
            }


            newTriangle = true; //reinitialize tests variables
            newEdge = true;
            newPoint = true;
        }
    }
}


void TriangleSetTopologyContainer::reOrientateTriangle(TriangleID id)
{
    if (id >= (TriangleID)this->getNbTriangles())
    {
        msg_warning() << "Triangle ID out of bounds.";
        return;
    }

    Triangle& tri = (*d_triangle.beginEdit())[id];
    const PointID tmp = tri[1];
    tri[1] = tri[2];
    tri[2] = tmp;
    d_triangle.endEdit();

    return;
}


const sofa::type::vector<TriangleSetTopologyContainer::Triangle> & TriangleSetTopologyContainer::getTriangleArray()
{
    return d_triangle.getValue();
}


const TriangleSetTopologyContainer::Triangle TriangleSetTopologyContainer::getTriangle (TriangleID i)
{
    if ((size_t)i >= getNbTriangles())
        return Triangle(InvalidID, InvalidID, InvalidID);
    else
        return (d_triangle.getValue())[i];
}



TriangleSetTopologyContainer::TriangleID TriangleSetTopologyContainer::getTriangleIndex(PointID v1, PointID v2, PointID v3)
{
    if(!hasTrianglesAroundVertex())
    {
        return InvalidID;
    }

    sofa::type::vector<TriangleID> set1 = getTrianglesAroundVertex(v1);
    sofa::type::vector<TriangleID> set2 = getTrianglesAroundVertex(v2);
    sofa::type::vector<TriangleID> set3 = getTrianglesAroundVertex(v3);

    sort(set1.begin(), set1.end());
    sort(set2.begin(), set2.end());
    sort(set3.begin(), set3.end());

    // The destination vector must be large enough to contain the result.
    sofa::type::vector<TriangleID> out1(set1.size()+set2.size());
    sofa::type::vector<TriangleID>::iterator result1;
    result1 = std::set_intersection(set1.begin(),set1.end(),set2.begin(),set2.end(),out1.begin());
    out1.erase(result1,out1.end());
    sofa::type::vector<TriangleID> out2(set3.size()+out1.size());
    sofa::type::vector<TriangleID>::iterator result2;
    result2 = std::set_intersection(set3.begin(),set3.end(),out1.begin(),out1.end(),out2.begin());
    out2.erase(result2,out2.end());
    
    msg_warning_when(out2.size() > 1) << "More than one triangle found for indices: [" << v1 << "; " << v2 << "; " << v3 << "]";

    if (out2.size()==1)
        return (int) (out2[0]);

    return InvalidID;
}

Size TriangleSetTopologyContainer::getNumberOfTriangles() const
{
    return sofa::Size(d_triangle.getValue().size());
}

Size TriangleSetTopologyContainer::getNumberOfElements() const
{
    return this->getNumberOfTriangles();
}

const sofa::type::vector< TriangleSetTopologyContainer::TrianglesAroundVertex > &TriangleSetTopologyContainer::getTrianglesAroundVertexArray()
{
    return m_trianglesAroundVertex;
}

const sofa::type::vector< TriangleSetTopologyContainer::TrianglesAroundEdge > &TriangleSetTopologyContainer::getTrianglesAroundEdgeArray()
{
    return m_trianglesAroundEdge;
}

const sofa::type::vector<TriangleSetTopologyContainer::EdgesInTriangle> &TriangleSetTopologyContainer::getEdgesInTriangleArray()
{
    return m_edgesInTriangle;
}

const TriangleSetTopologyContainer::TrianglesAroundVertex& TriangleSetTopologyContainer::getTrianglesAroundVertex(PointID id)
{
    if (id < m_trianglesAroundVertex.size())
        return m_trianglesAroundVertex[id];

    return InvalidSet;
}

const TriangleSetTopologyContainer::TrianglesAroundEdge& TriangleSetTopologyContainer::getTrianglesAroundEdge(EdgeID id)
{
    if (id < m_trianglesAroundEdge.size())
        return m_trianglesAroundEdge[id];

    return InvalidSet;
}

const TriangleSetTopologyContainer::EdgesInTriangle &TriangleSetTopologyContainer::getEdgesInTriangle(const TriangleID id)
{
    if (id < m_edgesInTriangle.size())
        return m_edgesInTriangle[id];

    return InvalidTriangle;
}

int TriangleSetTopologyContainer::getVertexIndexInTriangle(const Triangle &t, PointID vertexIndex) const
{
    if (t[0]==vertexIndex)
        return 0;
    else if (t[1]==vertexIndex)
        return 1;
    else if (t[2]==vertexIndex)
        return 2;
    else
        return -1;
}

int TriangleSetTopologyContainer::getEdgeIndexInTriangle(const EdgesInTriangle &t, EdgeID edgeIndex) const
{
    if (t[0]==edgeIndex)
        return 0;
    else if (t[1]==edgeIndex)
        return 1;
    else if (t[2]==edgeIndex)
        return 2;
    else
        return -1;
}


TriangleSetTopologyContainer::PointID TriangleSetTopologyContainer::getOtherPointInTriangle(const Triangle& t, PointID p1, PointID p2) const
{
    if (t[0] != p1 && t[0] != p2) return t[0];
    else if (t[1] != p1 && t[1] != p2) return t[1];
    else return t[2];
}


const sofa::type::vector<TriangleSetTopologyContainer::TriangleID>& TriangleSetTopologyContainer::getTrianglesOnBorder()
{
    if (!hasBorderElementLists()) // this method should only be called when border lists exists
    {
        dmsg_warning() << "getTrianglesOnBorder: trianglesOnBorder array is empty. Be sure to call createElementsOnBorder first.";
        createElementsOnBorder();
    }

    return m_trianglesOnBorder;
}


const sofa::type::vector<TriangleSetTopologyContainer::EdgeID>& TriangleSetTopologyContainer::getEdgesOnBorder()
{
    if (!hasBorderElementLists()) // this method should only be called when border lists exists
    {
        dmsg_warning() << "getEdgesOnBorder: edgesOnBorder array is empty. Be sure to call createElementsOnBorder first.";
        createElementsOnBorder();
    }

    return m_edgesOnBorder;
}


const sofa::type::vector<TriangleSetTopologyContainer::PointID>& TriangleSetTopologyContainer::getPointsOnBorder()
{
    if (!hasBorderElementLists()) // this method should only be called when border lists exists
    {
        dmsg_warning() << "getPointsOnBorder: pointsOnBorder array is empty. Be sure to call createElementsOnBorder first.";
        createElementsOnBorder();
    }

    return m_pointsOnBorder;
}


TriangleSetTopologyContainer::TrianglesAroundEdge &TriangleSetTopologyContainer::getTrianglesAroundEdgeForModification(const EdgeID i)
{
    if(!hasTrianglesAroundEdge())	// this method should only be called when the shell array exists
    {
        dmsg_warning() << "getTrianglesAroundEdgeForModification: TrianglesAroundEdgeArray is empty. Be sure to call createTrianglesAroundEdgeArray first.";
        createTrianglesAroundEdgeArray();
    }

    //TODO epernod (2020-04): this method should be removed as it can create a seg fault.
    return m_trianglesAroundEdge[i];
}

TriangleSetTopologyContainer::TrianglesAroundVertex &TriangleSetTopologyContainer::getTrianglesAroundVertexForModification(const PointID i)
{
    if(!hasTrianglesAroundVertex())	// this method should only be called when the shell array exists
    {
        dmsg_warning() << "getTrianglesAroundVertexForModification: TrianglesAroundVertexArray is empty. Be sure to call createTrianglesAroundVertexArray first.";
        createTrianglesAroundVertexArray();
    }

    //TODO epernod (2020-04): this method should be removed as it can create a seg fault.
    return m_trianglesAroundVertex[i];
}

bool TriangleSetTopologyContainer::checkTopology() const
{
    if (!d_checkTopology.getValue())
        return true;
    
    bool ret = true;
    const helper::ReadAccessor< Data< sofa::type::vector<Triangle> > > m_triangle = d_triangle;

    if (hasTrianglesAroundVertex())
    {
        std::set <int> triangleSet;

        for (size_t i = 0; i < m_trianglesAroundVertex.size(); ++i)
        {
            const sofa::type::vector<TriangleID> &tvs = m_trianglesAroundVertex[i];
            for (size_t j = 0; j < tvs.size(); ++j)
            {
                const Triangle& triangle = m_triangle[tvs[j]];
                const bool check_triangle_vertex_shell = (triangle[0] == i) || (triangle[1] == i) || (triangle[2] == i);
                if (!check_triangle_vertex_shell)
                {
                    msg_error() << "TriangleSetTopologyContainer::checkTopology() failed: triangle " << tvs[j] << ": [" << triangle << "] not around vertex: " << i;
                    ret = false;
                }

                triangleSet.insert(tvs[j]);
            }
        }

        if (triangleSet.size() != m_triangle.size())
        {
            msg_error() << "TriangleSetTopologyContainer::checkTopology() failed: found " << triangleSet.size() << " triangles in m_trianglesAroundVertex out of " << m_triangle.size();
            ret = false;
        }
    }


    if (hasTrianglesAroundEdge() && hasEdgesInTriangle())
    {
        // check first m_edgesInTriangle
        const helper::ReadAccessor< Data< sofa::type::vector<Edge> > > m_edge = d_edge;

        if (m_edgesInTriangle.size() != m_triangle.size())
        {
            msg_error() << "TriangleSetTopologyContainer::checkTopology() failed: m_edgesInTriangle size: " << m_edgesInTriangle.size() << " not equal to " << m_triangle.size();
            ret = false;
        }

        for (size_t i=0; i<m_edgesInTriangle.size(); ++i)
        {
            const Triangle& triangle = m_triangle[i];
            const EdgesInTriangle& eInTri = m_edgesInTriangle[i];

            for (unsigned int j=0; j<3; j++)
            {
                const Edge& edge = m_edge[eInTri[j]];
                int cptFound = 0;
                for (unsigned int k=0; k<3; k++)
                    if (edge[0] == triangle[k] || edge[1] == triangle[k])
                        cptFound++;

                if (cptFound != 2)
                {
                    msg_error() << "TriangleSetTopologyContainer::checkTopology() failed: edge: " << eInTri[j] << ": [" << edge << "] not found in triangle: " << i << ": " << triangle;
                    ret = false;
                }
            }
        }

        // check m_trianglesAroundEdge using checked m_edgesInTriangle
        std::set <int> triangleSet;
        for (size_t i = 0; i < m_trianglesAroundEdge.size(); ++i)
        {
            const sofa::type::vector<TriangleID> &tes = m_trianglesAroundEdge[i];
            for (size_t j = 0; j < tes.size(); ++j)
            {
                const EdgesInTriangle& eInTri = m_edgesInTriangle[tes[j]];
                const bool check_triangle_edge_shell = (eInTri[0] == i)
                    || (eInTri[1] == i)
                    || (eInTri[2] == i);
                if (!check_triangle_edge_shell)
                {
                    msg_error() << "TriangleSetTopologyContainer::checkTopology() failed: triangle: " << tes[j] << " with edges: [" << eInTri << "] not found around edge: " << i;
                    ret = false;
                }

                triangleSet.insert(tes[j]);
            }
        }

        if (triangleSet.size() != m_triangle.size())
        {
            msg_error() << "TriangleSetTopologyContainer::checkTopology() failed: found " << triangleSet.size() << " triangles in m_trianglesAroundEdge out of " << m_triangle.size();
            ret = false;
        }

    }

    return ret && EdgeSetTopologyContainer::checkTopology();
}


/// Get information about connexity of the mesh
/// @{

bool TriangleSetTopologyContainer::checkConnexity()
{
    const size_t nbr = this->getNbTriangles();

    if (nbr == 0)
    {
        msg_error() << "CheckConnexity: Can't compute connexity as there are no triangles";
        return false;
    }

    const VecTriangleID elemAll = this->getConnectedElement(0);

    if (elemAll.size() != nbr)
    {
        msg_warning() << "CheckConnexity: Triangles are missings. There is more than one connexe component.";
        return false;
    }
    return true;
}


Size TriangleSetTopologyContainer::getNumberOfConnectedComponent()
{
    const auto nbr = this->getNbTriangles();

    if (nbr == 0)
    {
        msg_error() << "Can't getNumberOfConnectedComponent as there are no triangles";
        return 0;
    }

    VecTriangleID elemAll = this->getConnectedElement(0);
    sofa::Size cpt = 1;

    while (elemAll.size() < nbr)
    {
        std::sort(elemAll.begin(), elemAll.end());
        sofa::Index other_triangleID = sofa::Index(elemAll.size());

        for (sofa::Index i = 0; i<elemAll.size(); ++i)
            if (elemAll[i] != i)
            {
                other_triangleID = i;
                break;
            }

        VecTriangleID elemTmp = this->getConnectedElement((TriangleID)other_triangleID);
        cpt++;

        elemAll.insert(elemAll.begin(), elemTmp.begin(), elemTmp.end());
    }
    return cpt;
}


const TriangleSetTopologyContainer::VecTriangleID TriangleSetTopologyContainer::getConnectedElement(TriangleID elem)
{
    VecTriangleID elemAll;
    if(!hasTrianglesAroundVertex())	// this method should only be called when the shell array exists
    {
        dmsg_warning() << "getElementAroundElements: TrianglesAroundVertexArray is empty. Be sure to call createTrianglesAroundVertexArray first.";
        return elemAll;
    }

    VecTriangleID elemOnFront, elemPreviousFront, elemNextFront;
    bool end = false;
    size_t cpt = 0;
    const size_t nbr = this->getNbTriangles();

    // init algo
    elemAll.push_back(elem);
    elemOnFront.push_back(elem);
    elemPreviousFront.clear();
    cpt++;

    while (!end && cpt < nbr)
    {
        // First Step - Create new region
        elemNextFront = this->getElementAroundElements(elemOnFront); // for each triangleID on the propagation front
        // Second Step - Avoid backward direction
        for (size_t i = 0; i<elemNextFront.size(); ++i)
        {
            bool find = false;
            TriangleID id = elemNextFront[i];

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
            msg_error() << "Loop for computing connexity has reach end.";
        }

        // iterate
        elemOnFront = elemPreviousFront;
        elemPreviousFront.clear();
    }
    return elemAll;
}


const TriangleSetTopologyContainer::VecTriangleID TriangleSetTopologyContainer::getElementAroundElement(TriangleID elem)
{
    VecTriangleID elems;

    if (!hasTrianglesAroundVertex())
    {
        dmsg_warning() << "getElementAroundElements: TrianglesAroundVertexArray is empty. Be sure to call createTrianglesAroundVertexArray first.";
        return elems;
    }

    Triangle the_tri = this->getTriangle(elem);

    for(PointID i = 0; i<3; ++i) // for each node of the triangle
    {
        TrianglesAroundVertex triAV = this->getTrianglesAroundVertex(the_tri[i]);

        for (size_t j = 0; j<triAV.size(); ++j) // for each triangle around the node
        {
            bool find = false;
            TriangleID id = triAV[j];

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


const TriangleSetTopologyContainer::VecTriangleID TriangleSetTopologyContainer::getElementAroundElements(VecTriangleID elems)
{
    VecTriangleID elemAll;
    if (!hasTrianglesAroundVertex())
    {
        dmsg_warning() << "getElementAroundElements: TrianglesAroundVertexArray is empty. Be sure to call createTrianglesAroundVertexArray first.";
        return elemAll;
    }

    VecTriangleID elemTmp;
    for (size_t i = 0; i <elems.size(); ++i) // for each triangleId of input vector
    {
        VecTriangleID elemTmp2 = this->getElementAroundElement(elems[i]);

        elemTmp.insert(elemTmp.end(), elemTmp2.begin(), elemTmp2.end());
    }

    for (size_t i = 0; i<elemTmp.size(); ++i) // for each Triangle Id found
    {
        bool find = false;
        TriangleID id = elemTmp[i];

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

bool TriangleSetTopologyContainer::hasTriangles() const
{
    d_triangle.updateIfDirty();
    return !(d_triangle.getValue()).empty();
}

bool TriangleSetTopologyContainer::hasEdgesInTriangle() const
{
    return !m_edgesInTriangle.empty();
}

bool TriangleSetTopologyContainer::hasTrianglesAroundVertex() const
{
    return !m_trianglesAroundVertex.empty();
}

bool TriangleSetTopologyContainer::hasTrianglesAroundEdge() const
{
    return !m_trianglesAroundEdge.empty();
}

bool TriangleSetTopologyContainer::hasBorderElementLists() const
{
    if(!m_trianglesOnBorder.empty() && !m_edgesOnBorder.empty() && !m_pointsOnBorder.empty())
        return true;
    else
        return false;
}

void TriangleSetTopologyContainer::clearTrianglesAroundVertex()
{
    for(size_t i=0; i<m_trianglesAroundVertex.size(); ++i)
        m_trianglesAroundVertex[i].clear();

    m_trianglesAroundVertex.clear();
}

void TriangleSetTopologyContainer::clearTrianglesAroundEdge()
{
    for(size_t i=0; i<m_trianglesAroundEdge.size(); ++i)
        m_trianglesAroundEdge[i].clear();

    m_trianglesAroundEdge.clear();
}

void TriangleSetTopologyContainer::clearEdgesInTriangle()
{
    m_edgesInTriangle.clear();
}

void TriangleSetTopologyContainer::clearTriangles()
{
    helper::WriteAccessor< Data< sofa::type::vector<Triangle> > > m_triangle = d_triangle;
    m_triangle.clear();

}

void TriangleSetTopologyContainer::clearBorderElementLists()
{
    m_trianglesOnBorder.clear();
    m_edgesOnBorder.clear();
    m_pointsOnBorder.clear();
}

void TriangleSetTopologyContainer::clear()
{
    clearTrianglesAroundVertex();
    clearTrianglesAroundEdge();
    clearEdgesInTriangle();
    clearTriangles();
    clearBorderElementLists();
    EdgeSetTopologyContainer::clear();
}

void TriangleSetTopologyContainer::setTriangleTopologyToDirty()
{
    // set this container to dirty
    m_triangleTopologyDirty = true;

    // set all engines link to this container to dirty
    auto& triangleTopologyHandlerList = getTopologyHandlerList(sofa::geometry::ElementType::TRIANGLE);
    for (const auto topoHandler : triangleTopologyHandlerList)
    {
        topoHandler->setDirtyValue();
        msg_info() << "Triangle Topology Set dirty engine: " << topoHandler->getName();
    }
}

void TriangleSetTopologyContainer::cleanTriangleTopologyFromDirty()
{
    m_triangleTopologyDirty = false;

    // security, clean all engines to avoid loops
    auto& triangleTopologyHandlerList = getTopologyHandlerList(sofa::geometry::ElementType::TRIANGLE);
    for (const auto topoHandler : triangleTopologyHandlerList)
    {
        if (topoHandler->isDirty())
        {
            msg_warning() << "Triangle Topology update did not clean engine: " << topoHandler->getName();
            topoHandler->cleanDirty();
        }
    }
}

bool TriangleSetTopologyContainer::linkTopologyHandlerToData(core::topology::TopologyHandler* topologyHandler, sofa::geometry::ElementType elementType)
{
    if (elementType == sofa::geometry::ElementType::TRIANGLE)
    {
        d_triangle.addOutput(topologyHandler);
        return true;
    }
    else
    {
        return EdgeSetTopologyContainer::linkTopologyHandlerToData(topologyHandler, elementType);
    }
}

bool TriangleSetTopologyContainer::unlinkTopologyHandlerToData(core::topology::TopologyHandler* topologyHandler, sofa::geometry::ElementType elementType)
{
    if (elementType == sofa::geometry::ElementType::TRIANGLE)
    {
        d_triangle.delOutput(topologyHandler);
        return true;
    }
    else
    {
        return EdgeSetTopologyContainer::unlinkTopologyHandlerToData(topologyHandler, elementType);
    }
}

} //namespace sofa::component::topology::container::dynamic
