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
#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace std;
using namespace sofa::defaulttype;


SOFA_DECL_CLASS(TriangleSetTopologyContainer)
int TriangleSetTopologyContainerClass = core::RegisterObject("Triangle set topology container")
        .add< TriangleSetTopologyContainer >()
        ;

TriangleSetTopologyContainer::TriangleSetTopologyContainer()
    : EdgeSetTopologyContainer()
    , d_triangle(initData(&d_triangle, "triangles", "List of triangle indices"))
{

}


void TriangleSetTopologyContainer::addTriangle( int a, int b, int c )
{
    helper::WriteAccessor< Data< sofa::helper::vector<Triangle> > > m_triangle = d_triangle;
    m_triangle.push_back(Triangle(a,b,c));
    if (a >= getNbPoints()) setNbPoints(a+1);
    if (b >= getNbPoints()) setNbPoints(b+1);
    if (c >= getNbPoints()) setNbPoints(c+1);
}

void TriangleSetTopologyContainer::init()
{
    EdgeSetTopologyContainer::init();
    d_triangle.updateIfDirty(); // make sure m_triangle is up to date
}

void TriangleSetTopologyContainer::reinit()
{
    EdgeSetTopologyContainer::reinit();
}


void TriangleSetTopologyContainer::createTriangleSetArray()
{
	if (CHECK_TOPOLOGY)
		msg_error() << "This method must be implemented by a child topology.";
}

void TriangleSetTopologyContainer::createTrianglesAroundVertexArray ()
{
    if(!hasTriangles()) // this method should only be called when triangles exist
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "Triangle array is empty.";

        createTriangleSetArray();
    }

    if(hasTrianglesAroundVertex())
    {
        clearTrianglesAroundVertex();
    }

    int nbPoints = getNbPoints();
    if (nbPoints == 0) // in case only Data have been copied and not going thourgh AddTriangle methods.
        this->setNbPoints(d_initPoints.getValue().size());

    m_trianglesAroundVertex.resize(getNbPoints());
    
    helper::ReadAccessor< Data< sofa::helper::vector<Triangle> > > m_triangle = d_triangle;

    for (unsigned int i = 0; i < m_triangle.size(); ++i)
    {
        // adding edge i in the edge shell of both points
        for (unsigned int j=0; j<3; ++j)
            m_trianglesAroundVertex[ m_triangle[i][j]  ].push_back( i );
    }
}

void TriangleSetTopologyContainer::createTrianglesAroundEdgeArray ()
{
    if(!hasTriangles()) // this method should only be called when triangles exist
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "Triangle array is empty.";

        createTriangleSetArray();
    }

    if(!hasEdges()) // this method should only be called when edges exist
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "Edge array is empty.";

        createEdgeSetArray();
    }

    if(!hasEdgesInTriangle())
        createEdgesInTriangleArray();

    const unsigned int numTriangles = getNumberOfTriangles();
    const unsigned int numEdges = getNumberOfEdges();

    if(hasTrianglesAroundEdge())
    {
        clearTrianglesAroundEdge();
    }

    m_trianglesAroundEdge.resize( numEdges );

    for (unsigned int i = 0; i < numTriangles; ++i)
    {
        // adding triangle i in the triangle shell of all edges
        for (unsigned int j=0; j<3; ++j)
        {
            m_trianglesAroundEdge[ m_edgesInTriangle[i][j] ].push_back( i );
        }
    }
}

void TriangleSetTopologyContainer::createEdgeSetArray()
{

    if(!hasTriangles()) // this method should only be called when triangles exist
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "Triangle array is empty.";

        createTriangleSetArray();
    }

    if(hasEdges())
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "Edge array is not empty.";


        // clear edges and all shells that depend on edges
        EdgeSetTopologyContainer::clear();

        if(hasEdgesInTriangle())
            clearEdgesInTriangle();

        if(hasTrianglesAroundEdge())
            clearTrianglesAroundEdge();
    }

    // create a temporary map to find redundant edges
    std::map<Edge, unsigned int> edgeMap;
    helper::WriteAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;
    helper::ReadAccessor< Data< sofa::helper::vector<Triangle> > > m_triangle = d_triangle;

    for (unsigned int i=0; i<m_triangle.size(); ++i)
    {
        const Triangle &t = m_triangle[i];
        for(unsigned int j=0; j<3; ++j)
        {
            const unsigned int v1 = t[(j+1)%3];
            const unsigned int v2 = t[(j+2)%3];

            // sort vertices in lexicographic order
            const Edge e = ((v1<v2) ? Edge(v1,v2) : Edge(v2,v1));

            if(edgeMap.find(e) == edgeMap.end())
            {
                // edge not in edgeMap so create a new one
                const unsigned int edgeIndex = (unsigned int)edgeMap.size();
                edgeMap[e] = edgeIndex;
//	      m_edge.push_back(e); Changed to have oriented edges on the border of the triangulation
                m_edge.push_back(Edge(v1,v2));
            }
        }
    }
}

void TriangleSetTopologyContainer::createEdgesInTriangleArray()
{
    if(!hasTriangles()) // this method should only be called when triangles exist
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "Triangle array is empty.";

        createTriangleSetArray();
    }

    // this should never be called : remove existing triangle edges
    if(hasEdgesInTriangle())
        clearEdgesInTriangle();

    helper::ReadAccessor< Data< sofa::helper::vector<Triangle> > > m_triangle = d_triangle;

    if(!hasEdges()) // To optimize, this method should be called without creating edgesArray before.
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "Edge array is empty.";


        /// create edge array and triangle edge array at the same time
        const unsigned int numTriangles = getNumberOfTriangles();
        m_edgesInTriangle.resize(numTriangles);


        // create a temporary map to find redundant edges
        std::map<Edge, unsigned int> edgeMap;
        helper::WriteAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;

        for (unsigned int i=0; i<m_triangle.size(); ++i)
        {
            const Triangle &t = m_triangle[i];
            for(unsigned int j=0; j<3; ++j)
            {
                const unsigned int v1 = t[(j+1)%3];
                const unsigned int v2 = t[(j+2)%3];

                // sort vertices in lexicographic order
                const Edge e = ((v1<v2) ? Edge(v1,v2) : Edge(v2,v1));

                if(edgeMap.find(e) == edgeMap.end())
                {
                    // edge not in edgeMap so create a new one
                    const unsigned int edgeIndex = (unsigned int)edgeMap.size();
                    /// add new edge
                    edgeMap[e] = edgeIndex;
//			  m_edge.push_back(e);
                    m_edge.push_back(Edge(v1,v2));

                }
                m_edgesInTriangle[i][j] = edgeMap[e];
            }
        }
    }
    else
    {
        /// there are already existing edges : must use an inefficient method. Parse all triangles and find the edge that match each triangle edge
        helper::ReadAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;
        const unsigned int numTriangles = getNumberOfTriangles();
        const unsigned int numEdges = getNumberOfEdges();

        m_edgesInTriangle.resize(numTriangles);
        /// create a multi map where the key is a vertex index and the content is the indices of edges adjacent to that vertex.
        std::multimap<PointID, EdgeID> edgesAroundVertexMap;
        std::multimap<PointID, EdgeID>::iterator it;
        bool foundEdge;

        for (unsigned int edge=0; edge<numEdges; ++edge)  //Todo: check if not better using multimap <PointID ,TriangleID> and for each edge, push each triangle present in both shell
        {
            edgesAroundVertexMap.insert(std::pair<PointID, EdgeID> (m_edge[edge][0],edge));
            edgesAroundVertexMap.insert(std::pair<PointID, EdgeID> (m_edge[edge][1],edge));
        }

        for(unsigned int i=0; i<numTriangles; ++i)
        {
            const Triangle &t = m_triangle[i];
            // adding edge i in the edge shell of both points
            for(unsigned int j=0; j<3; ++j)
            {
                //finding edge i in edge array
                std::pair<std::multimap<PointID, EdgeID>::iterator, std::multimap<PointID, EdgeID>::iterator > itPair=edgesAroundVertexMap.equal_range(t[(j+1)%3]);

                foundEdge=false;
                for(it=itPair.first; (it!=itPair.second) && (foundEdge==false); ++it)
                {
                    unsigned int edge = (*it).second;
                    if ( (m_edge[edge][0] == t[(j+1)%3] && m_edge[edge][1] == t[(j+2)%3]) || (m_edge[edge][0] == t[(j+2)%3] && m_edge[edge][1] == t[(j+1)%3]))
                    {
                        m_edgesInTriangle[i][j] = edge;
                        foundEdge=true;
                    }
                }

				if (CHECK_TOPOLOGY)
					if (foundEdge==false)
						msg_warning() << "Cannot find edge for triangle " << i << "and edge "<< j;

            }
        }
    }
}



void TriangleSetTopologyContainer::createElementsOnBorder()
{

    if(!hasTrianglesAroundEdge())	// Use the trianglesAroundEdgeArray. Should check if it is consistent
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "Triangle edge shell array is empty.";


        createTrianglesAroundEdgeArray();
    }

    if(!m_trianglesOnBorder.empty())
        m_trianglesOnBorder.clear();

    if(!m_edgesOnBorder.empty())
        m_edgesOnBorder.clear();

    if(!m_pointsOnBorder.empty())
        m_pointsOnBorder.clear();

    const unsigned int nbrEdges = getNumberOfEdges();
    bool newTriangle = true;
    bool newEdge = true;
    bool newPoint = true;

    helper::ReadAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;
    for (unsigned int i = 0; i < nbrEdges; i++)
    {
        if (m_trianglesAroundEdge[i].size() == 1) // I.e this edge is on a border
        {

            // --- Triangle case ---
            for (unsigned int j = 0; j < m_trianglesOnBorder.size(); j++) // Loop to avoid duplicated indices
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
            for (unsigned int j = 0; j < m_edgesOnBorder.size(); j++) // Loop to avoid duplicated indices
            {
                if (m_edgesOnBorder[j] == i)
                {
                    newEdge = false;
                    break;
                }
            }

            if(newEdge) // If index doesn't already exist, add it to the list of edges On border.
            {
                m_edgesOnBorder.push_back (i);
            }


            // --- Point case ---
            PointID firstVertex = m_edge[i][0];
            for (unsigned int j = 0; j < m_pointsOnBorder.size(); j++) // Loop to avoid duplicated indices
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
    if (id >= (unsigned int)this->getNbTriangles())
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "Triangle ID out of bounds.";

        return;
    }
    Triangle& tri = (*d_triangle.beginEdit())[id];
    unsigned int tmp = tri[1];
    tri[1] = tri[2];
    tri[2] = tmp;
    d_triangle.endEdit();

    return;
}


const sofa::helper::vector<TriangleSetTopologyContainer::Triangle> & TriangleSetTopologyContainer::getTriangleArray()
{
    if(!hasTriangles() && getNbPoints()>0)
    {
		if (CHECK_TOPOLOGY)
			msg_info() << "Creating triangle array.";

        createTriangleSetArray();
    }

    return d_triangle.getValue();
}


const TriangleSetTopologyContainer::Triangle TriangleSetTopologyContainer::getTriangle (TriangleID i)
{
    if(!hasTriangles())
        createTriangleSetArray();

    if (i >= getNbTriangles())
        return Triangle(-1, -1, -1);
    else
        return (d_triangle.getValue())[i];
}



int TriangleSetTopologyContainer::getTriangleIndex(PointID v1, PointID v2, PointID v3)
{
    if(!hasTrianglesAroundVertex())
        createTrianglesAroundVertexArray();

    sofa::helper::vector<unsigned int> set1 = getTrianglesAroundVertex(v1);
    sofa::helper::vector<unsigned int> set2 = getTrianglesAroundVertex(v2);
    sofa::helper::vector<unsigned int> set3 = getTrianglesAroundVertex(v3);

    sort(set1.begin(), set1.end());
    sort(set2.begin(), set2.end());
    sort(set3.begin(), set3.end());

    // The destination vector must be large enough to contain the result.
    sofa::helper::vector<unsigned int> out1(set1.size()+set2.size());
    sofa::helper::vector<unsigned int>::iterator result1;
    result1 = std::set_intersection(set1.begin(),set1.end(),set2.begin(),set2.end(),out1.begin());
    out1.erase(result1,out1.end());

    sofa::helper::vector<unsigned int> out2(set3.size()+out1.size());
    sofa::helper::vector<unsigned int>::iterator result2;
    result2 = std::set_intersection(set3.begin(),set3.end(),out1.begin(),out1.end(),out2.begin());
    out2.erase(result2,out2.end());

	if (CHECK_TOPOLOGY)
		if(out2.size() > 1)
			msg_warning() << "More than one triangle found";


    if (out2.size()==1)
        return (int) (out2[0]);
    else
        return -1;
}

unsigned int TriangleSetTopologyContainer::getNumberOfTriangles() const
{
    return (unsigned int)d_triangle.getValue().size();
}

unsigned int TriangleSetTopologyContainer::getNumberOfElements() const
{
    return this->getNumberOfTriangles();
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &TriangleSetTopologyContainer::getTrianglesAroundVertexArray()
{
    if(!hasTrianglesAroundVertex())	// this method should only be called when the shell array exists
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "Triangle vertex shell array is empty.";

        createTrianglesAroundVertexArray();
    }

    return m_trianglesAroundVertex;
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &TriangleSetTopologyContainer::getTrianglesAroundEdgeArray()
{
    if(!hasTrianglesAroundEdge())	// this method should only be called when the shell array exists
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "Triangle edge shell array is empty.";

        createTrianglesAroundEdgeArray();
    }

    return m_trianglesAroundEdge;
}

const sofa::helper::vector<TriangleSetTopologyContainer::EdgesInTriangle> &TriangleSetTopologyContainer::getEdgesInTriangleArray()
{
    if(m_edgesInTriangle.empty())
        createEdgesInTriangleArray();

    return m_edgesInTriangle;
}

const TriangleSetTopologyContainer::TrianglesAroundVertex& TriangleSetTopologyContainer::getTrianglesAroundVertex(PointID i)
{
    if(!hasTrianglesAroundVertex())	// this method should only be called when the shell array exists
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "Triangle vertex shell array is empty.";

        createTrianglesAroundVertexArray();
    }
    else if( i >= m_trianglesAroundVertex.size())
    {
		if (CHECK_TOPOLOGY)
			msg_error() << "Index out of bounds.";

        createTrianglesAroundVertexArray();
    }

    return m_trianglesAroundVertex[i];
}

const TriangleSetTopologyContainer::TrianglesAroundEdge& TriangleSetTopologyContainer::getTrianglesAroundEdge(EdgeID i)
{
    if(!hasTrianglesAroundEdge())	// this method should only be called when the shell array exists
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "Triangle edge shell array is empty.";

        createTrianglesAroundEdgeArray();
    }
    else if( i >= m_trianglesAroundEdge.size())
    {
		if (CHECK_TOPOLOGY)
			msg_error() << "Index out of bounds.";

        createTrianglesAroundEdgeArray();
    }

    return m_trianglesAroundEdge[i];
}

const TriangleSetTopologyContainer::EdgesInTriangle &TriangleSetTopologyContainer::getEdgesInTriangle(const unsigned int i)
{
    if(m_edgesInTriangle.empty())
        createEdgesInTriangleArray();

    if( i >= m_edgesInTriangle.size())
    {
		if (CHECK_TOPOLOGY)
			msg_error() << "Index out of bounds.";

        createEdgesInTriangleArray();
    }

    return m_edgesInTriangle[i];
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


const sofa::helper::vector <TriangleSetTopologyContainer::TriangleID>& TriangleSetTopologyContainer::getTrianglesOnBorder()
{
    if (!hasBorderElementLists()) // this method should only be called when border lists exists
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "A border element list is empty.";

        createElementsOnBorder();
    }

    return m_trianglesOnBorder;
}


const sofa::helper::vector <TriangleSetTopologyContainer::EdgeID>& TriangleSetTopologyContainer::getEdgesOnBorder()
{
    if (!hasBorderElementLists()) // this method should only be called when border lists exists
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "A border element list is empty.";

        createElementsOnBorder();
    }

    return m_edgesOnBorder;
}


const sofa::helper::vector <TriangleSetTopologyContainer::PointID>& TriangleSetTopologyContainer::getPointsOnBorder()
{
    if (!hasBorderElementLists()) // this method should only be called when border lists exists
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "A border element list is empty.";

        createElementsOnBorder();
    }

    return m_pointsOnBorder;
}


sofa::helper::vector< unsigned int > &TriangleSetTopologyContainer::getTrianglesAroundEdgeForModification(const unsigned int i)
{
    if(!hasTrianglesAroundEdge())	// this method should only be called when the shell array exists
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "Triangle edge shell array is empty.";

        createTrianglesAroundEdgeArray();
    }

    if( i >= m_trianglesAroundEdge.size())
    {
		if (CHECK_TOPOLOGY)
			msg_error() << "Index out of bounds.";

        createTrianglesAroundEdgeArray();
    }

    return m_trianglesAroundEdge[i];
}

sofa::helper::vector< unsigned int > &TriangleSetTopologyContainer::getTrianglesAroundVertexForModification(const unsigned int i)
{
    if(!hasTrianglesAroundVertex())	// this method should only be called when the shell array exists
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "Triangle vertex shell array is empty.";

        createTrianglesAroundVertexArray();
    }

    if( i >= m_trianglesAroundVertex.size())
    {
		if (CHECK_TOPOLOGY)
			msg_error() << "Index out of bounds.";

        createTrianglesAroundVertexArray();
    }

    return m_trianglesAroundVertex[i];
}

bool TriangleSetTopologyContainer::checkTopology() const
{
	if (CHECK_TOPOLOGY)
	{
		bool ret = true;
		helper::ReadAccessor< Data< sofa::helper::vector<Triangle> > > m_triangle = d_triangle;

		if (hasTrianglesAroundVertex())
		{
			std::set <int> triangleSet;
			std::set<int>::iterator it;

			for (unsigned int i = 0; i < m_trianglesAroundVertex.size(); ++i)
			{
				const sofa::helper::vector<unsigned int> &tvs = m_trianglesAroundVertex[i];
				for (unsigned int j = 0; j < tvs.size(); ++j)
				{
					bool check_triangle_vertex_shell = (m_triangle[tvs[j]][0] == i)
						|| (m_triangle[tvs[j]][1] == i)
						|| (m_triangle[tvs[j]][2] == i);
					if (!check_triangle_vertex_shell)
					{
						msg_error() << "*** CHECK FAILED : check_triangle_vertex_shell, i = " << i << " , j = " << j;
						ret = false;
					}

					it = triangleSet.find(tvs[j]);
					if (it == triangleSet.end())
					{
						triangleSet.insert(tvs[j]);
					}
				}
			}

			if (triangleSet.size() != m_triangle.size())
			{
				msg_error() << "*** CHECK FAILED : check_triangle_vertex_shell, triangle are missing in m_trianglesAroundVertex";
				ret = false;
			}
		}

		if (hasTrianglesAroundEdge())
		{
			std::set <int> triangleSet;
			std::set<int>::iterator it;

			for (unsigned int i = 0; i < m_trianglesAroundEdge.size(); ++i)
			{
				const sofa::helper::vector<unsigned int> &tes = m_trianglesAroundEdge[i];
				for (unsigned int j = 0; j < tes.size(); ++j)
				{
					bool check_triangle_edge_shell = (m_edgesInTriangle[tes[j]][0] == i)
						|| (m_edgesInTriangle[tes[j]][1] == i)
						|| (m_edgesInTriangle[tes[j]][2] == i);
					if (!check_triangle_edge_shell)
					{
						msg_error() << "*** CHECK FAILED : check_triangle_edge_shell, i = " << i << " , j = " << j;
						ret = false;
					}

					it = triangleSet.find(tes[j]);
					if (it == triangleSet.end())
					{
						triangleSet.insert(tes[j]);
					}

				}
			}

			if (triangleSet.size() != m_triangle.size())
			{
				msg_error() << "*** CHECK FAILED : check_triangle_edge_shell, triangle are missing in m_trianglesAroundEdge";
				ret = false;
			}

		}

		return ret && EdgeSetTopologyContainer::checkTopology();
	}

    return true;
}


/// Get information about connexity of the mesh
/// @{

bool TriangleSetTopologyContainer::checkConnexity()
{
    unsigned int nbr = this->getNbTriangles();

    if (nbr == 0)
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "Can't compute connexity as there are no triangles";

        return false;
    }

    VecTriangleID elemAll = this->getConnectedElement(0);

    if (elemAll.size() != nbr)
    {
		msg_warning() << "Computing connexity, triangles are missings. There is more than one connexe component.";
        return false;
    }
    return true;
}


unsigned int TriangleSetTopologyContainer::getNumberOfConnectedComponent()
{
    unsigned int nbr = this->getNbTriangles();

    if (nbr == 0)
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "Can't compute connexity as there are no triangles";

        return 0;
    }

    VecTriangleID elemAll = this->getConnectedElement(0);
    unsigned int cpt = 1;

    while (elemAll.size() < nbr)
    {
        std::sort(elemAll.begin(), elemAll.end());
        TriangleID other_triangleID = elemAll.size();

        for (TriangleID i = 0; i<elemAll.size(); ++i)
            if (elemAll[i] != i)
            {
                other_triangleID = i;
                break;
            }

        VecTriangleID elemTmp = this->getConnectedElement(other_triangleID);
        cpt++;

        elemAll.insert(elemAll.begin(), elemTmp.begin(), elemTmp.end());
    }
    return cpt;
}


const TriangleSetTopologyContainer::VecTriangleID TriangleSetTopologyContainer::getConnectedElement(TriangleID elem)
{
    if(!hasTrianglesAroundVertex())	// this method should only be called when the shell array exists
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "Triangle vertex shell array is empty.";

        createTrianglesAroundVertexArray();
    }

    VecTriangleID elemAll;
    VecTriangleID elemOnFront, elemPreviousFront, elemNextFront;
    bool end = false;
    size_t cpt = 0;
    size_t nbr = this->getNbTriangles();

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
        for (unsigned int i = 0; i<elemNextFront.size(); ++i)
        {
            bool find = false;
            TriangleID id = elemNextFront[i];

            for (unsigned int j = 0; j<elemAll.size(); ++j)
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
			if (CHECK_TOPOLOGY)
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
		if (CHECK_TOPOLOGY)
			msg_warning() << "Triangle vertex shell array is empty.";

        createTrianglesAroundVertexArray();
    }

    Triangle the_tri = this->getTriangle(elem);

    for(unsigned int i = 0; i<3; ++i) // for each node of the triangle
    {
        TrianglesAroundVertex triAV = this->getTrianglesAroundVertex(the_tri[i]);

        for (unsigned int j = 0; j<triAV.size(); ++j) // for each triangle around the node
        {
            bool find = false;
            TriangleID id = triAV[j];

            if (id == elem)
                continue;

            for (unsigned int k = 0; k<elems.size(); ++k) // check no redundancy
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
    VecTriangleID elemTmp;

    if (!hasTrianglesAroundVertex())
    {
		if (CHECK_TOPOLOGY)
			msg_warning() << "Triangle vertex shell array is empty.";

        createTrianglesAroundVertexArray();
    }

    for (unsigned int i = 0; i <elems.size(); ++i) // for each triangleId of input vector
    {
        VecTriangleID elemTmp2 = this->getElementAroundElement(elems[i]);

        elemTmp.insert(elemTmp.end(), elemTmp2.begin(), elemTmp2.end());
    }

    for (unsigned int i = 0; i<elemTmp.size(); ++i) // for each Triangle Id found
    {
        bool find = false;
        TriangleID id = elemTmp[i];

        for (unsigned int j = 0; j<elems.size(); ++j) // check no redundancy with input vector
            if (id == elems[j])
            {
                find = true;
                break;
            }

        if (!find)
        {
            for (unsigned int j = 0; j<elemAll.size(); ++j) // check no redundancy in output vector
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
    for(unsigned int i=0; i<m_trianglesAroundVertex.size(); ++i)
        m_trianglesAroundVertex[i].clear();

    m_trianglesAroundVertex.clear();
}

void TriangleSetTopologyContainer::clearTrianglesAroundEdge()
{
    for(unsigned int i=0; i<m_trianglesAroundEdge.size(); ++i)
        m_trianglesAroundEdge[i].clear();

    m_trianglesAroundEdge.clear();
}

void TriangleSetTopologyContainer::clearEdgesInTriangle()
{
    m_edgesInTriangle.clear();
}

void TriangleSetTopologyContainer::clearTriangles()
{
    helper::WriteAccessor< Data< sofa::helper::vector<Triangle> > > m_triangle = d_triangle;
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



void TriangleSetTopologyContainer::updateTopologyEngineGraph()
{
    // calling real update Data graph function implemented once in PointSetTopologyModifier
    this->updateDataEngineGraph(this->d_triangle, this->m_enginesList);

    // will concatenate with edges one:
    EdgeSetTopologyContainer::updateTopologyEngineGraph();
}



} // namespace topology

} // namespace component

} // namespace sofa
