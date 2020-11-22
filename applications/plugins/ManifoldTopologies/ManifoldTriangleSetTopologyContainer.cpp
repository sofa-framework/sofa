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

#include <ManifoldTopologies/ManifoldTriangleSetTopologyContainer.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace std;
using namespace sofa::defaulttype;


int ManifoldTriangleSetTopologyContainerClass = core::RegisterObject("Manifold Triangle set topology container")
        .add< ManifoldTriangleSetTopologyContainer >()
        ;



ManifoldTriangleSetTopologyContainer::ManifoldTriangleSetTopologyContainer()
    : TriangleSetTopologyContainer()
{
}




bool ManifoldTriangleSetTopologyContainer::checkTopology() const
{
    if (!d_checkTopology.getValue())
        return true;

    bool ret = true;
    helper::ReadAccessor< Data< sofa::helper::vector<Triangle> > > m_triangle = d_triangle;
    helper::ReadAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;

    //Test the shell m_trianglesAroundVertex
    if(hasTrianglesAroundVertex())
    {
        //Number of different elements needed for this function
        const size_t nbrVertices = getNbPoints();
        const size_t nbrTriangles = getNumberOfTriangles();

        //Temporary objects
        Triangle vertexTriangle;
//        unsigned int firstVertex;
        unsigned int vertex, vertexNext;

        //Temporary containers
        sofa::helper::vector< std::map<unsigned int, unsigned int> > map_Triangles;
        std::map<unsigned int, unsigned int>::iterator it1;
        map_Triangles.resize(nbrVertices);

        //Fill temporary an other TrianglesAroundVertexarray
        for (unsigned int triangleIndex = 0; triangleIndex < nbrTriangles; ++triangleIndex)
        {

            vertexTriangle = m_triangle[triangleIndex];

            for (unsigned int i=0; i<3; ++i)
            {
                // Test if there are several triangles with the same edge in the same order.
                it1 = map_Triangles[vertexTriangle[i]].find(vertexTriangle[(i+1)%3]);
                if (it1 != map_Triangles[vertexTriangle[i]].end())
                {
                    msg_error() << "Manifold_triangle_vertex_shell: Bad connection or triangle orientation between vertices: " << vertexTriangle[i] << " and " << vertexTriangle[(i+1)%3];
                    ret = false;
                }

                map_Triangles[vertexTriangle[i]].insert(std::pair<unsigned int,unsigned int> (vertexTriangle[(i+1)%3], triangleIndex)); //multi
            }
        }


        //Check if ManifoldTrianglesAroundVertexArray is contiguous and if no triangles are missing in the shell
        for (unsigned int vertexIndex = 0; vertexIndex < nbrVertices; ++vertexIndex)
        {

            //Test on the size of the shell: If size differ from the previous map, this means the shell is not well fill.
            if ( m_trianglesAroundVertex[vertexIndex].size() != map_Triangles[vertexIndex].size())
            {
                msg_error() << "Manifold_triangle_vertex_shell: Triangles are missing in the shell around the vertex: " << vertexIndex;
                ret = false;
            }

            vertexTriangle = m_triangle[ m_trianglesAroundVertex[vertexIndex][0] ];
//            firstVertex = vertexTriangle[ ( getVertexIndexInTriangle(vertexTriangle, vertexIndex)+1 )%3 ];

            //For each vertex, test if the triangle adjacent are fill in the right contiguous way.
            //Triangles should be adjacent one to the next other in counterclockwise direction.
            for (unsigned int triangleIndex = 0; triangleIndex < m_trianglesAroundVertex[vertexIndex].size()-1; ++triangleIndex)
            {
                vertexTriangle = m_triangle[ m_trianglesAroundVertex[vertexIndex][triangleIndex] ];
                vertex = vertexTriangle[ ( getVertexIndexInTriangle(vertexTriangle, vertexIndex)+2 )%3 ];

                vertexTriangle = m_triangle[ m_trianglesAroundVertex[vertexIndex][triangleIndex+1] ];
                vertexNext = vertexTriangle[ ( getVertexIndexInTriangle(vertexTriangle, vertexIndex)+1 )%3 ];

                if (vertex != vertexNext)
                {
                    msg_error() << "Manifold_triangle_vertex_shell: Triangles are not contiguous or not well connected around the vertex: " << vertexIndex;
                    ret = false;
                    break;
                }
            }
        }
    }


    //Test the shell m_trianglesAroundEdge
    if (hasTrianglesAroundEdge())
    {

        //Number of different elements needed for this function
        const size_t nbrEdges = getNumberOfEdges();
        const size_t nbrTriangles = getNumberOfTriangles();

        //Temporary objects
        Triangle vertexTriangle;
        size_t nbrEdgesInTriangle;
        unsigned int vertexInTriangle;

        //Temporary containers
        std::vector <unsigned int> nbr_edgesInTriangle;
        nbr_edgesInTriangle.resize(nbrEdges);

        //Create a vector containing the number of triangle adjacent to each edge
        for (unsigned int triangleIndex = 0; triangleIndex < nbrTriangles; ++triangleIndex)
        {
            // adding triangle i in the triangle shell of all edges
            for (unsigned int indexEdge = 0; indexEdge<3 ; ++indexEdge)
            {
                nbr_edgesInTriangle[ m_edgesInTriangle[triangleIndex][indexEdge] ]++;
            }
        }

        for (unsigned int indexEdge = 0; indexEdge < nbrEdges; ++indexEdge)
        {

            nbrEdgesInTriangle = m_trianglesAroundEdge[indexEdge].size();

            //Test on the size of the shell: If size differ from the previous vector, this means the shell is not well fill.
            if (nbrEdgesInTriangle != nbr_edgesInTriangle[indexEdge])
            {
                msg_error() << "Manifold_triangle_edge_shell: Triangles are missing in the shell around the edge: " << indexEdge;
                ret = false;
            }



            /*
             Test if there is at least 1 and not more than 2 triangles adjacent to each edge.
             Test if edges are well oriented in the triangles: in the first triangle of m_trianglesAroundEdge, vertices of the
             correspondant edge are in oriented in counterclockwise direction in this triangle.
             And in the clockwise direction in the second triangle (if this one exist)
             */
            if (nbrEdgesInTriangle > 0)
            {
                vertexTriangle = m_triangle[ m_trianglesAroundEdge[indexEdge][0] ];
                vertexInTriangle = getVertexIndexInTriangle(vertexTriangle, m_edge[indexEdge][0] );

                if ( m_edge[indexEdge][1] != vertexTriangle[ (vertexInTriangle+1)%3 ])
                {
                    msg_error() << "Manifold_triangle_edge_shell: Edge " << indexEdge << " is not well oriented regarding the first triangle of the shell.";
                    ret = false;
                }
            }
            else
            {
                msg_error() << "Manifold_triangle_edge_shell: Edge " << indexEdge << " is not part of a triangle.";
                ret = false;
            }

            if (nbrEdgesInTriangle == 2)
            {
                vertexTriangle = m_triangle[ m_trianglesAroundEdge[indexEdge][1] ];
                vertexInTriangle = getVertexIndexInTriangle(vertexTriangle, m_edge[indexEdge][0] );

                if ( m_edge[indexEdge][1] != vertexTriangle[ (vertexInTriangle+2)%3 ])
                {
                    msg_error() << "Manifold_triangle_edge_shell: Edge " << indexEdge << " is not well oriented regarding the second triangle of the shell.";
                    ret = false;
                }

            }
            else if (nbrEdgesInTriangle >2 )
            {
                msg_error() << "Manifold_triangle_edge_shell: Edge " << indexEdge << " has more than two adjacent triangles.";
                ret = false;
            }
        }
    }

    return ret && TriangleSetTopologyContainer::checkTopology();
}



void ManifoldTriangleSetTopologyContainer::clear()
{
    TriangleSetTopologyContainer::clear();
}


void ManifoldTriangleSetTopologyContainer::init()
{
    TriangleSetTopologyContainer::init();
}



void ManifoldTriangleSetTopologyContainer::createEdgeSetArray()
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

            const Edge e = ((v1<v2) ? Edge(v1,v2) : Edge(v2,v1));
            const Edge real_e = Edge(v1,v2);

            if(edgeMap.find(e) == edgeMap.end())
            {
                // edge not in edgeMap so create a new one
                const EdgeID edgeIndex = EdgeID(edgeMap.size());
                edgeMap[e] = edgeIndex;
                m_edge.push_back(real_e);
            }

        }
    }

}



void ManifoldTriangleSetTopologyContainer::createEdgesAroundVertexArray()
{

    if(!hasEdges())	// this method should only be called when edges exist
        createEdgeSetArray();

    if(hasEdgesAroundVertex())
    {
        clearEdgesAroundVertex();
    }


    //Number of different elements needed for this function
    const size_t nbrVertices = getNbPoints();
    const size_t nbrEdges = getNumberOfEdges();
    const size_t nbrTriangles = getNumberOfTriangles();

    //Temporary objects
    Triangle vertexTriangle;
    EdgesInTriangle edgeTriangle;
    int cpt;
    int firstVertex;
    int nextVertex;

    //Temporary containers
    sofa::helper::vector< std::multimap<unsigned int, unsigned int> > map_Adjacents;
    sofa::helper::vector< std::map<unsigned int, unsigned int> > map_NextEdgeVertex;
    sofa::helper::vector< std::map<unsigned int, unsigned int> > map_OppositeEdgeVertex;

    std::multimap<unsigned int, unsigned int>::iterator it_multimap;
    std::map<unsigned int, unsigned int>::iterator it_map;

    helper::ReadAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;

    m_edgesAroundVertex.resize(nbrVertices);
    map_Adjacents.resize(nbrVertices);
    map_NextEdgeVertex.resize(nbrVertices);
    map_OppositeEdgeVertex.resize(nbrVertices);


    /*	Creation of the differents maps: For each vertex i of each triangles:
     - map_NextEdgeVertex: key = vertex i+1, value = Edge i+2
     - map_OppositeEdgeVertex: key = vertex i+1, value = vertex i+2
     - map_Adjacents: key = vertex i+1 et i+2, value = Edge i	*/
    for (unsigned int triangleIndex = 0; triangleIndex < nbrTriangles; triangleIndex++)
    {
        vertexTriangle = getTriangleArray()[triangleIndex];
        edgeTriangle = getEdgesInTriangle(triangleIndex);

        for (unsigned int i=0; i<3; ++i)
        {
            map_NextEdgeVertex[vertexTriangle[i]].insert(std::pair<unsigned int,unsigned int> (vertexTriangle[(i+1)%3], edgeTriangle[(i+2)%3]));
            map_OppositeEdgeVertex[vertexTriangle[i]].insert(std::pair<unsigned int,unsigned int> (vertexTriangle[(i+1)%3], vertexTriangle[(i+2)%3]));

            map_Adjacents[vertexTriangle[i]].insert(std::pair<unsigned int,unsigned int> (vertexTriangle[(i+1)%3], edgeTriangle[i]));
            map_Adjacents[vertexTriangle[i]].insert(std::pair<unsigned int,unsigned int> (vertexTriangle[(i+2)%3], edgeTriangle[i]));
        }
    }


    for (unsigned int vertexIndex = 0; vertexIndex < nbrVertices; vertexIndex++)
    {
        it_map = map_OppositeEdgeVertex[vertexIndex].begin();
        firstVertex = (*it_map).first;

        for (it_multimap = map_Adjacents[vertexIndex].begin(); it_multimap != map_Adjacents[vertexIndex].end(); it_multimap++)
        {
            cpt = (int)map_Adjacents[vertexIndex].count((*it_multimap).first);

            if( cpt > 2)
            {
                msg_error() << "createEdgesAroundVertexArray The mapping is not manifold. In the neighborhood of the vertex: " << vertexIndex;
                msg_error() << "There are " << cpt << " edges connected to the vertex: " << (*it_multimap).first;
            }
            else if ( cpt == 1)
            {
                it_map = map_OppositeEdgeVertex[vertexIndex].find( (*it_multimap).first );
                if(it_map != map_OppositeEdgeVertex[vertexIndex].end())
                {
                    firstVertex = (*it_map).first;
                }
            }
        }

        m_edgesAroundVertex[vertexIndex].push_back(map_NextEdgeVertex[vertexIndex][firstVertex]);
        nextVertex = (*(it_map = map_OppositeEdgeVertex[vertexIndex].find(firstVertex))).second;

        for (unsigned int indexEdge = 1; indexEdge < map_OppositeEdgeVertex[vertexIndex].size(); indexEdge++)
        {
            m_edgesAroundVertex[vertexIndex].push_back(map_NextEdgeVertex[vertexIndex][nextVertex]);
            nextVertex = (*(it_map = map_OppositeEdgeVertex[vertexIndex].find(nextVertex))).second;
            //std::cout << "nextVertex: " << nextVertex;
            //si different de fin
        }

        if (nextVertex != firstVertex)
        {
            const Edge lastEdge = Edge(nextVertex,vertexIndex);

            for ( unsigned int i = 0; i < nbrEdges; ++i)
            {
                if( m_edge[i][0] == lastEdge[0] && m_edge[i][1] == lastEdge[1])
                {
                    m_edgesAroundVertex[vertexIndex].push_back(i);
                    break;
                }
            }
        }
    }

    map_Adjacents.clear();
    map_NextEdgeVertex.clear();
    map_OppositeEdgeVertex.clear();
}



void ManifoldTriangleSetTopologyContainer::createTrianglesAroundVertexArray ()
{
    if(!hasTriangles()) // this method should only be called when triangles exist
    {
        createTriangleSetArray();
    }

    if(hasTrianglesAroundVertex())
    {
        clearTrianglesAroundVertex();
    }

    //Number of different elements needed for this function
    const size_t nbrVertices = getNbPoints();
    const size_t nbrTriangles = getNumberOfTriangles();

    //Temporary objects
    Triangle vertexTriangle;
    unsigned int cpt;
    unsigned int firstVertex;

    //Temporary containers
    sofa::helper::vector< std::map<unsigned int, unsigned int> > map_Triangles;
    sofa::helper::vector< std::map<unsigned int, unsigned int> > map_NextVertex;
    sofa::helper::vector< std::map<unsigned int, unsigned int> > map_PreviousVertex;

    std::map<unsigned int, unsigned int>::iterator it1;
    std::map<unsigned int, unsigned int>::iterator it2;

    m_trianglesAroundVertex.resize(nbrVertices);
    map_Triangles.resize(nbrVertices);
    map_NextVertex.resize(nbrVertices);
    map_PreviousVertex.resize(nbrVertices);

    /*	Creation of the differents maps: For each vertex i of each triangles:
     - map_Triangles: key = vertex i+1, value = index triangle
     - map_Nextvertex: key = vertex i+1, value = vertex i+2
     - map_PreviousVertex: key = vertex i+2, value = vertex i+1	*/
    for (unsigned int triangleIndex = 0; triangleIndex < nbrTriangles; ++triangleIndex)
    {
        vertexTriangle = getTriangleArray()[triangleIndex];

        for (unsigned int i=0; i<3; ++i)
        {
            map_Triangles[vertexTriangle[i]].insert(std::pair<unsigned int,unsigned int> (vertexTriangle[(i+1)%3], triangleIndex)); //multi

            map_NextVertex[vertexTriangle[i]].insert(std::pair<unsigned int,unsigned int> (vertexTriangle[(i+1)%3], vertexTriangle[(i+2)%3]));
            map_PreviousVertex[vertexTriangle[i]].insert(std::pair<unsigned int,unsigned int> (vertexTriangle[(i+2)%3], vertexTriangle[(i+1)%3]));
        }
    }

    // General loop for m_trianglesAroundVertex creation
    for (unsigned int vertexIndex = 0; vertexIndex < nbrVertices; ++vertexIndex)
    {
        it1 = map_NextVertex[vertexIndex].begin();
        firstVertex = (*it1).first;

        for (it1 = map_NextVertex[vertexIndex].begin(); it1 != map_NextVertex[vertexIndex].end(); ++it1)
        {
            it2 = map_PreviousVertex[vertexIndex].find((*it1).first);

            if (it2 == map_PreviousVertex[vertexIndex].end()) //it2 didn't find the it1 correspondant element in his map, means it's a border
            {
                firstVertex = (*it1).first;
                break;
            }//else we are not on a border. we keep the initialised value for firstVertex
        }
        m_trianglesAroundVertex[vertexIndex].push_back(map_Triangles[vertexIndex][firstVertex]);
        cpt=1;

        for (unsigned int i = 1; i < map_NextVertex[vertexIndex].size(); ++i)
        {
            it2 = map_NextVertex[vertexIndex].find(firstVertex);

            if (((*it2).first == firstVertex) && (it2 == map_NextVertex[vertexIndex].end()))
            {
                // Contour has been done without reaching the end of the map
                break;
            }
            firstVertex = (*it2).second;

            m_trianglesAroundVertex[vertexIndex].push_back(map_Triangles[vertexIndex][firstVertex]);
            cpt++;
        }

        if (cpt != map_Triangles[vertexIndex].size())
        {
            msg_error() << "CreateEdgesAroundVertexArray The mapping is not manifold. There is a wrong connection between triangles adjacent to the vertex: "<< vertexIndex;
        }
    }
    map_Triangles.clear();
    map_NextVertex.clear();
    map_PreviousVertex.clear();
}



void ManifoldTriangleSetTopologyContainer::createTrianglesAroundEdgeArray()
{

    if(!hasTriangles()) // this method should only be called when triangles exist
    {
        createTriangleSetArray();
    }

    if(!hasEdges()) // this method should only be called when edges exist
    {
        createEdgeSetArray();
    }

    if(!hasEdgesInTriangle())
        createEdgesInTriangleArray();

    if(hasTrianglesAroundEdge())
        clearTrianglesAroundEdge();



    //Number of different elements needed for this function
    const size_t nbrEdges = getNumberOfEdges();
    const size_t nbrTriangles = getNumberOfTriangles();

    //Temporary objects
    Triangle vertexTriangle;
    size_t cpt;
    int firstVertex;
    int vertexInTriangle;

    //Temporary containers
    std::multimap<unsigned int, unsigned int> map_edgesInTriangle;
    std::multimap<unsigned int, unsigned int>::iterator it;
    std::pair< std::multimap <unsigned int, unsigned int>::iterator, std::multimap <unsigned int, unsigned int>::iterator> pair_equal_range;
    helper::ReadAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;
    helper::ReadAccessor< Data< sofa::helper::vector<Triangle> > > m_triangle = d_triangle;

    m_trianglesAroundEdge.resize(nbrEdges);


    for (unsigned int triangleIndex = 0; triangleIndex < nbrTriangles; ++triangleIndex)
    {
        // adding triangle i in the triangle shell of all edges
        for (unsigned int indexEdge = 0; indexEdge<3 ; ++indexEdge)
        {
            map_edgesInTriangle.insert(std::pair < unsigned int, unsigned int> (m_edgesInTriangle[triangleIndex][indexEdge], triangleIndex));
        }
    }


    for (unsigned int indexEdge = 0; indexEdge < nbrEdges; indexEdge++)
    {
        cpt = map_edgesInTriangle.count(indexEdge);

        if (cpt > 2)
        {
            msg_error() << "createTrianglesAroundEdgeArray The mapping is not manifold. There are more than 2 triangles adjacents to the Edge: " << indexEdge;

            //Even if this structure is not Manifold, we chosed to fill the shell with all the triangles:
            pair_equal_range = map_edgesInTriangle.equal_range(indexEdge);

            for (it = pair_equal_range.first; it != pair_equal_range.second; ++it)
                m_trianglesAroundEdge[indexEdge].push_back((*it).second);
        }
        else if (cpt == 1)
        {
            it = map_edgesInTriangle.find(indexEdge);
            m_trianglesAroundEdge[indexEdge].push_back((*it).second);
        }
        else if (cpt == 2)
        {
            pair_equal_range = map_edgesInTriangle.equal_range(indexEdge);
            it = pair_equal_range.first;

            firstVertex = m_edge[indexEdge][0];

            vertexTriangle = m_triangle[(*it).second];
            vertexInTriangle = getVertexIndexInTriangle (vertexTriangle, firstVertex);

            if ((unsigned int)m_edge[indexEdge][1] == (unsigned int)vertexTriangle[(vertexInTriangle+1)%3])
            {
                m_trianglesAroundEdge[indexEdge].push_back((*it).second);

                it++;
                m_trianglesAroundEdge[indexEdge].push_back((*it).second);
            }
            else
            {
                it++;
                m_trianglesAroundEdge[indexEdge].push_back((*it).second);
                it--;
                m_trianglesAroundEdge[indexEdge].push_back((*it).second);
            }
        }
    }
}




int ManifoldTriangleSetTopologyContainer::getNextTrianglesAroundVertex(PointID vertexIndex, TriangleID triangleIndex)
{

    if(!hasTrianglesAroundVertex())	// this method should only be called when the shell array exists
    {
        createTrianglesAroundVertexArray();
    }


    if( vertexIndex >= m_trianglesAroundVertex.size())
    {
        msg_error() << "GetNextTrianglesAroundVertex Vertex index out of bounds.";
        return -2;
    }

    if( triangleIndex >= (d_triangle.getValue()).size())
    {
        msg_error() << "GetNextTrianglesAroundVertex Triangle index out of bounds.";
        return -2;
    }


    size_t nbrTriangle = m_trianglesAroundVertex[vertexIndex].size();

    for (unsigned int i = 0; i < nbrTriangle; ++i)
    {

        if ( m_trianglesAroundVertex[vertexIndex][i] == triangleIndex)
        {
            if ( i == nbrTriangle-1)
            {
                Triangle triangle1 = getTriangleArray()[m_trianglesAroundVertex[vertexIndex][nbrTriangle-1]];
                Triangle triangle2 = getTriangleArray()[m_trianglesAroundVertex[vertexIndex][0]];


                if ( triangle1[(getVertexIndexInTriangle(triangle1, vertexIndex)+2)%3] != triangle2[(getVertexIndexInTriangle(triangle2, vertexIndex)+1)%3])
                {
                    msg_error() << "GetNextTrianglesAroundVertex No Triangle has been found. Input Triangle must belong to the border.";
                    return -1;
                }
                else
                {
                    return m_trianglesAroundVertex[vertexIndex][0];
                }

            }
            else
            {
                return m_trianglesAroundVertex[vertexIndex][i+1];
            }
        }

    }

    msg_error() << "GetNextTrianglesAroundVertex No Triangle has been returned.";

    return -2;
}




int ManifoldTriangleSetTopologyContainer::getPreviousTrianglesAroundVertex(PointID vertexIndex, TriangleID triangleIndex)
{

    if(!hasTrianglesAroundVertex())	// this method should only be called when the shell array exists
    {
        createTrianglesAroundVertexArray();
    }


    if( vertexIndex >= m_trianglesAroundVertex.size())
    {
        msg_error() << "GetPreviousTrianglesAroundVertex Vertex index out of bounds.";
        return -2;
    }

    if( triangleIndex >= (d_triangle.getValue()).size())
    {
        msg_error() << "GetPreviousTrianglesAroundVertex Triangle index out of bounds.";
        return -2;
    }


    size_t nbrTriangle = m_trianglesAroundVertex[vertexIndex].size();

    for (unsigned int i = 0; i < nbrTriangle; ++i)
    {

        if ( m_trianglesAroundVertex[vertexIndex][i] == triangleIndex)
        {
            if ( i == 0)
            {
                Triangle triangle1 = getTriangleArray()[m_trianglesAroundVertex[vertexIndex][nbrTriangle-1]];
                Triangle triangle2 = getTriangleArray()[m_trianglesAroundVertex[vertexIndex][0]];

                if ( triangle1[(getVertexIndexInTriangle(triangle1, vertexIndex)+2)%3] != triangle2[(getVertexIndexInTriangle(triangle2, vertexIndex)+1)%3])
                {
                    msg_error() << "GetPreviousTrianglesAroundVertex No Triangle has been found. Input Triangle must belong to the border.";
                    return -1;
                }
                else
                {
                    return m_trianglesAroundVertex[vertexIndex][nbrTriangle-1];
                }

            }
            else
            {
                return m_trianglesAroundVertex[vertexIndex][i-1];
            }
        }

    }


    msg_error() << "GetPreviousTrianglesAroundVertex No Triangle has been returned.";
    return -2;
}




int ManifoldTriangleSetTopologyContainer::getOppositeTrianglesAroundEdge(EdgeID edgeIndex, TriangleID triangleIndex)
{

    if(!hasTrianglesAroundEdge())	// this method should only be called when the shell array exists
    {
        createTrianglesAroundEdgeArray();
    }


    if (edgeIndex >= m_trianglesAroundEdge.size())
    {
        msg_error() << "GetOppositeTrianglesAroundEdge Edge Index out of bounds.";
        return -2;
    }

    if (triangleIndex >= (d_triangle.getValue()).size())
    {
        msg_error() << "GetNextTrianglesAroundVertex Triangle index out of bounds.";
        return -2;
    }



    if (m_trianglesAroundEdge[edgeIndex].size() > 2)
    {
        msg_error() << "GetOppositeTrianglesAroundEdge The mapping is not manifold. There are more than 2 triangles adjacents to the Edge: " << edgeIndex;
        return -2;
    }
    else if (m_trianglesAroundEdge[edgeIndex].size() == 1)
    {
        msg_error() << "GetOppositeTrianglesAroundEdge No triangle has been returned. Input Edge belongs to the border.";
        return -1;
    }
    else if (m_trianglesAroundEdge[edgeIndex][0] == triangleIndex)
    {
        return m_trianglesAroundEdge[edgeIndex][1];
    }
    else if (m_trianglesAroundEdge[edgeIndex][1] == triangleIndex)
    {
        return m_trianglesAroundEdge[edgeIndex][0];
    }


    msg_error() << "GetOppositeTrianglesAroundEdge No Triangle has been returned.";

    return -2;
}




int ManifoldTriangleSetTopologyContainer::getNextEdgesAroundVertex(PointID vertexIndex, EdgeID edgeIndex)
{

    helper::ReadAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;
    helper::ReadAccessor< Data< sofa::helper::vector<Triangle> > > m_triangle = d_triangle;

    if(!hasEdgesAroundVertex())	// this method should only be called when the shell array exists
    {
        createEdgesAroundVertexArray();
    }

    if( vertexIndex >= m_edgesAroundVertex.size())
    {
        msg_error() << "GetNextEdgesAroundVertex Vertex index out of bounds.";
        return -2;
    }
    else if( edgeIndex >= m_edge.size())
    {
        msg_error() << "GetNextEdgesAroundVertex Edge index out of bounds.";
        return -2;
    }

    unsigned int vertex;

    if (m_edge[edgeIndex][0] == vertexIndex)
        vertex = m_edge[edgeIndex][1];
    else if (m_edge[edgeIndex][1] == vertexIndex)
        vertex = m_edge[edgeIndex][0];
    else
    {
        msg_error() << "GetNextEdgesAroundVertex Input vertex does not belongs to input edge.";
        return -2;
    }


    size_t nbrEdge = m_edgesAroundVertex[vertexIndex].size();

    for (unsigned int i = 0; i < nbrEdge; ++i)
    {

        if (m_edgesAroundVertex[vertexIndex][i] == edgeIndex)
        {
            if (i == nbrEdge-1)
            {
                Triangle triangle;


                for (unsigned int j = 0; j<m_triangle.size(); ++j)
                {
                    triangle = getTriangleArray()[j];

                    for (unsigned int k = 0; k<3; ++k)
                    {

                        if (triangle[k] == vertexIndex)
                        {
                            if (triangle[(k+1)%3] == vertex)
                            {
                                return m_edgesAroundVertex[vertexIndex][0];
                            }
                        }

                    }
                }

                msg_error() << "GetNextEdgesAroundVertex No edge has been returned. Input Edge belongs to the border ";

                return -1;
            }
            else
            {
                return m_edgesAroundVertex[vertexIndex][i+1];
            }
        }

    }


    msg_error() << "GetNextEdgesAroundVertex No Edge has been returned.";

    return -2;
}





int ManifoldTriangleSetTopologyContainer::getPreviousEdgesAroundVertex(PointID vertexIndex, EdgeID edgeIndex)
{
    helper::ReadAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;
    helper::ReadAccessor< Data< sofa::helper::vector<Triangle> > > m_triangle = d_triangle;

    if(!hasEdgesAroundVertex())	// this method should only be called when the shell array exists
    {
        createEdgesAroundVertexArray();
    }

    if( vertexIndex >= m_edgesAroundVertex.size())
    {
        msg_error() << "GetPreviousEdgesAroundVertex Vertex index out of bounds.";
        return -2;
    }
    else if( edgeIndex >= m_edge.size())
    {
        msg_error() << "GetPreviousEdgesAroundVertex Edge index out of bounds.";
        return -2;
    }

    unsigned int vertex;

    if (m_edge[edgeIndex][0] == vertexIndex)
        vertex = m_edge[edgeIndex][1];
    else if (m_edge[edgeIndex][1] == vertexIndex)
        vertex = m_edge[edgeIndex][0];
    else
    {
        msg_error() << "GetPreviousEdgesAroundVertex Input vertex does not belongs to input edge.";
        return -2;
    }


    size_t nbrEdge = m_edgesAroundVertex[vertexIndex].size();

    for (unsigned int i = 0; i < nbrEdge; ++i)
    {
        if (m_edgesAroundVertex[vertexIndex][i] == edgeIndex)
        {
            if (i == 0)
            {
                Triangle triangle;


                for (unsigned int j = 0; j<m_triangle.size(); ++j)
                {
                    triangle = getTriangleArray()[j];

                    for (unsigned int k = 0; k<3; ++k)
                    {

                        if (triangle[k] == vertexIndex)
                        {
                            if (triangle[(k+2)%3] == vertex)
                            {
                                return m_edgesAroundVertex[vertexIndex][nbrEdge-1];
                            }
                        }

                    }

                }

                msg_error() << "GetPreviousEdgesAroundVertex No edge has been returned. Input Edge belongs to the border ";

                return -1;
            }
            else
            {
                return m_edgesAroundVertex[vertexIndex][i-1];
            }

        }
    }


    msg_error() << "GetPreviousEdgesAroundVertex No Edge has been returned.";
    return -2;
}

int ManifoldTriangleSetTopologyContainer::getEdgeTriangleOrientation(const Triangle& f, const Edge& e)
{
    unsigned i = 0;
    for(; i < 3; ++i)
    {
        if(e[0] == f[i] && e[1] == f[(i+1)%3])
            return 1;
        if(e[0] == f[i] && e[1] == f[(i+2)%3])
            return -1;
    }
    return 0;
}

sofa::helper::vector< TriangleID > &ManifoldTriangleSetTopologyContainer::getTrianglesAroundEdgeForModification(const index_type i)
{

    if(!hasTrianglesAroundEdge())	// this method should only be called when the shell array exists
    {
        createTrianglesAroundEdgeArray();
    }

    if( i >= m_trianglesAroundEdge.size())
    {
        createTrianglesAroundEdgeArray();
    }

    return m_trianglesAroundEdge[i];
}



sofa::helper::vector< TriangleID > &ManifoldTriangleSetTopologyContainer::getTrianglesAroundVertexForModification(const index_type i)
{

    if(!hasTrianglesAroundVertex())	// this method should only be called when the shell array exists
    {
        createTrianglesAroundVertexArray();
    }

    if( i >= m_trianglesAroundVertex.size())
    {
        createTrianglesAroundVertexArray();
    }

    return m_trianglesAroundVertex[i];
}



sofa::helper::vector< EdgeID > &ManifoldTriangleSetTopologyContainer::getEdgesAroundVertexForModification(const index_type i)
{

    if(!hasEdgesAroundVertex())	// this method should only be called when the shell array exists
    {
        createEdgesAroundVertexArray();
    }

    if( i >= m_edgesAroundVertex.size())
    {
        createEdgesAroundVertexArray();
    }

    return m_edgesAroundVertex[i];
}


} // namespace topology

} // namespace component

} // namespace sofa
