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
#include <sofa/component/topology/ManifoldTriangleSetTopologyContainer.h>
#include <sofa/core/ObjectFactory.h>
//#include <sofa/component/container/MeshLoader.h>

namespace sofa
{

namespace component
{

namespace topology
{
using namespace std;
using namespace sofa::defaulttype;


SOFA_DECL_CLASS(ManifoldTriangleSetTopologyContainer)
int ManifoldTriangleSetTopologyContainerClass = core::RegisterObject("Manifold Triangle set topology container")
        .add< ManifoldTriangleSetTopologyContainer >()
        ;



ManifoldTriangleSetTopologyContainer::ManifoldTriangleSetTopologyContainer()
    : TriangleSetTopologyContainer()
{
}



ManifoldTriangleSetTopologyContainer::ManifoldTriangleSetTopologyContainer(const sofa::helper::vector< Triangle > &triangles )
    : TriangleSetTopologyContainer(triangles)
{
}



bool ManifoldTriangleSetTopologyContainer::checkTopology() const
{
#ifndef NDEBUG
    bool ret = true;

    //Test the shell m_triangleVertexShell
    if(hasTriangleVertexShell())
    {
        //Number of different elements needed for this function
        const unsigned int nbrVertices = getNbPoints();
        const unsigned int nbrTriangles = getNumberOfTriangles();

        //Temporary objects
        Triangle vertexTriangle;
        unsigned int firstVertex;
        unsigned int vertex, vertexNext;

        //Temporary containers
        sofa::helper::vector< std::map<unsigned int, unsigned int> > map_Triangles;
        std::map<unsigned int, unsigned int>::iterator it1;
        map_Triangles.resize(nbrVertices);

        //Fill temporary an other TriangleVertexShellarray
        for (unsigned int triangleIndex = 0; triangleIndex < nbrTriangles; ++triangleIndex)
        {

            vertexTriangle = m_triangle[triangleIndex];

            for (unsigned int i=0; i<3; ++i)
            {
                // Test if there are several triangles with the same edge in the same order.
                it1 = map_Triangles[vertexTriangle[i]].find(vertexTriangle[(i+1)%3]);
                if (it1 != map_Triangles[vertexTriangle[i]].end())
                {
                    std::cout << "*** CHECK FAILED: In Manifold_triangle_vertex_shell: Bad connection or triangle orientation between vertices: ";
                    std::cout << vertexTriangle[i] << " and " << vertexTriangle[(i+1)%3] << std::endl;

                    ret = false;
                }

                map_Triangles[vertexTriangle[i]].insert(std::pair<unsigned int,unsigned int> (vertexTriangle[(i+1)%3], triangleIndex)); //multi
            }
        }


        //Check if ManifoldTriangleVertexShellArray is contiguous and if no triangles are missing in the shell
        for (unsigned int vertexIndex = 0; vertexIndex < nbrVertices; ++vertexIndex)
        {

            //Test on the size of the shell: If size differ from the previous map, this means the shell is not well fill.
            if ( m_triangleVertexShell[vertexIndex].size() != map_Triangles[vertexIndex].size())
            {
                std::cout << "*** CHECK FAILED: In Manifold_triangle_vertex_shell: Triangles are missing in the shell around the vertex: ";
                std::cout << vertexIndex << std::endl;

                ret = false;
            }

            vertexTriangle = m_triangle[ m_triangleVertexShell[vertexIndex][0] ];
            firstVertex = vertexTriangle[ ( getVertexIndexInTriangle(vertexTriangle, vertexIndex)+1 )%3 ];

            //For each vertex, test if the triangle adjacent are fill in the right contiguous way.
            //Triangles should be adjacent one to the next other in counterclockwise direction.
            for (unsigned int triangleIndex = 0; triangleIndex < m_triangleVertexShell[vertexIndex].size()-1; ++triangleIndex)
            {
                vertexTriangle = m_triangle[ m_triangleVertexShell[vertexIndex][triangleIndex] ];
                vertex = vertexTriangle[ ( getVertexIndexInTriangle(vertexTriangle, vertexIndex)+2 )%3 ];

                vertexTriangle = m_triangle[ m_triangleVertexShell[vertexIndex][triangleIndex+1] ];
                vertexNext = vertexTriangle[ ( getVertexIndexInTriangle(vertexTriangle, vertexIndex)+1 )%3 ];

                if (vertex != vertexNext)
                {
                    std::cout << "*** CHECK FAILED: In Manifold_triangle_vertex_shell: Triangles are not contiguous or not well connected around the vertex: ";
                    std::cout << vertexIndex << std::endl;

                    ret = false;
                    break;
                }
            }
        }
    }


    //Test the shell m_triangleEdgeShell
    if (hasTriangleEdgeShell())
    {

        //Number of different elements needed for this function
        const unsigned int nbrEdges = getNumberOfEdges();
        const unsigned int nbrTriangles = getNumberOfTriangles();

        //Temporary objects
        Triangle vertexTriangle;
        unsigned int nbrTriangleEdge;
        unsigned int vertexInTriangle;

        //Temporary containers
        std::vector <unsigned int> nbr_triangleEdge;
        nbr_triangleEdge.resize(nbrEdges);

        //Create a vector containing the number of triangle adjacent to each edge
        for (unsigned int triangleIndex = 0; triangleIndex < nbrTriangles; ++triangleIndex)
        {
            // adding triangle i in the triangle shell of all edges
            for (unsigned int indexEdge = 0; indexEdge<3 ; ++indexEdge)
            {
                nbr_triangleEdge[ m_triangleEdge[triangleIndex][indexEdge] ]++;
            }
        }

        for (unsigned int indexEdge = 0; indexEdge < nbrEdges; ++indexEdge)
        {

            nbrTriangleEdge = m_triangleEdgeShell[indexEdge].size();

            //Test on the size of the shell: If size differ from the previous vector, this means the shell is not well fill.
            if (nbrTriangleEdge != nbr_triangleEdge[indexEdge])
            {
                std::cout << "*** CHECK FAILED: In Manifold_triangle_edge_shell: Triangles are missing in the shell around the edge: ";
                std::cout << indexEdge << std::endl;

                ret = false;
            }



            /*
              Test if there is at least 1 and not more than 2 triangles adjacent to each edge.
              Test if edges are well oriented in the triangles: in the first triangle of m_triangleEdgeShell, vertices of the
              correspondant edge are in oriented in counterclockwise direction in this triangle.
              And in the clockwise direction in the second triangle (if this one exist)
            */
            if (nbrTriangleEdge > 0)
            {
                vertexTriangle = m_triangle[ m_triangleEdgeShell[indexEdge][0] ];
                vertexInTriangle = getVertexIndexInTriangle(vertexTriangle, m_edge[indexEdge][0] );

                if ( m_edge[indexEdge][1] != vertexTriangle[ (vertexInTriangle+1)%3 ])
                {
                    std::cout << "*** CHECK FAILED: In Manifold_triangle_edge_shell: Edge ";
                    std::cout << indexEdge << " is not well oriented regarding the first triangle of the shell";

                    ret = false;
                }
            }
            else
            {
                std::cout << "*** CHECK FAILED: In Manifold_triangle_edge_shell: Edge ";
                std::cout << indexEdge << " is not part of a triangle";

                ret = false;
            }

            if (nbrTriangleEdge == 2)
            {
                vertexTriangle = m_triangle[ m_triangleEdgeShell[indexEdge][1] ];
                vertexInTriangle = getVertexIndexInTriangle(vertexTriangle, m_edge[indexEdge][0] );

                if ( m_edge[indexEdge][1] != vertexTriangle[ (vertexInTriangle+2)%3 ])
                {
                    std::cout << "*** CHECK FAILED: In Manifold_triangle_edge_shell: Edge ";
                    std::cout << indexEdge << " is not well oriented regarding the second triangle of the shell";

                    ret = false;
                }

            }
            else if (nbrTriangleEdge >2 )
            {
                std::cout << "*** CHECK FAILED: In Manifold_triangle_edge_shell: Edge ";
                std::cout << indexEdge << " has more than two adjacent triangles";

                ret = false;
            }
        }
    }


    return ret && TriangleSetTopologyContainer::checkTopology();
#else
    return true;
#endif
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

    d_edge.beginEdit();

    if(!hasTriangles()) // this method should only be called when triangles exist
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyContainer::createEdgeSetArray] triangle array is empty." << std::endl;
#endif
        createTriangleSetArray();
    }

    if(hasEdges())
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyContainer::createEdgeSetArray] edge array is not empty." << std::endl;
#endif

        // clear edges and all shells that depend on edges
        EdgeSetTopologyContainer::clear();

        if(hasTriangleEdges())
            clearTriangleEdges();

        if(hasTriangleEdgeShell())
            clearTriangleEdgeShell();
    }


    // create a temporary map to find redundant edges
    std::map<Edge, unsigned int> edgeMap;

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
                const int edgeIndex = edgeMap.size();
                edgeMap[e] = edgeIndex;
                m_edge.push_back(real_e);
            }

        }
    }

    d_edge.endEdit();
}



void ManifoldTriangleSetTopologyContainer::createEdgeVertexShellArray()
{

    if(!hasEdges())	// this method should only be called when edges exist
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyContainer::createEdgeVertexShellArray] edge array is empty." << std::endl;
#endif

        createEdgeSetArray();
    }

    if(hasEdgeVertexShell())
    {
        clearEdgeVertexShell();
    }


    //Number of different elements needed for this function
    const unsigned int nbrVertices = getNbPoints();
    const unsigned int nbrEdges = getNumberOfEdges();
    const unsigned int nbrTriangles = getNumberOfTriangles();

    //Temporary objects
    Triangle vertexTriangle;
    TriangleEdges edgeTriangle;
    int cpt;
    int firstVertex;
    int nextVertex;

    //Temporary containers
    sofa::helper::vector< std::multimap<unsigned int, unsigned int> > map_Adjacents;
    sofa::helper::vector< std::map<unsigned int, unsigned int> > map_NextEdgeVertex;
    sofa::helper::vector< std::map<unsigned int, unsigned int> > map_OppositeEdgeVertex;

    std::multimap<unsigned int, unsigned int>::iterator it_multimap;
    std::map<unsigned int, unsigned int>::iterator it_map;


    m_edgeVertexShell.resize(nbrVertices);
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
        edgeTriangle = getEdgeTriangleShell(triangleIndex);

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
                //#ifndef NDEBUG
                std::cout << "Error. [ManifoldTriangleSetTopologyContainer::createEdgeVertexShellArray] The mapping is not manifold. ";
                std::cout << "In the neighborhood of the vertex: " << vertexIndex;
                std::cout << ". There are " << cpt << " edges connected to the vertex: " << (*it_multimap).first << std::endl;
                //#endif
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

        m_edgeVertexShell[vertexIndex].push_back(map_NextEdgeVertex[vertexIndex][firstVertex]);
        nextVertex = (*(it_map = map_OppositeEdgeVertex[vertexIndex].find(firstVertex))).second;

        for (unsigned int indexEdge = 1; indexEdge < map_OppositeEdgeVertex[vertexIndex].size(); indexEdge++)
        {
            m_edgeVertexShell[vertexIndex].push_back(map_NextEdgeVertex[vertexIndex][nextVertex]);
            nextVertex = (*(it_map = map_OppositeEdgeVertex[vertexIndex].find(nextVertex))).second;
            //std::cout << "nextVertex: " << nextVertex << std::endl;
            //si different de fin
        }

        if (nextVertex != firstVertex)
        {
            const Edge lastEdge = Edge(nextVertex,vertexIndex);

            for ( unsigned int i = 0; i < nbrEdges; ++i)
            {
                if( m_edge[i][0] == lastEdge[0] && m_edge[i][1] == lastEdge[1])
                {
                    m_edgeVertexShell[vertexIndex].push_back(i);
                    break;
                }
            }
        }
    }

    map_Adjacents.clear();
    map_NextEdgeVertex.clear();
    map_OppositeEdgeVertex.clear();
}



void ManifoldTriangleSetTopologyContainer::createTriangleVertexShellArray ()
{
    if(!hasTriangles()) // this method should only be called when triangles exist
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyContainer::createTriangleVertexShellArray] triangle array is empty." << std::endl;
#endif

        createTriangleSetArray();
    }

    if(hasTriangleVertexShell())
    {
        clearTriangleVertexShell();
    }

    //Number of different elements needed for this function
    const unsigned int nbrVertices = getNbPoints();
    const unsigned int nbrTriangles = getNumberOfTriangles();

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

    m_triangleVertexShell.resize(nbrVertices);
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

    // General loop for m_triangleVertexShell creation
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
        m_triangleVertexShell[vertexIndex].push_back(map_Triangles[vertexIndex][firstVertex]);
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

            m_triangleVertexShell[vertexIndex].push_back(map_Triangles[vertexIndex][firstVertex]);
            cpt++;
        }

        if (cpt != map_Triangles[vertexIndex].size())
        {
#ifndef NDEBUG
            std::cout << "Error. [ManifoldTriangleSetTopologyContainer::createEdgeVertexShellArray] The mapping is not manifold.";
            std::cout << "There is a wrong connection between triangles adjacent to the vertex: "<< vertexIndex << std::endl;
#endif
        }
    }
    map_Triangles.clear();
    map_NextVertex.clear();
    map_PreviousVertex.clear();
}



void ManifoldTriangleSetTopologyContainer::createTriangleEdgeShellArray()
{

    if(!hasTriangles()) // this method should only be called when triangles exist
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyContainer::createTriangleEdgeShellArray] Triangle array is empty." << std::endl;
#endif
        createTriangleSetArray();
    }

    if(!hasEdges()) // this method should only be called when edges exist
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyContainer::createTriangleEdgeShellArray] Edge array is empty." << std::endl;
#endif
        createEdgeSetArray();
    }

    if(!hasTriangleEdges())
        createTriangleEdgeArray();

    if(hasTriangleEdgeShell())
        clearTriangleEdgeShell();



    //Number of different elements needed for this function
    const unsigned int nbrEdges = getNumberOfEdges();
    const unsigned int nbrTriangles = getNumberOfTriangles();

    //Temporary objects
    Triangle vertexTriangle;
    int cpt;
    int firstVertex;
    int vertexInTriangle;

    //Temporary containers
    std::multimap<unsigned int, unsigned int> map_triangleEdge;
    std::multimap<unsigned int, unsigned int>::iterator it;
    std::pair< std::multimap <unsigned int, unsigned int>::iterator, std::multimap <unsigned int, unsigned int>::iterator> pair_equal_range;


    m_triangleEdgeShell.resize(nbrEdges);


    for (unsigned int triangleIndex = 0; triangleIndex < nbrTriangles; ++triangleIndex)
    {
        // adding triangle i in the triangle shell of all edges
        for (unsigned int indexEdge = 0; indexEdge<3 ; ++indexEdge)
        {
            map_triangleEdge.insert(std::pair < unsigned int, unsigned int> (m_triangleEdge[triangleIndex][indexEdge], triangleIndex));
        }
    }


    for (unsigned int indexEdge = 0; indexEdge < nbrEdges; indexEdge++)
    {
        cpt = map_triangleEdge.count(indexEdge);

        if (cpt > 2)
        {
#ifndef NDEBUG
            std::cout << "Error. [ManifoldTriangleSetTopologyContainer::createTriangleEdgeShellArray] The mapping is not manifold.";
            std::cout << "There are more than 2 triangles adjacents to the Edge: " << indexEdge << std::endl;
#endif

            //Even if this structure is not Manifold, we chosed to fill the shell with all the triangles:
            pair_equal_range = map_triangleEdge.equal_range(indexEdge);

            for (it = pair_equal_range.first; it != pair_equal_range.second; ++it)
                m_triangleEdgeShell[indexEdge].push_back((*it).second);
        }
        else if (cpt == 1)
        {
            it = map_triangleEdge.find(indexEdge);
            m_triangleEdgeShell[indexEdge].push_back((*it).second);
        }
        else if (cpt == 2)
        {
            pair_equal_range = map_triangleEdge.equal_range(indexEdge);
            it = pair_equal_range.first;

            firstVertex = m_edge[indexEdge][0];

            vertexTriangle = m_triangle[(*it).second];
            vertexInTriangle = getVertexIndexInTriangle (vertexTriangle, firstVertex);

            if ((unsigned int)m_edge[indexEdge][1] == (unsigned int)vertexTriangle[(vertexInTriangle+1)%3])
            {
                m_triangleEdgeShell[indexEdge].push_back((*it).second);

                it++;
                m_triangleEdgeShell[indexEdge].push_back((*it).second);
            }
            else
            {
                it++;
                m_triangleEdgeShell[indexEdge].push_back((*it).second);
                it--;
                m_triangleEdgeShell[indexEdge].push_back((*it).second);
            }
        }
    }
}




int ManifoldTriangleSetTopologyContainer::getNextTriangleVertexShell(PointID vertexIndex, TriangleID triangleIndex)
{

    if(!hasTriangleVertexShell())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyContainer::getNextTriangleVertexShell] Triangle vertex shell array is empty." << std::endl;
#endif

        createTriangleVertexShellArray();
    }


    if( vertexIndex >= m_triangleVertexShell.size())
    {
#ifndef NDEBUG
        std::cout << "Error. [ManifoldTriangleSetTopologyContainer::getNextTriangleVertexShell] Vertex index out of bounds." << std::endl;
#endif
        return -2;
    }

    if( triangleIndex >= m_triangle.size())
    {
#ifndef NDEBUG
        std::cout << "Error. [ManifoldTriangleSetTopologyContainer::getNextTriangleVertexShell] Triangle index out of bounds." << std::endl;
#endif
        return -2;
    }


    unsigned int nbrTriangle = m_triangleVertexShell[vertexIndex].size();

    for (unsigned int i = 0; i < nbrTriangle; ++i)
    {

        if ( m_triangleVertexShell[vertexIndex][i] == triangleIndex)
        {
            if ( i == nbrTriangle-1)
            {
                Triangle triangle1 = getTriangleArray()[m_triangleVertexShell[vertexIndex][nbrTriangle-1]];
                Triangle triangle2 = getTriangleArray()[m_triangleVertexShell[vertexIndex][0]];


                if ( triangle1[(getVertexIndexInTriangle(triangle1, vertexIndex)+2)%3] != triangle2[(getVertexIndexInTriangle(triangle2, vertexIndex)+1)%3])
                {
#ifndef NDEBUG
                    std::cout << "Warning. [ManifoldTriangleSetTopologyContainer::getNextTriangleVertexShell] No Triangle has been found. Input Triangle must belong to the border." << std::endl;
#endif
                    return -1;
                }
                else
                {
                    return m_triangleVertexShell[vertexIndex][0];
                }

            }
            else
            {
                return m_triangleVertexShell[vertexIndex][i+1];
            }
        }

    }

#ifndef NDEBUG
    std::cout << "Error. [ManifoldTriangleSetTopologyContainer::getNextTriangleVertexShell] No Triangle has been returned." << std::endl;
#endif
    return -2;
}




int ManifoldTriangleSetTopologyContainer::getPreviousTriangleVertexShell(PointID vertexIndex, TriangleID triangleIndex)
{

    if(!hasTriangleVertexShell())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyContainer::getPreviousTriangleVertexShell] Triangle vertex shell array is empty." << std::endl;
#endif

        createTriangleVertexShellArray();
    }


    if( vertexIndex >= m_triangleVertexShell.size())
    {
#ifndef NDEBUG
        std::cout << "Error. [ManifoldTriangleSetTopologyContainer::getPreviousTriangleVertexShell] Vertex index out of bounds." << std::endl;
#endif
        return -2;
    }

    if( triangleIndex >= m_triangle.size())
    {
#ifndef NDEBUG
        std::cout << "Error. [ManifoldTriangleSetTopologyContainer::getPreviousTriangleVertexShell] Triangle index out of bounds." << std::endl;
#endif
        return -2;
    }


    unsigned int nbrTriangle = m_triangleVertexShell[vertexIndex].size();

    for (unsigned int i = 0; i < nbrTriangle; ++i)
    {

        if ( m_triangleVertexShell[vertexIndex][i] == triangleIndex)
        {
            if ( i == 0)
            {
                Triangle triangle1 = getTriangleArray()[m_triangleVertexShell[vertexIndex][nbrTriangle-1]];
                Triangle triangle2 = getTriangleArray()[m_triangleVertexShell[vertexIndex][0]];

                if ( triangle1[(getVertexIndexInTriangle(triangle1, vertexIndex)+2)%3] != triangle2[(getVertexIndexInTriangle(triangle2, vertexIndex)+1)%3])
                {
#ifndef NDEBUG
                    std::cout << "Warning. [ManifoldTriangleSetTopologyContainer::getPreviousTriangleVertexShell] No Triangle has been found. Input Triangle must belong to the border." << std::endl;
#endif
                    return -1;
                }
                else
                {
                    return m_triangleVertexShell[vertexIndex][nbrTriangle-1];
                }

            }
            else
            {
                return m_triangleVertexShell[vertexIndex][i-1];
            }
        }

    }


#ifndef NDEBUG
    std::cout << "Error. [ManifoldTriangleSetTopologyContainer::getPreviousTriangleVertexShell] No Triangle has been returned." << std::endl;
#endif
    return -2;
}




int ManifoldTriangleSetTopologyContainer::getOppositeTriangleEdgeShell(EdgeID edgeIndex, TriangleID triangleIndex)
{

    if(!hasTriangleEdgeShell())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyContainer::getOppositeTriangleEdgeShell] Triangle edge shell array is empty." << std::endl;
#endif

        createTriangleEdgeShellArray();
    }


    if (edgeIndex >= m_triangleEdgeShell.size())
    {
#ifndef NDEBUG
        std::cout << "Error. [ManifoldTriangleSetTopologyContainer::getOppositeTriangleEdgeShell] Edge Index out of bounds." << std::endl;
#endif
        return -2;
    }

    if (triangleIndex >= m_triangle.size())
    {
#ifndef NDEBUG
        std::cout << "Error. [ManifoldTriangleSetTopologyContainer::getNextTriangleVertexShell] Triangle index out of bounds." << std::endl;
#endif
        return -2;
    }



    if (m_triangleEdgeShell[edgeIndex].size() > 2)
    {
#ifndef NDEBUG
        std::cout << "Error. [ManifoldTriangleSetTopologyContainer::getOppositeTriangleEdgeShell] The mapping is not manifold.";
        std::cout << "There are more than 2 triangles adjacents to the Edge: " << edgeIndex << std::endl;
#endif
        return -2;
    }
    else if (m_triangleEdgeShell[edgeIndex].size() == 1)
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyContainer::getOppositeTriangleEdgeShell] No triangle has been returned. Input Edge belongs to the border." << std::endl;
#endif
        return -1;
    }
    else if (m_triangleEdgeShell[edgeIndex][0] == triangleIndex)
    {
        return m_triangleEdgeShell[edgeIndex][1];
    }
    else if (m_triangleEdgeShell[edgeIndex][1] == triangleIndex)
    {
        return m_triangleEdgeShell[edgeIndex][0];
    }


#ifndef NDEBUG
    std::cout << "Error. [ManifoldTriangleSetTopologyContainer::getOppositeTriangleEdgeShell] No Triangle has been returned." << std::endl;
#endif
    return -2;
}




int ManifoldTriangleSetTopologyContainer::getNextEdgeVertexShell(PointID vertexIndex, EdgeID edgeIndex)
{

    if(!hasEdgeVertexShell())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyContainer::getNextEdgeVertexShell] Edge vertex shell array is empty." << std::endl;
#endif

        createEdgeVertexShellArray();
    }

    if( vertexIndex >= m_edgeVertexShell.size())
    {
#ifndef NDEBUG
        std::cout << "Error. [ManifoldTriangleSetTopologyContainer::getNextEdgeVertexShell] Vertex index out of bounds." << std::endl;
#endif
        return -2;
    }
    else if( edgeIndex >= m_edge.size())
    {
#ifndef NDEBUG
        std::cout << "Error. [ManifoldTriangleSetTopologyContainer::getNextEdgeVertexShell] Edge index out of bounds." << std::endl;
#endif
        return -2;
    }

    unsigned int vertex;

    if (m_edge[edgeIndex][0] == vertexIndex)
        vertex = m_edge[edgeIndex][1];
    else if (m_edge[edgeIndex][1] == vertexIndex)
        vertex = m_edge[edgeIndex][0];
    else
    {
#ifndef NDEBUG
        std::cout << "Error. [ManifoldTriangleSetTopologyContainer::getNextEdgeVertexShell] Input vertex does not belongs to input edge." << std::endl;
#endif
        return -2;
    }


    unsigned int nbrEdge = m_edgeVertexShell[vertexIndex].size();

    for (unsigned int i = 0; i < nbrEdge; ++i)
    {

        if (m_edgeVertexShell[vertexIndex][i] == edgeIndex)
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
                                return m_edgeVertexShell[vertexIndex][0];
                            }
                        }

                    }
                }

#ifndef NDEBUG
                std::cout << "Warning. [ManifoldTriangleSetTopologyContainer::getNextEdgeVertexShell] No edge has been returned. Input Edge belongs to the border " << std::endl;
#endif
                return -1;
            }
            else
            {
                return m_edgeVertexShell[vertexIndex][i+1];
            }
        }

    }


#ifndef NDEBUG
    std::cout << "Error. [ManifoldTriangleSetTopologyContainer::getNextEdgeVertexShell] No Edge has been returned." << std::endl;
#endif
    return -2;
}





int ManifoldTriangleSetTopologyContainer::getPreviousEdgeVertexShell(PointID vertexIndex, EdgeID edgeIndex)
{

    if(!hasEdgeVertexShell())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyContainer::getPreviousEdgeVertexShell] Edge vertex shell array is empty." << std::endl;
#endif

        createEdgeVertexShellArray();
    }

    if( vertexIndex >= m_edgeVertexShell.size())
    {
#ifndef NDEBUG
        std::cout << "Error. [ManifoldTriangleSetTopologyContainer::getPreviousEdgeVertexShell] Vertex index out of bounds." << std::endl;
#endif

        return -2;
    }
    else if( edgeIndex >= m_edge.size())
    {
#ifndef NDEBUG
        std::cout << "Error. [ManifoldTriangleSetTopologyContainer::getPreviousEdgeVertexShell] Edge index out of bounds." << std::endl;
#endif

        return -2;
    }

    unsigned int vertex;

    if (m_edge[edgeIndex][0] == vertexIndex)
        vertex = m_edge[edgeIndex][1];
    else if (m_edge[edgeIndex][1] == vertexIndex)
        vertex = m_edge[edgeIndex][0];
    else
    {
#ifndef NDEBUG
        std::cout << "Error. [ManifoldTriangleSetTopologyContainer::getPreviousEdgeVertexShell] Input vertex does not belongs to input edge." << std::endl;
#endif

        return -2;
    }


    unsigned int nbrEdge = m_edgeVertexShell[vertexIndex].size();

    for (unsigned int i = 0; i < nbrEdge; ++i)
    {
        if (m_edgeVertexShell[vertexIndex][i] == edgeIndex)
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
                                return m_edgeVertexShell[vertexIndex][nbrEdge-1];
                            }
                        }

                    }

                }

#ifndef NDEBUG
                std::cout << "Warning. [ManifoldTriangleSetTopologyContainer::getPreviousEdgeVertexShell] No edge has been returned. Input Edge belongs to the border " << std::endl;
#endif
                return -1;
            }
            else
            {
                return m_edgeVertexShell[vertexIndex][i-1];
            }

        }
    }


#ifndef NDEBUG
    std::cout << "Error. [ManifoldTriangleSetTopologyContainer::getPreviousEdgeVertexShell] No Edge has been returned." << std::endl;
#endif
    return -2;
}


sofa::helper::vector< TriangleID > &ManifoldTriangleSetTopologyContainer::getTriangleEdgeShellForModification(const unsigned int i)
{

    if(!hasTriangleEdgeShell())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        sout << "Warning. [ManifoldTriangleSetTopologyContainer::getTriangleEdgeShellForModification] triangle edge shell array is empty." << endl;
#endif
        createTriangleEdgeShellArray();
    }

    if( i >= m_triangleEdgeShell.size())
    {
#ifndef NDEBUG
        sout << "Error. [ManifoldTriangleSetTopologyContainer::getTriangleEdgeShellForModification] index out of bounds." << endl;
#endif
        createTriangleEdgeShellArray();
    }

    return m_triangleEdgeShell[i];
}



sofa::helper::vector< TriangleID > &ManifoldTriangleSetTopologyContainer::getTriangleVertexShellForModification(const unsigned int i)
{

    if(!hasTriangleVertexShell())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyContainer::getTriangleVertexShellForModification] triangle vertex shell array is empty." << endl;
#endif
        createTriangleVertexShellArray();
    }

    if( i >= m_triangleVertexShell.size())
    {
#ifndef NDEBUG
        std::cout << "Error. [ManifoldTriangleSetTopologyContainer::getTriangleVertexShellForModification] index out of bounds." << std::endl;
#endif
        createTriangleVertexShellArray();
    }

    return m_triangleVertexShell[i];
}



sofa::helper::vector< EdgeID > &ManifoldTriangleSetTopologyContainer::getEdgeVertexShellForModification(const unsigned int i)
{

    if(!hasEdgeVertexShell())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        sout << "Warning. [ManifoldTriangleSetTopologyContainer::getEdgeVertexShellForModification] triangle vertex shell array is empty." << endl;
#endif
        createEdgeVertexShellArray();
    }

    if( i >= m_edgeVertexShell.size())
    {
#ifndef NDEBUG
        sout << "Error. [ManifoldTriangleSetTopologyContainer::getEdgeVertexShellForModification] index out of bounds." << endl;
#endif
        createEdgeVertexShellArray();
    }

    return m_edgeVertexShell[i];
}


} // namespace topology

} // namespace component

} // namespace sofa
