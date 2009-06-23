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
#include <sofa/component/topology/TriangleSetTopologyContainer.h>

#include <sofa/core/ObjectFactory.h>
//#include <sofa/component/container/MechanicalObject.inl>
//#include <sofa/helper/gl/glText.inl>

#include <sofa/helper/system/glut.h>

#include <sofa/component/container/MeshLoader.h>

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
    , d_triangle(initDataPtr(&d_triangle, &m_triangle, "triangles", "List of triangle indices"))
{

}

TriangleSetTopologyContainer::TriangleSetTopologyContainer(const sofa::helper::vector< Triangle > &triangles )
    : EdgeSetTopologyContainer()
    , m_triangle( triangles )
    , d_triangle(initDataPtr(&d_triangle, &m_triangle, "triangles", "List of triangle indices"))
{

    serr << getNbPoints() << "Constructor" << sendl;
    for (unsigned int i=0; i<m_triangle.size(); ++i)
    {
        for(unsigned int j=0; j<3; ++j)
        {
            int a = m_triangle[i][j];
            if (a >= getNbPoints())
                nbPoints.setValue(a+1);
        }
    }

    serr << "Constructor" << sendl;
}

void TriangleSetTopologyContainer::addTriangle( int a, int b, int c )
{
    d_triangle.beginEdit();
    m_triangle.push_back(Triangle(a,b,c));
    d_triangle.endEdit();
    if (a >= getNbPoints()) nbPoints.setValue(a+1);
    if (b >= getNbPoints()) nbPoints.setValue(b+1);
    if (c >= getNbPoints()) nbPoints.setValue(c+1);

    sout << "ADD TRIANGLE" << sendl;
}

void TriangleSetTopologyContainer::init()
{
    EdgeSetTopologyContainer::init();
}

void TriangleSetTopologyContainer::loadFromMeshLoader(sofa::component::container::MeshLoader* loader)
{
    // load points
    if (m_triangle.empty())
    {
        PointSetTopologyContainer::loadFromMeshLoader(loader);
        d_triangle.beginEdit();
        loader->getTriangles(m_triangle);
        d_triangle.endEdit();
    }
}

void TriangleSetTopologyContainer::createTriangleSetArray()
{
#ifndef NDEBUG
    sout << "Error. [TriangleSetTopologyContainer::createTriangleSetArray] This method must be implemented by a child topology." << endl;
#endif
}

void TriangleSetTopologyContainer::createTriangleVertexShellArray ()
{
    if(!hasTriangles()) // this method should only be called when triangles exist
    {
#ifndef NDEBUG
        sout << "Warning. [TriangleSetTopologyContainer::createTriangleVertexShellArray] triangle array is empty." << endl;
#endif
        createTriangleSetArray();
    }

    if(hasTriangleVertexShell())
    {
        clearTriangleVertexShell();
    }

    m_triangleVertexShell.resize( getNbPoints() );

    for (unsigned int i = 0; i < m_triangle.size(); ++i)
    {
        // adding edge i in the edge shell of both points
        for (unsigned int j=0; j<3; ++j)
            m_triangleVertexShell[ m_triangle[i][j]  ].push_back( i );
    }
}

void TriangleSetTopologyContainer::createTriangleEdgeShellArray ()
{
    if(!hasTriangles()) // this method should only be called when triangles exist
    {
#ifndef NDEBUG
        sout << "Warning. [TriangleSetTopologyContainer::createTriangleEdgeShellArray] triangle array is empty." << endl;
#endif
        createTriangleSetArray();
    }

    if(!hasEdges()) // this method should only be called when edges exist
    {
#ifndef NDEBUG
        sout << "Warning. [TriangleSetTopologyContainer::createTriangleEdgeShellArray] edge array is empty." << endl;
#endif
        createEdgeSetArray();
    }

    if(!hasTriangleEdges())
        createTriangleEdgeArray();

    const unsigned int numTriangles = getNumberOfTriangles();
    const unsigned int numEdges = getNumberOfEdges();

    if(hasTriangleEdgeShell())
    {
        clearTriangleEdgeShell();
    }

    m_triangleEdgeShell.resize( numEdges );

    for (unsigned int i = 0; i < numTriangles; ++i)
    {
        // adding triangle i in the triangle shell of all edges
        for (unsigned int j=0; j<3; ++j)
        {
            m_triangleEdgeShell[ m_triangleEdge[i][j] ].push_back( i );
        }
    }
}

void TriangleSetTopologyContainer::createEdgeSetArray()
{
    d_edge.beginEdit();
    if(!hasTriangles()) // this method should only be called when triangles exist
    {
#ifndef NDEBUG
        sout << "Warning. [TriangleSetTopologyContainer::createEdgeSetArray] triangle array is empty." << endl;
#endif
        createTriangleSetArray();
    }

    if(hasEdges())
    {
#ifndef NDEBUG
        sout << "Warning. [TriangleSetTopologyContainer::createEdgeSetArray] edge array is not empty." << endl;
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

            // sort vertices in lexicographic order
            const Edge e = ((v1<v2) ? Edge(v1,v2) : Edge(v2,v1));

            if(edgeMap.find(e) == edgeMap.end())
            {
                // edge not in edgeMap so create a new one
                const int edgeIndex = edgeMap.size();
                edgeMap[e] = edgeIndex;
                m_edge.push_back(e);
            }
        }
    }
    d_edge.endEdit();
}

void TriangleSetTopologyContainer::createTriangleEdgeArray()
{
    if(!hasTriangles()) // this method should only be called when triangles exist
    {
#ifndef NDEBUG
        sout << "Warning. [TriangleSetTopologyContainer::createTriangleEdgeArray] triangle array is empty." << endl;
#endif
        createTriangleSetArray();
    }

    // this should never be called : remove existing triangle edges
    if(hasTriangleEdges())
        clearTriangleEdges();

    if(!hasEdges()) // To optimize, this method should be called without creating edgesArray before.
    {
#ifndef NDEBUG
        sout << "Warning. [TriangleSetTopologyContainer::createTriangleEdgeArray] edge array is empty." << endl;
#endif

        /// create edge array and triangle edge array at the same time
        const unsigned int numTriangles = getNumberOfTriangles();
        m_triangleEdge.resize(numTriangles);

        d_edge.beginEdit();
        // create a temporary map to find redundant edges
        std::map<Edge, unsigned int> edgeMap;

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
                    const int edgeIndex = edgeMap.size();
                    /// add new edge
                    edgeMap[e] = edgeIndex;
                    m_edge.push_back(e);
                }
                m_triangleEdge[i][j] = edgeMap[e];
            }
        }
        d_edge.endEdit();
    }
    else
    {
        /// there are already existing edges : must use an inefficient method. Parse all triangles and find the edge that match each triangle edge

        const unsigned int numTriangles = getNumberOfTriangles();
        const unsigned int numEdges = getNumberOfEdges();

        m_triangleEdge.resize(numTriangles);
        /// create a multi map where the key is a vertex index and the content is the indices of edges adjacent to that vertex.
        std::multimap<PointID, EdgeID> edgeVertexShellMap;
        std::multimap<PointID, EdgeID>::iterator it;
        bool foundEdge;

        for (unsigned int edge=0; edge<numEdges; ++edge)  //Todo: check if not better using multimap <PointID ,TriangleID> and for each edge, push each triangle present in both shell
        {
            edgeVertexShellMap.insert(std::pair<PointID, EdgeID> (m_edge[edge][0],edge));
            edgeVertexShellMap.insert(std::pair<PointID, EdgeID> (m_edge[edge][1],edge));
        }

        for(unsigned int i=0; i<numTriangles; ++i)
        {
            Triangle &t = m_triangle[i];
            // adding edge i in the edge shell of both points
            for(unsigned int j=0; j<3; ++j)
            {
                //finding edge i in edge array
                std::pair<std::multimap<PointID, EdgeID>::iterator, std::multimap<PointID, EdgeID>::iterator > itPair=edgeVertexShellMap.equal_range(t[(j+1)%3]);

                foundEdge=false;
                for(it=itPair.first; (it!=itPair.second) && (foundEdge==false); ++it)
                {
                    unsigned int edge = (*it).second;
                    if ( (m_edge[edge][0] == t[(j+1)%3] && m_edge[edge][1] == t[(j+2)%3]) || (m_edge[edge][0] == t[(j+2)%3] && m_edge[edge][1] == t[(j+1)%3]))
                    {
                        m_triangleEdge[i][j] = edge;
                        foundEdge=true;
                    }
                }
#ifndef NDEBUG
                if (foundEdge==false)
                    sout << "[TriangleSetTopologyContainer::getTriangleArray] cannot find edge for triangle " << i << "and edge "<< j << endl;
#endif
            }
        }
    }
}



void TriangleSetTopologyContainer::createElementsOnBorder()
{

    if(!hasTriangleEdgeShell())	// Use the triangleEdgeShellArray. Should check if it is consistent
    {
#ifndef NDEBUG
        std::cout << "Warning. [ManifoldTriangleSetTopologyContainer::createElementsOnBorder] Triangle edge shell array is empty." << std::endl;
#endif

        createTriangleEdgeShellArray();
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


    for (unsigned int i = 0; i < nbrEdges; i++)
    {
        if (m_triangleEdgeShell[i].size() == 1) // I.e this edge is on a border
        {

            // --- Triangle case ---
            for (unsigned int j = 0; j < m_trianglesOnBorder.size(); j++) // Loop to avoid duplicated indices
            {
                if (m_trianglesOnBorder[j] == m_triangleEdgeShell[i][0])
                {
                    newTriangle = false;
                    break;
                }
            }

            if(newTriangle) // If index doesn't already exist, add it to the list of triangles On border.
            {
                m_trianglesOnBorder.push_back (m_triangleEdgeShell[i][0]);
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


const sofa::helper::vector<Triangle> & TriangleSetTopologyContainer::getTriangleArray()
{
    if(!hasTriangles() && getNbPoints()>0)
    {
#ifndef NDEBUG
        sout << "[TriangleSetTopologyContainer::getTriangleArray] creating triangle array." << endl;
#endif
        createTriangleSetArray();
    }

    return m_triangle;
}


int TriangleSetTopologyContainer::getTriangleIndex(PointID v1, PointID v2, PointID v3)
{
    if(!hasTriangleVertexShell())
        createTriangleVertexShellArray();

    sofa::helper::vector<unsigned int> set1 = getTriangleVertexShell(v1);
    sofa::helper::vector<unsigned int> set2 = getTriangleVertexShell(v2);
    sofa::helper::vector<unsigned int> set3 = getTriangleVertexShell(v3);

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

#ifndef NDEBUG
    if(out2.size() > 1)
        sout << "Warning. [TriangleSetTopologyContainer::getTriangleIndex] more than one triangle found" << endl;
#endif

    if (out2.size()==1)
        return (int) (out2[0]);
    else
        return -1;
}

unsigned int TriangleSetTopologyContainer::getNumberOfTriangles() const
{
    return m_triangle.size();
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &TriangleSetTopologyContainer::getTriangleVertexShellArray()
{
    if(!hasTriangleVertexShell())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        sout << "Warning. [TriangleSetTopologyContainer::getTriangleVertexShellArray] triangle vertex shell array is empty." << endl;
#endif
        createTriangleVertexShellArray();
    }

    return m_triangleVertexShell;
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &TriangleSetTopologyContainer::getTriangleEdgeShellArray()
{
    if(!hasTriangleEdgeShell())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        sout << "Warning. [TriangleSetTopologyContainer::getTriangleEdgeShellArray] triangle edge shell array is empty." << endl;
#endif
        createTriangleEdgeShellArray();
    }

    return m_triangleEdgeShell;
}

const sofa::helper::vector<TriangleEdges> &TriangleSetTopologyContainer::getTriangleEdgeArray()
{
    if(m_triangleEdge.empty())
        createTriangleEdgeArray();

    return m_triangleEdge;
}

const VertexTriangles& TriangleSetTopologyContainer::getTriangleVertexShell(PointID i)
{
    if(!hasTriangleVertexShell())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        sout << "Warning. [TriangleSetTopologyContainer::getTriangleVertexShell] triangle vertex shell array is empty." << endl;
#endif
        createTriangleVertexShellArray();
    }
    else if( i >= m_triangleVertexShell.size())
    {
#ifndef NDEBUG
        sout << "Error. [TriangleSetTopologyContainer::getTriangleVertexShell] index out of bounds." << endl;
#endif
        createTriangleVertexShellArray();
    }

    return m_triangleVertexShell[i];
}

const EdgeTriangles& TriangleSetTopologyContainer::getTriangleEdgeShell(EdgeID i)
{
    if(!hasTriangleEdgeShell())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        sout << "Warning. [TriangleSetTopologyContainer::getTriangleEdgeShell] triangle edge shell array is empty." << endl;
#endif
        createTriangleEdgeShellArray();
    }
    else if( i >= m_triangleEdgeShell.size())
    {
#ifndef NDEBUG
        sout << "Error. [TriangleSetTopologyContainer::getTriangleEdgeShell] index out of bounds." << endl;
#endif
        createTriangleEdgeShellArray();
    }

    return m_triangleEdgeShell[i];
}

const TriangleEdges &TriangleSetTopologyContainer::getTriangleEdge(const unsigned int i)
{
    if(m_triangleEdge.empty())
        createTriangleEdgeArray();

    if( i >= m_triangleEdge.size())
    {
#ifndef NDEBUG
        sout << "Error. [TriangleSetTopologyContainer::getTriangleEdge] index out of bounds." << endl;
#endif
        createTriangleEdgeArray();
    }

    return m_triangleEdge[i];
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

int TriangleSetTopologyContainer::getEdgeIndexInTriangle(const TriangleEdges &t, EdgeID edgeIndex) const
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


const sofa::helper::vector <TriangleID>& TriangleSetTopologyContainer::getTrianglesOnBorder()
{
    if (!hasBorderElementLists()) // this method should only be called when border lists exists
    {
#ifndef NDEBUG
        sout << "Warning. [ManifoldTriangleSetTopologyContainer::getTrianglesOnBorder] A border element list is empty." << endl;
#endif
        createElementsOnBorder();
    }

    return m_trianglesOnBorder;
}


const sofa::helper::vector <EdgeID>& TriangleSetTopologyContainer::getEdgesOnBorder()
{
    if (!hasBorderElementLists()) // this method should only be called when border lists exists
    {
#ifndef NDEBUG
        sout << "Warning. [ManifoldTriangleSetTopologyContainer::getEdgesOnBorder] A border element list is empty." << endl;
#endif
        createElementsOnBorder();
    }

    return m_edgesOnBorder;
}


const sofa::helper::vector <PointID>& TriangleSetTopologyContainer::getPointsOnBorder()
{
    if (!hasBorderElementLists()) // this method should only be called when border lists exists
    {
#ifndef NDEBUG
        sout << "Warning. [ManifoldTriangleSetTopologyContainer::getPointsOnBorder] A border element list is empty." << endl;
#endif
        createElementsOnBorder();
    }

    return m_pointsOnBorder;
}


sofa::helper::vector< unsigned int > &TriangleSetTopologyContainer::getTriangleEdgeShellForModification(const unsigned int i)
{
    if(!hasTriangleEdgeShell())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        sout << "Warning. [TriangleSetTopologyContainer::getTriangleEdgeShellForModification] triangle edge shell array is empty." << endl;
#endif
        createTriangleEdgeShellArray();
    }

    if( i >= m_triangleEdgeShell.size())
    {
#ifndef NDEBUG
        sout << "Error. [TriangleSetTopologyContainer::getTriangleEdgeShellForModification] index out of bounds." << endl;
#endif
        createTriangleEdgeShellArray();
    }

    return m_triangleEdgeShell[i];
}

sofa::helper::vector< unsigned int > &TriangleSetTopologyContainer::getTriangleVertexShellForModification(const unsigned int i)
{
    if(!hasTriangleVertexShell())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        sout << "Warning. [TriangleSetTopologyContainer::getTriangleVertexShellForModification] triangle vertex shell array is empty." << endl;
#endif
        createTriangleVertexShellArray();
    }

    if( i >= m_triangleVertexShell.size())
    {
#ifndef NDEBUG
        sout << "Error. [TriangleSetTopologyContainer::getTriangleVertexShellForModification] index out of bounds." << endl;
#endif
        createTriangleVertexShellArray();
    }

    return m_triangleVertexShell[i];
}

bool TriangleSetTopologyContainer::checkTopology() const
{

#ifndef NDEBUG
    bool ret = true;

    if (hasTriangleVertexShell())
    {
        std::set <int> triangleSet;
        std::set<int>::iterator it;

        for (unsigned int i=0; i<m_triangleVertexShell.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tvs = m_triangleVertexShell[i];
            for (unsigned int j=0; j<tvs.size(); ++j)
            {
                bool check_triangle_vertex_shell = (m_triangle[tvs[j]][0]==i)
                        || (m_triangle[tvs[j]][1]==i)
                        || (m_triangle[tvs[j]][2]==i);
                if(!check_triangle_vertex_shell)
                {
                    std::cout << "*** CHECK FAILED : check_triangle_vertex_shell, i = " << i << " , j = " << j << std::endl;
                    ret = false;
                }

                it=triangleSet.find(tvs[j]);
                if(it == triangleSet.end())
                {
                    triangleSet.insert (tvs[j]);
                }
            }
        }

        if(triangleSet.size()  != m_triangle.size())
        {
            std::cout << "*** CHECK FAILED : check_triangle_vertex_shell, triangle are missing in m_triangleVertexShell" <<std::endl;
            ret = false;
        }
    }

    if (hasTriangleEdgeShell())
    {
        std::set <int> triangleSet;
        std::set<int>::iterator it;

        for (unsigned int i=0; i<m_triangleEdgeShell.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tes=m_triangleEdgeShell[i];
            for (unsigned int j=0; j<tes.size(); ++j)
            {
                bool check_triangle_edge_shell =   (m_triangleEdge[tes[j]][0]==i)
                        || (m_triangleEdge[tes[j]][1]==i)
                        || (m_triangleEdge[tes[j]][2]==i);
                if(!check_triangle_edge_shell)
                {
                    std::cout << "*** CHECK FAILED : check_triangle_edge_shell, i = " << i << " , j = " << j << std::endl;
                    ret = false;
                }

                it=triangleSet.find(tes[j]);
                if(it == triangleSet.end())
                {
                    triangleSet.insert (tes[j]);
                }

            }
        }

        if(triangleSet.size()  != m_triangle.size())
        {
            std::cout << "*** CHECK FAILED : check_triangle_edge_shell, triangle are missing in m_triangleEdgeShell" <<std::endl;
            ret = false;
        }

    }

    return ret && EdgeSetTopologyContainer::checkTopology();

#else
    return true;
#endif
}

bool TriangleSetTopologyContainer::hasTriangles() const
{
    return !m_triangle.empty();
}

bool TriangleSetTopologyContainer::hasTriangleEdges() const
{
    return !m_triangleEdge.empty();
}

bool TriangleSetTopologyContainer::hasTriangleVertexShell() const
{
    return !m_triangleVertexShell.empty();
}

bool TriangleSetTopologyContainer::hasTriangleEdgeShell() const
{
    return !m_triangleEdgeShell.empty();
}

bool TriangleSetTopologyContainer::hasBorderElementLists() const
{
    if(!m_trianglesOnBorder.empty() && !m_edgesOnBorder.empty() && !m_pointsOnBorder.empty())
        return true;
    else
        return false;
}

void TriangleSetTopologyContainer::clearTriangleVertexShell()
{
    for(unsigned int i=0; i<m_triangleVertexShell.size(); ++i)
        m_triangleVertexShell[i].clear();

    m_triangleVertexShell.clear();
}

void TriangleSetTopologyContainer::clearTriangleEdgeShell()
{
    for(unsigned int i=0; i<m_triangleEdgeShell.size(); ++i)
        m_triangleEdgeShell[i].clear();

    m_triangleEdgeShell.clear();
}

void TriangleSetTopologyContainer::clearTriangleEdges()
{
    m_triangleEdge.clear();
}

void TriangleSetTopologyContainer::clearTriangles()
{
    d_triangle.beginEdit();
    m_triangle.clear();
    d_triangle.endEdit();
}

void TriangleSetTopologyContainer::clear()
{
    clearTriangleVertexShell();
    clearTriangleEdgeShell();
    clearTriangleEdges();
    clearTriangles();
    EdgeSetTopologyContainer::clear();
}

} // namespace topology

} // namespace component

} // namespace sofa
