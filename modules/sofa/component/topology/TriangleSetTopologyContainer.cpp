/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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

TriangleSetTopologyContainer::TriangleSetTopologyContainer(core::componentmodel::topology::BaseTopology *top )
    : EdgeSetTopologyContainer(top)
{}

TriangleSetTopologyContainer::TriangleSetTopologyContainer(core::componentmodel::topology::BaseTopology *top,
        const sofa::helper::vector< Triangle > &triangles )
    : EdgeSetTopologyContainer(top),
      m_triangle( triangles )
{}

void TriangleSetTopologyContainer::createTriangleSetArray()
{
#ifndef NDEBUG
    cout << "Error. [TriangleSetTopologyContainer::createTriangleSetArray] This method must be implemented by a child topology." << endl;
#endif
}

void TriangleSetTopologyContainer::createTriangleVertexShellArray ()
{
    if(!hasTriangles()) // TODO : this method should only be called when triangles exist
    {
#ifndef NDEBUG
        cout << "Warning. [TriangleSetTopologyContainer::createTriangleVertexShellArray] triangle array is empty." << endl;
#endif
        createTriangleSetArray();
    }

    if(hasTriangleVertexShell())
    {
        clearTriangleVertexShell();
    }

    m_triangleVertexShell.resize( m_basicTopology->getNbPoints() );
    unsigned int j;

    for (unsigned int i = 0; i < m_triangle.size(); ++i)
    {
        // adding edge i in the edge shell of both points
        for (j=0; j<3; ++j)
            m_triangleVertexShell[ m_triangle[i][j]  ].push_back( i );
    }
}

void TriangleSetTopologyContainer::createTriangleEdgeShellArray ()
{
    if(!hasTriangles()) // TODO : this method should only be called when triangles exist
    {
#ifndef NDEBUG
        cout << "Warning. [TriangleSetTopologyContainer::createTriangleEdgeShellArray] triangle array is empty." << endl;
#endif
        createTriangleSetArray();
    }

    if(!hasEdges()) // TODO : this method should only be called when edges exist
    {
#ifndef NDEBUG
        cout << "Warning. [TriangleSetTopologyContainer::createTriangleEdgeShellArray] edge array is empty." << endl;
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
    unsigned int j;
    const sofa::helper::vector< TriangleEdges > &tea=getTriangleEdgeArray();

    for (unsigned int i = 0; i < numTriangles; ++i)
    {
        // adding edge i in the edge shell of both points
        for (j=0; j<3; ++j)
        {
            m_triangleEdgeShell[ tea[i][j] ].push_back( i );
        }
    }
}

void TriangleSetTopologyContainer::createEdgeSetArray()
{
    if(!hasTriangles()) // TODO : this method should only be called when triangles exist
    {
#ifndef NDEBUG
        cout << "Warning. [TriangleSetTopologyContainer::createEdgeSetArray] triangle array is empty." << endl;
#endif
        createTriangleSetArray();
    }

    if(hasEdges()) // TODO : this method should only be called when triangles exist
    {
#ifndef NDEBUG
        cout << "Warning. [TriangleSetTopologyContainer::createEdgeSetArray] edge array is not empty." << endl;
#endif

        // clear edges and all shells that depend on edges
        clearEdges();

        if(hasEdgeVertexShell())
            clearEdgeVertexShell();

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
}

void TriangleSetTopologyContainer::createTriangleEdgeArray()
{
    if(!hasTriangles()) // TODO : this method should only be called when triangles exist
    {
#ifndef NDEBUG
        cout << "Warning. [TriangleSetTopologyContainer::createTriangleEdgeArray] triangle array is empty." << endl;
#endif
        createTriangleSetArray();
    }

    if(!hasEdges()) // TODO : this method should only be called when edges exist
    {
#ifndef NDEBUG
        cout << "Warning. [TriangleSetTopologyContainer::createTriangleEdgeArray] edge array is empty." << endl;
#endif
        createEdgeSetArray();
    }

    if(hasTriangleEdges())
        clearTriangleEdges();

    const unsigned int numTriangles = getNumberOfTriangles();

    m_triangleEdge.resize(numTriangles);
    for(unsigned int i=0; i<numTriangles; ++i)
    {
        Triangle &t = m_triangle[i];
        // adding edge i in the edge shell of both points
        for(unsigned int j=0; j<3; ++j)
        {
            int edgeIndex = getEdgeIndex(t[(j+1)%3],t[(j+2)%3]);
            assert(edgeIndex!= -1);
            m_triangleEdge[i][j] = edgeIndex;
        }
    }
}

const sofa::helper::vector<Triangle> &TriangleSetTopologyContainer::getTriangleArray()
{
    if(!hasTriangles()) // TODO : this method should only be called when triangles exist
    {
#ifndef NDEBUG
        cout << "Warning. [TriangleSetTopologyContainer::getTriangleArray] triangle array is empty." << endl;
#endif
        createTriangleSetArray();
    }

    return m_triangle;
}


int TriangleSetTopologyContainer::getTriangleIndex(const unsigned int v1,
        const unsigned int v2,
        const unsigned int v3)
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
        cout << "Warning. [TriangleSetTopologyContainer::getTriangleIndex] more than one triangle found" << endl;
#endif

    if (out2.size()==1)
        return (int) (out2[0]);
    else
        return -1;
}

const Triangle &TriangleSetTopologyContainer::getTriangle(const unsigned int i) // TODO : const
{
    if(!hasTriangles()) // TODO : this method should only be called when triangles exist
    {
#ifndef NDEBUG
        cout << "Warning. [TriangleSetTopologyContainer::] triangle array is empty." << endl;
#endif
        createTriangleSetArray();
    }

    return m_triangle[i];
}

unsigned int TriangleSetTopologyContainer::getNumberOfTriangles()
{
    if(!hasTriangles()) // TODO : this method should only be called when triangles exist
    {
#ifndef NDEBUG
        cout << "Warning. [TriangleSetTopologyContainer::] triangle array is empty." << endl;
#endif
        createTriangleSetArray();
    }

    return m_triangle.size();
}


const sofa::helper::vector< sofa::helper::vector<unsigned int> > &TriangleSetTopologyContainer::getTriangleVertexShellArray()
{
    if(!hasTriangleVertexShell())	// TODO : this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        cout << "Warning. [TriangleSetTopologyContainer::getTriangleVertexShellArray] triangle vertex shell array is empty." << endl;
#endif
        createTriangleVertexShellArray();
    }

    return m_triangleVertexShell;
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &TriangleSetTopologyContainer::getTriangleEdgeShellArray()
{
    if(!hasTriangleEdgeShell())	// TODO : this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        cout << "Warning. [TriangleSetTopologyContainer::getTriangleEdgeShellArray] triangle edge shell array is empty." << endl;
#endif
        createTriangleEdgeShellArray();
    }

    return m_triangleEdgeShell;
}

const sofa::helper::vector< TriangleEdges> &TriangleSetTopologyContainer::getTriangleEdgeArray()
{
    if(m_triangleEdge.empty())
        createTriangleEdgeArray();

    return m_triangleEdge;
}

const sofa::helper::vector< unsigned int > &TriangleSetTopologyContainer::getTriangleVertexShell(const unsigned int i)
{
    if(!hasTriangleVertexShell())	// TODO : this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        cout << "Warning. [TriangleSetTopologyContainer::getTriangleVertexShell] triangle vertex shell array is empty." << endl;
#endif
        createTriangleVertexShellArray();
    }
    else if( i >= m_triangleVertexShell.size())
    {
#ifndef NDEBUG
        cout << "Error. [TriangleSetTopologyContainer::getTriangleVertexShell] index out of bounds." << endl;
#endif
        createTriangleVertexShellArray();
    }

    return m_triangleVertexShell[i];
}

const sofa::helper::vector< unsigned int > &TriangleSetTopologyContainer::getTriangleEdgeShell(const unsigned int i)
{
    if(!hasTriangleEdgeShell())	// TODO : this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        cout << "Warning. [TriangleSetTopologyContainer::getTriangleEdgeShell] triangle edge shell array is empty." << endl;
#endif
        createTriangleEdgeShellArray();
    }
    else if( i >= m_triangleEdgeShell.size())
    {
#ifndef NDEBUG
        cout << "Error. [TriangleSetTopologyContainer::getTriangleEdgeShell] index out of bounds." << endl;
#endif
        createTriangleEdgeShellArray();
    }

    return m_triangleEdgeShell[i];
}

const TriangleEdges &TriangleSetTopologyContainer::getTriangleEdge(const unsigned int i)
{
    if(m_triangleEdge.empty())
        createTriangleEdgeArray();
    else if( i >= m_triangleEdge.size())
    {
#ifndef NDEBUG
        cout << "Error. [TriangleSetTopologyContainer::getTriangleEdge] index out of bounds." << endl;
#endif
        createTriangleEdgeArray();
    }

    return m_triangleEdge[i];
}

int TriangleSetTopologyContainer::getVertexIndexInTriangle(const Triangle &t,const unsigned int vertexIndex) const
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

int TriangleSetTopologyContainer::getEdgeIndexInTriangle(const TriangleEdges &t,const unsigned int edgeIndex) const
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

sofa::helper::vector< unsigned int > &TriangleSetTopologyContainer::getTriangleEdgeShellForModification(const unsigned int i)
{
    if(!hasTriangleEdgeShell())	// TODO : this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        cout << "Warning. [TriangleSetTopologyContainer::getTriangleEdgeShellForModification] triangle edge shell array is empty." << endl;
#endif
        createTriangleEdgeShellArray();
    }
    else if( i >= m_triangleEdgeShell.size())
    {
#ifndef NDEBUG
        cout << "Error. [TriangleSetTopologyContainer::getTriangleEdgeShellForModification] index out of bounds." << endl;
#endif
        createTriangleEdgeShellArray();
    }

    return m_triangleEdgeShell[i];
}

sofa::helper::vector< unsigned int > &TriangleSetTopologyContainer::getTriangleVertexShellForModification(const unsigned int i)
{
    if(!hasTriangleVertexShell())	// TODO : this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        cout << "Warning. [TriangleSetTopologyContainer::getTriangleVertexShellForModification] triangle vertex shell array is empty." << endl;
#endif
        createTriangleVertexShellArray();
    }
    else if( i >= m_triangleVertexShell.size())
    {
#ifndef NDEBUG
        cout << "Error. [TriangleSetTopologyContainer::getTriangleVertexShellForModification] index out of bounds." << endl;
#endif
        createTriangleVertexShellArray();
    }

    return m_triangleVertexShell[i];
}

bool TriangleSetTopologyContainer::checkTopology() const
{
#ifndef NDEBUG
    bool ret = true;

    if(!hasTriangles()) // TODO : this method should only be called when triangles exist
    {
        cout << "Warning. [TriangleSetTopologyContainer::checkTopology] triangle array is empty." << endl;

        if(hasEdges())
            ret = EdgeSetTopologyContainer::checkTopology();

        return ret;
    }

    if(hasEdges())
        ret = EdgeSetTopologyContainer::checkTopology();

    if (hasTriangleVertexShell())
    {
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
            }
        }
    }

    if (hasTriangleEdgeShell())
    {
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
            }
        }
    }

    return ret;
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
    m_triangle.clear();
}

void snapping_test_triangle(double epsilon, double alpha0, double alpha1, double alpha2,
        bool& is_snap_0, bool& is_snap_1, bool& is_snap_2)
{
    is_snap_0=false;
    is_snap_1=false;
    is_snap_2=false;

    if(alpha0>=alpha1 && alpha0>=alpha2)
    {
        is_snap_0=(alpha1+alpha2<epsilon);
    }
    else
    {
        if(alpha1>=alpha0 && alpha1>=alpha2)
        {
            is_snap_1=(alpha0+alpha2<epsilon);
        }
        else // alpha2>=alpha0 && alpha2>=alpha1
        {
            is_snap_2=(alpha0+alpha1<epsilon);
        }
    }
}

void snapping_test_edge(double epsilon,	double alpha0, double alpha1,
        bool& is_snap_0, bool& is_snap_1)
{
    is_snap_0=false;
    is_snap_1=false;

    if(alpha0>=alpha1)
    {
        is_snap_0=(alpha1<epsilon);
    }
    else // alpha1>=alpha0
    {
        is_snap_1=(alpha0<epsilon);
    }
}

} // namespace topology

} // namespace component

} // namespace sofa
