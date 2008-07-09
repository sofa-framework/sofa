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
#include <sofa/component/topology/TriangleSetTopology.h>
#include <sofa/component/topology/TriangleSetTopology.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(TriangleSetTopology)

int TriangleSetTopologyClass = core::RegisterObject("Triangle set topology")
#ifndef SOFA_FLOAT
        .add< TriangleSetTopology<Vec3dTypes> >()
        .add< TriangleSetTopology<Vec2dTypes> >()
        .add< TriangleSetTopology<Vec1dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< TriangleSetTopology<Vec3fTypes> >()
        .add< TriangleSetTopology<Vec2fTypes> >()
        .add< TriangleSetTopology<Vec1fTypes> >()
#endif
        ;


#ifndef SOFA_FLOAT
template class TriangleSetTopology<Vec3dTypes>;
template class TriangleSetTopology<Vec2dTypes>;
template class TriangleSetTopology<Vec1dTypes>;

template class TriangleSetTopologyAlgorithms<Vec3dTypes>;
template class TriangleSetTopologyAlgorithms<Vec2dTypes>;
template class TriangleSetTopologyAlgorithms<Vec1dTypes>;

template class TriangleSetGeometryAlgorithms<Vec3dTypes>;
template class TriangleSetGeometryAlgorithms<Vec2dTypes>;
template class TriangleSetGeometryAlgorithms<Vec1dTypes>;


template class TriangleSetTopologyModifier<Vec3dTypes>;
template class TriangleSetTopologyModifier<Vec2dTypes>;
template class TriangleSetTopologyModifier<Vec1dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class TriangleSetTopology<Vec3fTypes>;
template class TriangleSetTopology<Vec2fTypes>;
template class TriangleSetTopology<Vec1fTypes>;

template class TriangleSetTopologyAlgorithms<Vec3fTypes>;
template class TriangleSetTopologyAlgorithms<Vec2fTypes>;
template class TriangleSetTopologyAlgorithms<Vec1fTypes>;


template class TriangleSetGeometryAlgorithms<Vec3fTypes>;
template class TriangleSetGeometryAlgorithms<Vec2fTypes>;
template class TriangleSetGeometryAlgorithms<Vec1fTypes>;


template class TriangleSetTopologyModifier<Vec3fTypes>;
template class TriangleSetTopologyModifier<Vec2fTypes>;
template class TriangleSetTopologyModifier<Vec1fTypes>;
#endif
// implementation TriangleSetTopologyContainer

void TriangleSetTopologyContainer::createTriangleVertexShellArray ()
{
    m_triangleVertexShell.resize( m_basicTopology->getDOFNumber() );
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
    m_triangleEdgeShell.resize( getNumberOfEdges());
    unsigned int j;
    const sofa::helper::vector< TriangleEdges > &tea=getTriangleEdgeArray();


    for (unsigned int i = 0; i < m_triangle.size(); ++i)
    {
        // adding edge i in the edge shell of both points
        for (j=0; j<3; ++j)
        {
            m_triangleEdgeShell[ tea[i][j] ].push_back( i );
        }
    }
}


void TriangleSetTopologyContainer::createTriangleEdgeArray ()
{
    m_triangleEdge.resize( getNumberOfTriangles());
    unsigned int j;
    int edgeIndex;

    if (m_edge.size()>0)
    {

        for (unsigned int i = 0; i < m_triangle.size(); ++i)
        {
            Triangle &t=m_triangle[i];
            // adding edge i in the edge shell of both points
            for (j=0; j<3; ++j)
            {
                edgeIndex=getEdgeIndex(t[(j+1)%3],t[(j+2)%3]);
                assert(edgeIndex!= -1);
                m_triangleEdge[i][j]=edgeIndex;
            }
        }
    }
    else
    {
        // create a temporary map to find redundant edges
        std::map<Edge,unsigned int> edgeMap;
        std::map<Edge,unsigned int>::iterator ite;
        Edge e;
        unsigned int v1,v2;
        /// create the m_edge array at the same time than it fills the m_tetrahedronEdges array
        for (unsigned int i = 0; i < m_triangle.size(); ++i)
        {
            Triangle &t=m_triangle[i];
            for (j=0; j<3; ++j)
            {
                v1=t[(j+1)%3];
                v2=t[(j+2)%3];
                // sort vertices in lexicographics order
                if (v1<v2)
                {
                    e=Edge(v1,v2);
                }
                else
                {
                    e=Edge(v2,v1);
                }
                ite=edgeMap.find(e);
                if (ite==edgeMap.end())
                {
                    // edge not in edgeMap so create a new one
                    edgeIndex=edgeMap.size();
                    edgeMap[e]=edgeIndex;
                    m_edge.push_back(e);
                }
                else
                {
                    edgeIndex=(*ite).second;
                }
                m_triangleEdge[i][j]=edgeIndex;
            }
        }
    }
}


const sofa::helper::vector<Triangle> &TriangleSetTopologyContainer::getTriangleArray()
{
    if (!m_triangle.size())
        createTriangleSetArray();
    return m_triangle;
}


int TriangleSetTopologyContainer::getTriangleIndex(const unsigned int v1, const unsigned int v2, const unsigned int v3)
{
    //const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tvs=getTriangleVertexShellArray();

    const sofa::helper::vector<unsigned int> &set1=getTriangleVertexShell(v1);
    const sofa::helper::vector<unsigned int> &set2=getTriangleVertexShell(v2);
    const sofa::helper::vector<unsigned int> &set3=getTriangleVertexShell(v3);

    // The destination vector must be large enough to contain the result.
    sofa::helper::vector<unsigned int> out1(set1.size()+set2.size());
    sofa::helper::vector<unsigned int>::iterator result1;
    result1 = std::set_intersection(set1.begin(),set1.end(),set2.begin(),set2.end(),out1.begin());
    out1.erase(result1,out1.end());

    sofa::helper::vector<unsigned int> out2(set3.size()+out1.size());
    sofa::helper::vector<unsigned int>::iterator result2;
    result2 = std::set_intersection(set3.begin(),set3.end(),out1.begin(),out1.end(),out2.begin());
    out2.erase(result2,out2.end());

    assert(out2.size()==0 || out2.size()==1);

    if (out2.size()==1)
        return (int) (out2[0]);
    else
        return -1;
}

const Triangle &TriangleSetTopologyContainer::getTriangle(const unsigned int i)
{
    if (!m_triangle.size())
        createTriangleSetArray();
    return m_triangle[i];
}



unsigned int TriangleSetTopologyContainer::getNumberOfTriangles()
{
    if (!m_triangle.size())
        createTriangleSetArray();
    return m_triangle.size();
}



const sofa::helper::vector< sofa::helper::vector<unsigned int> > &TriangleSetTopologyContainer::getTriangleVertexShellArray()
{
    if (!m_triangleVertexShell.size())
        createTriangleVertexShellArray();
    return m_triangleVertexShell;
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &TriangleSetTopologyContainer::getTriangleEdgeShellArray()
{
    if (!m_triangleEdgeShell.size())
        createTriangleEdgeShellArray();
    return m_triangleEdgeShell;
}

const sofa::helper::vector< TriangleEdges> &TriangleSetTopologyContainer::getTriangleEdgeArray()
{
    if (!m_triangleEdge.size())
        createTriangleEdgeArray();
    return m_triangleEdge;
}




const sofa::helper::vector< unsigned int > &TriangleSetTopologyContainer::getTriangleVertexShell(const unsigned int i)
{
    if (!m_triangleVertexShell.size() || i > m_triangleVertexShell.size()-1)
        createTriangleVertexShellArray();
    return m_triangleVertexShell[i];
}


const sofa::helper::vector< unsigned int > &TriangleSetTopologyContainer::getTriangleEdgeShell(const unsigned int i)
{
    if (!m_triangleEdgeShell.size() || i > m_triangleEdgeShell.size()-1)
        createTriangleEdgeShellArray();
    return m_triangleEdgeShell[i];
}

const TriangleEdges &TriangleSetTopologyContainer::getTriangleEdge(const unsigned int i)
{
    if (!m_triangleEdge.size() || i > m_triangleEdge.size()-1)
        createTriangleEdgeArray();
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
    if (!m_triangleEdgeShell.size() || i > m_triangleEdgeShell.size()-1)
        createTriangleEdgeShellArray();
    return m_triangleEdgeShell[i];
}
sofa::helper::vector< unsigned int > &TriangleSetTopologyContainer::getTriangleVertexShellForModification(const unsigned int i)
{
    if (!m_triangleVertexShell.size() || i > m_triangleVertexShell.size()-1)
        createTriangleVertexShellArray();
    return m_triangleVertexShell[i];
}



TriangleSetTopologyContainer::TriangleSetTopologyContainer(core::componentmodel::topology::BaseTopology *top, /* const sofa::helper::vector< unsigned int > &DOFIndex, */
        const sofa::helper::vector< Triangle >         &triangles )
    : EdgeSetTopologyContainer( top /*,DOFIndex*/), m_triangle( triangles )
{

}
bool TriangleSetTopologyContainer::checkTopology() const
{
    //std::cout << "*** CHECK TriangleSetTopologyContainer ***" << std::endl;

    EdgeSetTopologyContainer::checkTopology();
    if (m_triangleVertexShell.size()>0)
    {
        unsigned int i,j;
        for (i=0; i<m_triangleVertexShell.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tvs=m_triangleVertexShell[i];
            for (j=0; j<tvs.size(); ++j)
            {
                bool check_triangle_vertex_shell = (m_triangle[tvs[j]][0]==i) ||  (m_triangle[tvs[j]][1]==i) || (m_triangle[tvs[j]][2]==i);
                if(!check_triangle_vertex_shell)
                {
                    std::cout << "*** CHECK FAILED : check_triangle_vertex_shell, i = " << i << " , j = " << j << std::endl;
                }
                assert(check_triangle_vertex_shell);
            }
        }
        //std::cout << "******** DONE : check_triangle_vertex_shell" << std::endl;
    }

    if (m_triangleEdgeShell.size()>0)
    {
        unsigned int i,j;
        for (i=0; i<m_triangleEdgeShell.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tes=m_triangleEdgeShell[i];
            for (j=0; j<tes.size(); ++j)
            {
                bool check_triangle_edge_shell = (m_triangleEdge[tes[j]][0]==i) ||  (m_triangleEdge[tes[j]][1]==i) || (m_triangleEdge[tes[j]][2]==i);
                if(!check_triangle_edge_shell)
                {
                    std::cout << "*** CHECK FAILED : check_triangle_edge_shell, i = " << i << " , j = " << j << std::endl;
                }
                assert(check_triangle_edge_shell);
            }
        }
        //std::cout << "******** DONE : check_triangle_edge_shell" << std::endl;
    }
    return true;
}

void snapping_test_triangle(double epsilon, double alpha0, double alpha1, double alpha2, bool& is_snap_0, bool& is_snap_1, bool& is_snap_2)
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
        else   // alpha2>=alpha0 && alpha2>=alpha1
        {

            is_snap_2=(alpha0+alpha1<epsilon);
        }

    }

}

void snapping_test_edge(double epsilon, double alpha0, double alpha1, bool& is_snap_0, bool& is_snap_1)
{

    is_snap_0=false;
    is_snap_1=false;

    if(alpha0>=alpha1)
    {

        is_snap_0=(alpha1<epsilon);

    }
    else   // alpha1>=alpha0
    {

        is_snap_1=(alpha0<epsilon);
    }

}

} // namespace topology

} // namespace component

} // namespace sofa

