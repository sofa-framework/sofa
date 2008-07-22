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
#include <sofa/component/topology/ManifoldEdgeSetTopology.h>
#include <sofa/component/topology/ManifoldEdgeSetTopology.inl>

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

// Use BOOST GRAPH LIBRARY :

#include <boost/config.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
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

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(ManifoldEdgeSetTopology)

// factory related stuff

int ManifoldEdgeSetTopologyClass = core::RegisterObject("Manofold Edge set topology")
#ifndef SOFA_FLOAT
        .add< ManifoldEdgeSetTopology<Vec3dTypes> >()
        .add< ManifoldEdgeSetTopology<Vec2dTypes> >()
        .add< ManifoldEdgeSetTopology<Vec1dTypes> >()
        .add< ManifoldEdgeSetTopology<Rigid3dTypes> >()
        .add< ManifoldEdgeSetTopology<Rigid2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< ManifoldEdgeSetTopology<Vec3fTypes> >()
        .add< ManifoldEdgeSetTopology<Vec2fTypes> >()
        .add< ManifoldEdgeSetTopology<Vec1fTypes> >()
        .add< ManifoldEdgeSetTopology<Rigid3fTypes> >()
        .add< ManifoldEdgeSetTopology<Rigid2fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class ManifoldEdgeSetTopology<Vec3dTypes>;
template class ManifoldEdgeSetTopology<Vec2dTypes>;
template class ManifoldEdgeSetTopology<Vec1dTypes>;
template class ManifoldEdgeSetTopology<Rigid3dTypes>;
template class ManifoldEdgeSetTopology<Rigid2dTypes>;


template class ManifoldEdgeSetTopologyAlgorithms<Vec3dTypes>;
template class ManifoldEdgeSetTopologyAlgorithms<Vec2dTypes>;
template class ManifoldEdgeSetTopologyAlgorithms<Vec1dTypes>;
template class ManifoldEdgeSetTopologyAlgorithms<Rigid3dTypes>;
template class ManifoldEdgeSetTopologyAlgorithms<Rigid2dTypes>;


template class ManifoldEdgeSetGeometryAlgorithms<Vec3dTypes>;
template class ManifoldEdgeSetGeometryAlgorithms<Vec2dTypes>;
template class ManifoldEdgeSetGeometryAlgorithms<Vec1dTypes>;

template class ManifoldEdgeSetTopologyModifier<Vec3dTypes>;
template class ManifoldEdgeSetTopologyModifier<Vec2dTypes>;
template class ManifoldEdgeSetTopologyModifier<Vec1dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class ManifoldEdgeSetTopology<Vec3fTypes>;
template class ManifoldEdgeSetTopology<Vec2fTypes>;
template class ManifoldEdgeSetTopology<Vec1fTypes>;
template class ManifoldEdgeSetTopology<Rigid3fTypes>;
template class ManifoldEdgeSetTopology<Rigid2fTypes>;


template class ManifoldEdgeSetTopologyAlgorithms<Vec3fTypes>;
template class ManifoldEdgeSetTopologyAlgorithms<Vec2fTypes>;
template class ManifoldEdgeSetTopologyAlgorithms<Vec1fTypes>;
template class ManifoldEdgeSetTopologyAlgorithms<Rigid3fTypes>;
template class ManifoldEdgeSetTopologyAlgorithms<Rigid2fTypes>;


template class ManifoldEdgeSetGeometryAlgorithms<Vec3fTypes>;
template class ManifoldEdgeSetGeometryAlgorithms<Vec2fTypes>;
template class ManifoldEdgeSetGeometryAlgorithms<Vec1fTypes>;

template class ManifoldEdgeSetGeometryAlgorithms<Rigid3fTypes>;
template class ManifoldEdgeSetGeometryAlgorithms<Rigid2fTypes>;

template class ManifoldEdgeSetTopologyModifier<Vec3fTypes>;
template class ManifoldEdgeSetTopologyModifier<Vec2fTypes>;
template class ManifoldEdgeSetTopologyModifier<Vec1fTypes>;
#endif

// ManifoldEdgeSetTopologyContainer implementation

ManifoldEdgeSetTopologyContainer::ManifoldEdgeSetTopologyContainer(core::componentmodel::topology::BaseTopology *top)
    : EdgeSetTopologyContainer( top )
{}

ManifoldEdgeSetTopologyContainer::ManifoldEdgeSetTopologyContainer(core::componentmodel::topology::BaseTopology *top,
        const sofa::helper::vector< Edge > &edges )
    : EdgeSetTopologyContainer( top, edges )
{}

void ManifoldEdgeSetTopologyContainer::createEdgeVertexShellArray()
{
    if(!hasEdges())	// TODO : this method should only be called when edges exist
    {
#ifndef NDEBUG
        cout << "Warning. [ManifoldEdgeSetTopologyContainer::createEdgeVertexShellArray] edge array is empty." << endl;
#endif
        createEdgeSetArray();
    }

    if(hasEdgeVertexShell())
    {
        clearEdgeVertexShell();
    }

    m_edgeVertexShell.resize( m_basicTopology->getDOFNumber() );

    for (unsigned int i = 0; i < m_edge.size(); ++i)
    {
        unsigned int size1 = m_edgeVertexShell[m_edge[i][1]].size();

        // adding edge i in the edge shell of both points, while respecting the manifold orientation
        // (ie : the edge will be added in second position for its first extremity point, and in first position for its second extremity point)

        m_edgeVertexShell[ m_edge[i][0]  ].push_back( i );

        if(size1==0)
        {
            m_edgeVertexShell[ m_edge[i][1]  ].push_back( i );
        }
        else
        {
            if(size1==1)
            {
                unsigned int j = m_edgeVertexShell[ m_edge[i][1]  ][0];
                m_edgeVertexShell[ m_edge[i][1]  ][0]=i;
                m_edgeVertexShell[ m_edge[i][1]  ].push_back( j );
            }
            else   // not manifold
            {
                m_edgeVertexShell[ m_edge[i][1]  ].push_back( i );
            }
        }
    }
}

void ManifoldEdgeSetTopologyContainer::createEdgeSetArray()
{
#ifndef NDEBUG
    cout << "Error. [ManifoldEdgeSetTopologyContainer::createEdgeSetArray] This method must be implemented by a child topology." << endl;
#endif
}

const sofa::helper::vector<Edge> &ManifoldEdgeSetTopologyContainer::getEdgeArray() // const
{
    if(!hasEdges())	// TODO : this method should only be called when edges exist
    {
#ifndef NDEBUG
        cout << "Warning. [ManifoldEdgeSetTopologyContainer::getEdgeArray] edge array is empty." << endl;
#endif
        createEdgeSetArray();
    }

    return m_edge;
}

int ManifoldEdgeSetTopologyContainer::getEdgeIndex(const unsigned int v1, const unsigned int v2)
{
    if(!hasEdges()) // TODO : this method should only be called when edges exist
    {
#ifndef NDEBUG
        cout << "Warning. [ManifoldEdgeSetTopologyContainer::getEdgeIndex] edge array is empty." << endl;
#endif
        createEdgeSetArray();
    }

    if(!hasEdgeVertexShell())
        createEdgeVertexShellArray();

    const sofa::helper::vector< unsigned int > &es1=getEdgeVertexShell(v1) ;
    const sofa::helper::vector<Edge> &ea=getEdgeArray();

    unsigned int i=0;
    int result= -1;
    while ((i<es1.size()) && (result== -1))
    {

        const Edge &e=ea[es1[i]];
        if ((e[0]==v2)|| (e[1]==v2))
            result=(int) es1[i];

        i++;
    }
    return result;
}
const Edge &ManifoldEdgeSetTopologyContainer::getEdge(const unsigned int i) // const
{
    if(!hasEdges()) // TODO : this method should only be called when edges exist
    {
#ifndef NDEBUG
        cout << "Warning. [ManifoldEdgeSetTopologyContainer::getEdge] edge array is empty." << endl;
#endif
        createEdgeSetArray();
    }

#ifndef NDEBUG
    if(m_edge.size() <= i)
    {
        cout << "Error. [ManifoldEdgeSetTopologyContainer::getEdge] edge array out of bounds: "
                << i << " >= " << m_edge.size() << endl;
    }
#endif

    return m_edge[i];
}

// Return the number of connected components
int ManifoldEdgeSetTopologyContainer::getNumberConnectedComponents(sofa::helper::vector<unsigned int>& components) // const
{
    computeConnectedComponent();

    components = m_ComponentVertexArray;
    return m_ConnectedComponentArray.size();
}

void ManifoldEdgeSetTopologyContainer::computeConnectedComponent()
{
    using namespace boost;
    typedef adjacency_list <vecS, vecS, undirectedS> Graph;

    if(isvoid_ConnectedComponent())
    {

        Graph G;

        const sofa::helper::vector<Edge> &ea=getEdgeArray();
        for (unsigned int k=0; k<ea.size(); ++k)
        {
            add_edge(ea[k][0], ea[k][1], G);
        }

        m_ComponentVertexArray.resize(num_vertices(G));
        int num = connected_components(G, &m_ComponentVertexArray[0]);

        std::vector< std::vector<int> > components(num);
        for(int i=0; i<num; i++)
        {
            components[i].resize(4);
            components[i][0]=0;
            components[i][1]=-1;
            components[i][2]=-1;
            components[i][3]=-1;
        }

        for(unsigned int j=0; j<m_ComponentVertexArray.size(); j++)
        {

            components[m_ComponentVertexArray[j]][0]+=1;
            components[m_ComponentVertexArray[j]][1]=j;

            if((getEdgeVertexShell(j)).size()==1)
            {

                if((getEdge((getEdgeVertexShell(j))[0]))[0]==j)
                {
                    components[m_ComponentVertexArray[j]][2]=j;
                }
                else   // (getEdge((getEdgeVertexShell(j))[0]))[1]==j
                {
                    components[m_ComponentVertexArray[j]][3]=j;
                }
            }
        }

        for(int i=0; i<num; i++)
        {

            bool is_closed = (components[i][2]==-1 && components[i][3]==-1);
            if(is_closed)
            {
                components[i][2]=components[i][1];
            }
            ConnectedComponent cc= ConnectedComponent(components[i][2], components[i][3], components[i][0], i);
            m_ConnectedComponentArray.push_back(cc);
        }

    }
    else
    {
        return;
    }
}

bool ManifoldEdgeSetTopologyContainer::checkTopology() const
{
#ifndef NDEBUG
    bool ret = EdgeSetTopologyContainer::checkTopology();

    if(!hasEdges()) // TODO : this method should only be called when edges exist
    {
        cout << "Warning. [ManifoldEdgeSetTopologyContainer::checkTopology] edge array is empty." << endl;

        return ret;
    }

    if (hasEdgeVertexShell())
    {
        unsigned int i;
        for (i=0; i<m_edgeVertexShell.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &es=m_edgeVertexShell[i];

            if(!(es.size()==1 || es.size()==2))
            {
                //std::cerr << "ERROR: ManifoldEdgeSetTopologyContainer::checkTopology() fails .\n"; // BIBI
                std::cout << "*** CHECK FAILED : check_manifold_edge_vertex_shell, i = " << i << std::endl;
                ret = false;
            }

        }
    }

    return ret;
#else
    return true;
#endif
}

unsigned int ManifoldEdgeSetTopologyContainer::getNumberOfEdges() // const
{
    if(!hasEdges()) // TODO : this method should only be called when edges exist
    {
#ifndef NDEBUG
        cout << "Warning. [ManifoldEdgeSetTopologyContainer::getNumberOfEdges] edge array is empty." << endl;
#endif
        createEdgeSetArray();
    }

    return m_edge.size();
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &ManifoldEdgeSetTopologyContainer::getEdgeVertexShellArray() // const
{
    if(!hasEdgeVertexShell())	// TODO : this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        cout << "Warning. [ManifoldEdgeSetTopologyContainer::getEdgeVertexShellArray] edge vertex shell array is empty." << endl;
#endif
        createEdgeVertexShellArray();
    }

    return m_edgeVertexShell;
}

const sofa::helper::vector< unsigned int > &ManifoldEdgeSetTopologyContainer::getEdgeVertexShell(const unsigned int i) // const
{
    if(!hasEdgeVertexShell())	// TODO : this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        cout << "Warning. [ManifoldEdgeSetTopologyContainer::getEdgeVertexShell] edge vertex shell array is empty." << endl;
#endif
        createEdgeVertexShellArray();
    }

#ifndef NDEBUG
    if(m_edgeVertexShell.size() <= i)
        cout << "Error. [ManifoldEdgeSetTopologyContainer::getEdgeVertexShell] edge vertex shell array out of bounds: "
                << i << " >= " << m_edgeVertexShell.size() << endl;
#endif

    return m_edgeVertexShell[i];
}

sofa::helper::vector< unsigned int > &ManifoldEdgeSetTopologyContainer::getEdgeVertexShellForModification(const unsigned int i)
{
    if(!hasEdgeVertexShell())	// TODO : this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        cout << "Warning. [ManifoldEdgeSetTopologyContainer::getEdgeVertexShellForModification] edge vertex shell array is empty." << endl;
#endif
        createEdgeVertexShellArray();
    }

    return m_edgeVertexShell[i];
}

bool ManifoldEdgeSetTopologyContainer::hasEdges() const
{
    return !m_edge.empty();
}

bool ManifoldEdgeSetTopologyContainer::hasEdgeVertexShell() const
{
    return !m_edgeVertexShell.empty();
}

void ManifoldEdgeSetTopologyContainer::clearEdges()
{
    m_edge.clear();
}

void ManifoldEdgeSetTopologyContainer::clearEdgeVertexShell()
{
    for(unsigned int i=0; i<m_edgeVertexShell.size(); ++i)
        m_edgeVertexShell[i].clear();

    m_edgeVertexShell.clear();
}

} // namespace topology

} // namespace component

} // namespace sofa

