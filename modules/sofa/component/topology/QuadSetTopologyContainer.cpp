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

#include <sofa/component/topology/QuadSetTopologyContainer.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/component/MeshLoader.h>

namespace sofa
{
namespace component
{
namespace topology
{
using namespace std;
using namespace sofa::defaulttype;
SOFA_DECL_CLASS(QuadSetTopologyContainer)
int QuadSetTopologyContainerClass = core::RegisterObject("Quad set topology container")
        .add< QuadSetTopologyContainer >()
        ;

QuadSetTopologyContainer::QuadSetTopologyContainer()
    : EdgeSetTopologyContainer()
{}

QuadSetTopologyContainer::QuadSetTopologyContainer(const sofa::helper::vector< Quad >& quads )
    : EdgeSetTopologyContainer(),
      m_quad( quads )
{}

void QuadSetTopologyContainer::init()
{
    sofa::component::MeshLoader* loader;
    this->getContext()->get(loader);

    if(loader)
    {
        m_quad = loader->getQuads();
    }

    // load points
    PointSetTopologyContainer::init();
}

void QuadSetTopologyContainer::createQuadSetArray()
{
#ifndef NDEBUG
    cout << "Error. [QuadSetTopologyContainer::createQuadSetArray] This method must be implemented by a child topology." << endl;
#endif
}

void QuadSetTopologyContainer::createQuadVertexShellArray()
{
    if(!hasQuads()) // TODO : this method should only be called when quads exist
    {
#ifndef NDEBUG
        cout << "Warning. [QuadSetTopologyContainer::createQuadVertexShellArray] quad array is empty." << endl;
#endif
        createQuadSetArray();
    }

    if(hasQuadVertexShell())
    {
        clearQuadVertexShell();
    }

    m_quadVertexShell.resize( getNbPoints() );

    for (unsigned int i=0; i<m_quad.size(); ++i)
    {
        // adding quad i in the quad shell of all points
        for (unsigned int j=0; j<4; ++j)
        {
            m_quadVertexShell[ m_quad[i][j] ].push_back( i );
        }
    }
}

void QuadSetTopologyContainer::createQuadEdgeShellArray()
{
    if(!hasQuads()) // TODO : this method should only be called when quads exist
    {
#ifndef NDEBUG
        cout << "Warning. [QuadSetTopologyContainer::createQuadEdgeShellArray] quad array is empty." << endl;
#endif
        createQuadSetArray();
    }

    if(!hasEdges()) // TODO : this method should only be called when edges exist
    {
#ifndef NDEBUG
        cout << "Warning. [QuadSetTopologyContainer::createQuadEdgeShellArray] edge array is empty." << endl;
#endif
        createEdgeSetArray();
    }

    if(!hasQuadEdges())
        createQuadEdgeArray();

    const unsigned int numQuads = getNumberOfQuads();
    const unsigned int numEdges = getNumberOfEdges();

    if(hasQuadEdgeShell())
    {
        clearQuadEdgeShell();
    }

    m_quadEdgeShell.resize(numEdges);

    for (unsigned int i=0; i<numQuads; ++i)
    {
        // adding quad i in the quad shell of all edges
        for (unsigned int j=0; j<4; ++j)
        {
            m_quadEdgeShell[ m_quadEdge[i][j] ].push_back( i );
        }
    }
}

void QuadSetTopologyContainer::createEdgeSetArray()
{
    if(!hasQuads()) // TODO : this method should only be called when quads exist
    {
#ifndef NDEBUG
        cout << "Warning. [QuadSetTopologyContainer::createEdgeSetArray] quad array is empty." << endl;
#endif
        createQuadSetArray();
    }

    if(hasEdges()) // TODO : this method should only be called when quads exist
    {
#ifndef NDEBUG
        cout << "Warning. [QuadSetTopologyContainer::createEdgeSetArray] edge array is not empty." << endl;
#endif

        // clear edges and all shells that depend on edges
        clearEdges();

        if(hasEdgeVertexShell())
            clearEdgeVertexShell();

        if(hasQuadEdges())
            clearQuadEdges();

        if(hasQuadEdgeShell())
            clearQuadEdgeShell();
    }

    // create a temporary map to find redundant edges
    std::map<Edge, unsigned int> edgeMap;

    for (unsigned int i=0; i<m_quad.size(); ++i)
    {
        const Quad &t = m_quad[i];
        for(unsigned int j=0; j<4; ++j)
        {
            const unsigned int v1 = t[(j+1)%4];
            const unsigned int v2 = t[(j+2)%4];

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

void QuadSetTopologyContainer::createQuadEdgeArray()
{
    if(!hasQuads()) // TODO : this method should only be called when quads exist
    {
#ifndef NDEBUG
        cout << "Warning. [QuadSetTopologyContainer::createQuadEdgeArray] quad array is empty." << endl;
#endif
        createQuadSetArray();
    }

    if(!hasEdges()) // TODO : this method should only be called when edges exist
    {
#ifndef NDEBUG
        cout << "Warning. [QuadSetTopologyContainer::createQuadEdgeArray] edge array is empty." << endl;
#endif
        createEdgeSetArray();
    }

    if(hasQuadEdges())
        clearQuadEdges();

    const unsigned int numQuads = getNumberOfQuads();

    m_quadEdge.resize( numQuads );

    for(unsigned int i=0; i<numQuads; ++i)
    {
        Quad &t = m_quad[i];
        // adding edge i in the edge shell of both points
        for (unsigned int j=0; j<4; ++j)
        {
            const int edgeIndex = getEdgeIndex(t[(j+1)%4],t[(j+2)%4]);

#ifndef NDEBUG
            if(edgeIndex == -1)
                cout << "Error. [QuadSetTopologyContainer::createQuadEdgeArray] edge not found." << endl;
#endif
            m_quadEdge[i][j]=edgeIndex;
        }
    }
}

const sofa::helper::vector<Quad> &QuadSetTopologyContainer::getQuadArray()
{
    if(!hasQuads()) // TODO : this method should only be called when quads exist
    {
#ifndef NDEBUG
        cout << "Warning. [QuadSetTopologyContainer::getQuadArray] quad array is empty." << endl;
#endif
        createQuadSetArray();
    }

    return m_quad;
}

int QuadSetTopologyContainer::getQuadIndex(const unsigned int v1,
        const unsigned int v2,
        const unsigned int v3,
        const unsigned int v4)
{
    if(!hasQuadVertexShell())
        createQuadVertexShellArray();

    sofa::helper::vector<unsigned int> set1 = getQuadVertexShell(v1);
    sofa::helper::vector<unsigned int> set2 = getQuadVertexShell(v2);
    sofa::helper::vector<unsigned int> set3 = getQuadVertexShell(v3);
    sofa::helper::vector<unsigned int> set4 = getQuadVertexShell(v4);

    sort(set1.begin(), set1.end());
    sort(set2.begin(), set2.end());
    sort(set3.begin(), set3.end());
    sort(set4.begin(), set4.end());

    // The destination vector must be large enough to contain the result.
    sofa::helper::vector<unsigned int> out1(set1.size()+set2.size());
    sofa::helper::vector<unsigned int>::iterator result1;
    result1 = std::set_intersection(set1.begin(),set1.end(),set2.begin(),set2.end(),out1.begin());
    out1.erase(result1,out1.end());

    sofa::helper::vector<unsigned int> out2(set3.size()+out1.size());
    sofa::helper::vector<unsigned int>::iterator result2;
    result2 = std::set_intersection(set3.begin(),set3.end(),out1.begin(),out1.end(),out2.begin());
    out2.erase(result2,out2.end());

    sofa::helper::vector<unsigned int> out3(set4.size()+out2.size());
    sofa::helper::vector<unsigned int>::iterator result3;
    result3 = std::set_intersection(set4.begin(),set4.end(),out2.begin(),out2.end(),out3.begin());
    out3.erase(result3,out3.end());

#ifndef NDEBUG
    if(out3.size() > 1)
        cout << "Warning. [QuadSetTopologyContainer::getQuadIndex] more than one quad found" << endl;
#endif

    if(out3.size()==1)
        return (int) (out3[0]);
    else
        return -1;
}

const Quad &QuadSetTopologyContainer::getQuad(const unsigned int i) // TODO : const
{
    if(!hasQuads()) // TODO : this method should only be called when quads exist
    {
#ifndef NDEBUG
        cout << "Warning. [QuadSetTopologyContainer::] quad array is empty." << endl;
#endif
        createQuadSetArray();
    }

    return m_quad[i];
}

unsigned int QuadSetTopologyContainer::getNumberOfQuads() // TODO : const
{
    if(!hasQuads()) // TODO : this method should only be called when quads exist
    {
#ifndef NDEBUG
        cout << "Warning. [QuadSetTopologyContainer::] quad array is empty." << endl;
#endif
        createQuadSetArray();
    }

    return m_quad.size();
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &QuadSetTopologyContainer::getQuadVertexShellArray()
{
    if(!hasQuadVertexShell())	// TODO : this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        cout << "Warning. [QuadSetTopologyContainer::getQuadVertexShellArray] quad vertex shell array is empty." << endl;
#endif
        createQuadVertexShellArray();
    }

    return m_quadVertexShell;
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &QuadSetTopologyContainer::getQuadEdgeShellArray()
{
    if(!hasQuadEdgeShell())	// TODO : this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        cout << "Warning. [QuadSetTopologyContainer::getQuadEdgeShellArray] quad edge shell array is empty." << endl;
#endif
        createQuadEdgeShellArray();
    }

    return m_quadEdgeShell;
}

const sofa::helper::vector< QuadEdges> &QuadSetTopologyContainer::getQuadEdgeArray()
{
    if(m_quadEdge.empty())
        createQuadEdgeArray();

    return m_quadEdge;
}

const sofa::helper::vector< unsigned int > &QuadSetTopologyContainer::getQuadVertexShell(const unsigned int i)
{
    if(!hasQuadVertexShell())	// TODO : this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        cout << "Warning. [QuadSetTopologyContainer::getQuadVertexShell] quad vertex shell array is empty." << endl;
#endif
        createQuadVertexShellArray();
    }
    else if( i >= m_quadVertexShell.size())
    {
#ifndef NDEBUG
        cout << "Error. [QuadSetTopologyContainer::getQuadVertexShell] index out of bounds." << endl;
#endif
        createQuadVertexShellArray();
    }

    return m_quadVertexShell[i];
}


const sofa::helper::vector< unsigned int > &QuadSetTopologyContainer::getQuadEdgeShell(const unsigned int i)
{
    if(!hasQuadEdgeShell())	// TODO : this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        cout << "Warning. [QuadSetTopologyContainer::getQuadEdgeShell] quad edge shell array is empty." << endl;
#endif
        createQuadEdgeShellArray();
    }
    else if( i >= m_quadEdgeShell.size())
    {
#ifndef NDEBUG
        cout << "Error. [QuadSetTopologyContainer::getQuadEdgeShell] index out of bounds." << endl;
#endif
        createQuadEdgeShellArray();
    }

    return m_quadEdgeShell[i];
}

const QuadEdges &QuadSetTopologyContainer::getQuadEdge(const unsigned int i)
{
    if(m_quadEdge.empty())
        createQuadEdgeArray();
    else if( i >= m_quadEdge.size())
    {
#ifndef NDEBUG
        cout << "Error. [QuadSetTopologyContainer::getQuadEdge] index out of bounds." << endl;
#endif
        createQuadEdgeArray();
    }

    return m_quadEdge[i];
}

int QuadSetTopologyContainer::getVertexIndexInQuad(Quad &t, unsigned int vertexIndex) const
{
    if(t[0]==vertexIndex)
        return 0;
    else if(t[1]==vertexIndex)
        return 1;
    else if(t[2]==vertexIndex)
        return 2;
    else if(t[3]==vertexIndex)
        return 3;
    else
        return -1;
}

int QuadSetTopologyContainer::getEdgeIndexInQuad(QuadEdges &t, unsigned int edgeIndex) const
{
    if(t[0]==edgeIndex)
        return 0;
    else if(t[1]==edgeIndex)
        return 1;
    else if(t[2]==edgeIndex)
        return 2;
    else if(t[3]==edgeIndex)
        return 3;
    else
        return -1;
}

sofa::helper::vector< unsigned int > &QuadSetTopologyContainer::getQuadEdgeShellForModification(const unsigned int i)
{
    if(!hasQuadEdgeShell())	// TODO : this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        cout << "Warning. [QuadSetTopologyContainer::getQuadEdgeShellForModification] quad edge shell array is empty." << endl;
#endif
        createQuadEdgeShellArray();
    }
    else if( i >= m_quadEdgeShell.size())
    {
#ifndef NDEBUG
        cout << "Error. [QuadSetTopologyContainer::getQuadEdgeShellForModification] index out of bounds." << endl;
#endif
        createQuadEdgeShellArray();
    }

    return m_quadEdgeShell[i];
}

sofa::helper::vector< unsigned int > &QuadSetTopologyContainer::getQuadVertexShellForModification(const unsigned int i)
{
    if(!hasQuadVertexShell())	// TODO : this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        cout << "Warning. [QuadSetTopologyContainer::getQuadVertexShellForModification] quad vertex shell array is empty." << endl;
#endif
        createQuadVertexShellArray();
    }
    else if( i >= m_quadVertexShell.size())
    {
#ifndef NDEBUG
        cout << "Error. [QuadSetTopologyContainer::getQuadVertexShellForModification] index out of bounds." << endl;
#endif
        createQuadVertexShellArray();
    }

    return m_quadVertexShell[i];
}


bool QuadSetTopologyContainer::checkTopology() const
{
#ifndef NDEBUG
    bool ret = true;

    if(!hasQuads()) // TODO : this method should only be called when quads exist
    {
        cout << "Warning. [QuadSetTopologyContainer::checkTopology] quad array is empty." << endl;
        if(hasEdges())
            ret = EdgeSetTopologyContainer::checkTopology();

        return ret;
    }

    if(hasEdges())
        ret = EdgeSetTopologyContainer::checkTopology();

    if(hasQuadVertexShell())
    {
        for (unsigned int i=0; i<m_quadVertexShell.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tvs = m_quadVertexShell[i];
            for (unsigned int j=0; j<tvs.size(); ++j)
            {
                if((m_quad[tvs[j]][0]!=i)
                   && (m_quad[tvs[j]][1]!=i)
                   && (m_quad[tvs[j]][2]!=i)
                   && (m_quad[tvs[j]][3]!=i))
                {
                    ret = false;
                    std::cout << "*** CHECK FAILED : check_quad_vertex_shell, i = " << i << " , j = " << j << std::endl;
                }
            }
        }
    }

    if(hasQuadEdgeShell())
    {
        for (unsigned int i=0; i<m_quadEdgeShell.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tes = m_quadEdgeShell[i];
            for (unsigned int j=0; j<tes.size(); ++j)
            {
                if((m_quadEdge[tes[j]][0]!=i)
                   && (m_quadEdge[tes[j]][1]!=i)
                   && (m_quadEdge[tes[j]][2]!=i)
                   && (m_quadEdge[tes[j]][3]!=i))
                {
                    ret = false;
                    std::cout << "*** CHECK FAILED : check_quad_edge_shell, i = " << i << " , j = " << j << std::endl;
                }
            }
        }
    }

    return ret;
#else
    return true;
#endif
}

bool QuadSetTopologyContainer::hasQuads() const
{
    return !m_quad.empty();
}

bool QuadSetTopologyContainer::hasQuadEdges() const
{
    return !m_quadEdge.empty();
}

bool QuadSetTopologyContainer::hasQuadVertexShell() const
{
    return !m_quadVertexShell.empty();
}

bool QuadSetTopologyContainer::hasQuadEdgeShell() const
{
    return !m_quadEdgeShell.empty();
}

void QuadSetTopologyContainer::clearQuadVertexShell()
{
    for(unsigned int i=0; i<m_quadVertexShell.size(); ++i)
        m_quadVertexShell[i].clear();

    m_quadVertexShell.clear();
}

void QuadSetTopologyContainer::clearQuadEdgeShell()
{
    for(unsigned int i=0; i<m_quadEdgeShell.size(); ++i)
        m_quadEdgeShell[i].clear();

    m_quadEdgeShell.clear();
}

void QuadSetTopologyContainer::clearQuadEdges()
{
    m_quadEdge.clear();
}

void QuadSetTopologyContainer::clearQuads()
{
    m_quad.clear();
}

} // namespace topology

} // namespace component

} // namespace sofa

