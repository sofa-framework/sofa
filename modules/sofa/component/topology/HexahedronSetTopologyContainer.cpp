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

#include <sofa/component/topology/HexahedronSetTopologyContainer.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/component/container/MeshLoader.h>

namespace sofa
{
namespace component
{
namespace topology
{

using namespace std;
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(HexahedronSetTopologyContainer)
int HexahedronSetTopologyContainerClass = core::RegisterObject("Hexahedron set topology container")
        .add< HexahedronSetTopologyContainer >()
        ;

const unsigned int hexahedronEdgeArray[12][2]= {{0,1},{0,3},{0,4},{1,2},{1,5},{2,3},{2,6},{3,7},{4,5},{4,7},{5,6},{6,7}};

HexahedronSetTopologyContainer::HexahedronSetTopologyContainer()
    : QuadSetTopologyContainer()
    , d_hexahedron(initDataPtr(&d_hexahedron, &m_hexahedron, "hexas", "List of hexahedron indices"))
{
}

HexahedronSetTopologyContainer::HexahedronSetTopologyContainer(const sofa::helper::vector< Hexahedron > &hexahedra )
    : QuadSetTopologyContainer()
    , m_hexahedron( hexahedra )
    , d_hexahedron(initDataPtr(&d_hexahedron, &m_hexahedron, "hexas", "List of hexahedron indices"))
{
    for (unsigned int i=0; i<m_hexahedron.size(); ++i)
    {
        for(unsigned int j=0; j<8; ++j)
        {
            int a = m_hexahedron[i][j];
            if (a >= getNbPoints()) nbPoints.setValue(a+1);
        }
    }
}

void HexahedronSetTopologyContainer::addHexa( int a, int b, int c, int d, int e, int f, int g, int h )
{
    d_hexahedron.beginEdit();
    m_hexahedron.push_back(Hexahedron(a,b,c,d,e,f,g,h));
    d_hexahedron.endEdit();
    if (a >= getNbPoints()) nbPoints.setValue(a+1);
    if (b >= getNbPoints()) nbPoints.setValue(b+1);
    if (c >= getNbPoints()) nbPoints.setValue(c+1);
    if (d >= getNbPoints()) nbPoints.setValue(d+1);
    if (e >= getNbPoints()) nbPoints.setValue(e+1);
    if (f >= getNbPoints()) nbPoints.setValue(f+1);
    if (g >= getNbPoints()) nbPoints.setValue(g+1);
    if (h >= getNbPoints()) nbPoints.setValue(h+1);
}

void HexahedronSetTopologyContainer::init()
{
    QuadSetTopologyContainer::init();
}

void HexahedronSetTopologyContainer::loadFromMeshLoader(sofa::component::container::MeshLoader* loader)
{
    // load points
    if (!m_hexahedron.empty()) return;
    PointSetTopologyContainer::loadFromMeshLoader(loader);
    d_hexahedron.beginEdit();
    loader->getHexas(m_hexahedron);
    d_hexahedron.endEdit();
}

void HexahedronSetTopologyContainer::createHexahedronSetArray()
{
#ifndef NDEBUG
    serr << "Error. [createHexahedronSetArray] This method must be implemented by a child topology." << sendl;
#endif
}

void HexahedronSetTopologyContainer::createEdgeSetArray()
{
    d_edge.beginEdit();
    if(hasEdges())
    {
        EdgeSetTopologyContainer::clear();

        clearQuadEdges();
        clearQuadEdgeShell();

        clearHexahedronEdges();
        clearHexahedronEdgeShell();
    }

    // create a temporary map to find redundant edges
    std::map<Edge,unsigned int> edgeMap;

    /// create the m_edge array at the same time than it fills the m_hexahedronEdge array
    for(unsigned int i=0; i<m_hexahedron.size(); ++i)
    {
        Hexahedron &t = m_hexahedron[i];
        for(unsigned int j=0; j<12; ++j)
        {
            unsigned int v1 = t[hexahedronEdgeArray[j][0]];
            unsigned int v2 = t[hexahedronEdgeArray[j][1]];
            const Edge e((v1<v2) ? Edge(v1,v2) : Edge(v2,v1));

            if(edgeMap.find(e)==edgeMap.end())
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

void HexahedronSetTopologyContainer::createHexahedronEdgeArray()
{
    if(!hasEdges())
        createEdgeSetArray();

    if(hasHexahedronEdges())
        clearHexahedronEdges();

    m_hexahedronEdge.resize( getNumberOfHexahedra());

    for(unsigned int i=0; i<m_hexahedron.size(); ++i)
    {
        Hexahedron &t = m_hexahedron[i];

        // adding edge i in the edge shell of both points
        for(unsigned int j=0; j<12; ++j)
        {
            const int edgeIndex = getEdgeIndex(t[hexahedronEdgeArray[j][0]],
                    t[hexahedronEdgeArray[j][1]]);
            m_hexahedronEdge[i][j] = edgeIndex;
        }
    }
}

void HexahedronSetTopologyContainer::createQuadSetArray()
{
    d_quad.beginEdit();
    if(hasQuads())
    {
        QuadSetTopologyContainer::clear();
        clearQuads();
        clearHexahedronQuads();
        clearHexahedronQuadShell();
    }

    // create a temporary map to find redundant quads
    std::map<Quad,unsigned int> quadMap;

    for(unsigned int i=0; i<m_hexahedron.size(); ++i)
    {
        Hexahedron &h = m_hexahedron[i];

        unsigned int v[4], val;

        // Quad 0 :
        v[0]=h[0];
        v[1]=h[3];
        v[2]=h[2];
        v[3]=h[1];

        // sort v such that v[0] is the smallest one
        while ((v[0]>v[1]) || (v[0]>v[2]) || (v[0]>v[3]))
        {
            val = v[0];
            v[0]=v[1];
            v[1]=v[2];
            v[2]=v[3];
            v[3]=val;
        }

        // sort vertices in lexicographics order
        int quadIndex;
        Quad qu=helper::make_array<unsigned int>(v[0],v[3],v[2],v[1]);
        std::map<Quad,unsigned int>::iterator itt = quadMap.find(qu);
        if(itt==quadMap.end())
        {
            // quad not in edgeMap so create a new one
            quadIndex=m_quad.size();
            quadMap[qu]=quadIndex;
            qu=helper::make_array<unsigned int>(v[0],v[1],v[2],v[3]);
            quadMap[qu]=quadIndex;
            m_quad.push_back(qu);
        }

        // Quad 1 :
        v[0]=h[4];
        v[1]=h[5];
        v[2]=h[6];
        v[3]=h[7];
        // sort v such that v[0] is the smallest one
        while ((v[0]>v[1]) || (v[0]>v[2]) || (v[0]>v[3]))
        {
            val=v[0];
            v[0]=v[1];
            v[1]=v[2];
            v[2]=v[3];
            v[3]=val;
        }
        // sort vertices in lexicographics order
        qu=helper::make_array<unsigned int>(v[0],v[3],v[2],v[1]);
        itt=quadMap.find(qu);
        if(itt==quadMap.end())
        {
            // quad not in edgeMap so create a new one
            quadIndex=m_quad.size();
            quadMap[qu]=quadIndex;
            qu=helper::make_array<unsigned int>(v[0],v[1],v[2],v[3]);
            quadMap[qu]=quadIndex;
            m_quad.push_back(qu);
        }

        // Quad 2 :
        v[0]=h[0]; v[1]=h[1]; v[2]=h[5]; v[3]=h[4];
        // sort v such that v[0] is the smallest one
        while ((v[0]>v[1]) || (v[0]>v[2]) || (v[0]>v[3]))
        {
            val=v[0];
            v[0]=v[1];
            v[1]=v[2];
            v[2]=v[3];
            v[3]=val;
        }
        // sort vertices in lexicographics order
        qu=helper::make_array<unsigned int>(v[0],v[3],v[2],v[1]);
        itt=quadMap.find(qu);
        if(itt==quadMap.end())
        {
            // quad not in edgeMap so create a new one
            quadIndex=m_quad.size();
            quadMap[qu]=quadIndex;
            qu=helper::make_array<unsigned int>(v[0],v[1],v[2],v[3]);
            quadMap[qu]=quadIndex;
            m_quad.push_back(qu);
        }

        // Quad 3 :
        v[0]=h[1]; v[1]=h[2]; v[2]=h[6]; v[3]=h[5];
        // sort v such that v[0] is the smallest one
        while ((v[0]>v[1]) || (v[0]>v[2]) || (v[0]>v[3]))
        {
            val=v[0];
            v[0]=v[1];
            v[1]=v[2];
            v[2]=v[3];
            v[3]=val;
        }
        // sort vertices in lexicographics order
        qu=helper::make_array<unsigned int>(v[0],v[3],v[2],v[1]);
        itt=quadMap.find(qu);
        if(itt==quadMap.end())
        {
            // quad not in edgeMap so create a new one
            quadIndex=m_quad.size();
            quadMap[qu]=quadIndex;
            qu=helper::make_array<unsigned int>(v[0],v[1],v[2],v[3]);
            quadMap[qu]=quadIndex;
            m_quad.push_back(qu);
        }

        // Quad 4 :
        v[0]=h[2];
        v[1]=h[3];
        v[2]=h[7];
        v[3]=h[6];
        // sort v such that v[0] is the smallest one
        while ((v[0]>v[1]) || (v[0]>v[2]) || (v[0]>v[3]))
        {
            val=v[0];
            v[0]=v[1];
            v[1]=v[2];
            v[2]=v[3];
            v[3]=val;
        }
        // sort vertices in lexicographics order
        qu=helper::make_array<unsigned int>(v[0],v[3],v[2],v[1]);
        itt=quadMap.find(qu);
        if(itt==quadMap.end())
        {
            // quad not in edgeMap so create a new one
            quadIndex=m_quad.size();
            quadMap[qu]=quadIndex;
            qu=helper::make_array<unsigned int>(v[0],v[1],v[2],v[3]);
            quadMap[qu]=quadIndex;
            m_quad.push_back(qu);
        }

        // Quad 5 :
        v[0]=h[3];
        v[1]=h[0];
        v[2]=h[4];
        v[3]=h[7];
        // sort v such that v[0] is the smallest one
        while ((v[0]>v[1]) || (v[0]>v[2]) || (v[0]>v[3]))
        {
            val=v[0];
            v[0]=v[1];
            v[1]=v[2];
            v[2]=v[3];
            v[3]=val;
        }
        // sort vertices in lexicographics order
        qu=helper::make_array<unsigned int>(v[0],v[3],v[2],v[1]);
        itt=quadMap.find(qu);
        if(itt==quadMap.end())
        {
            // quad not in edgeMap so create a new one
            quadIndex=m_quad.size();
            quadMap[qu]=quadIndex;
            qu=helper::make_array<unsigned int>(v[0],v[1],v[2],v[3]);
            quadMap[qu]=quadIndex;
            m_quad.push_back(qu);
        }
    }
    d_quad.endEdit();
}

void HexahedronSetTopologyContainer::createHexahedronQuadArray()
{
    if(!hasQuads())
        createQuadSetArray();

    if(hasHexahedronQuads())
        clearHexahedronQuads();

    m_hexahedronQuad.resize( getNumberOfHexahedra());

    for(unsigned int i = 0; i < m_hexahedron.size(); ++i)
    {
        Hexahedron &h=m_hexahedron[i];
        int quadIndex;

        // adding the 6 quads in the quad list of the ith hexahedron  i
        // Quad 0 :
        quadIndex=getQuadIndex(h[0],h[3],h[2],h[1]);
        assert(quadIndex!= -1);
        m_hexahedronQuad[i][0]=quadIndex;
        // Quad 1 :
        quadIndex=getQuadIndex(h[4],h[5],h[6],h[7]);
        assert(quadIndex!= -1);
        m_hexahedronQuad[i][1]=quadIndex;
        // Quad 2 :
        quadIndex=getQuadIndex(h[0],h[1],h[5],h[4]);
        assert(quadIndex!= -1);
        m_hexahedronQuad[i][2]=quadIndex;
        // Quad 3 :
        quadIndex=getQuadIndex(h[1],h[2],h[6],h[5]);
        assert(quadIndex!= -1);
        m_hexahedronQuad[i][3]=quadIndex;
        // Quad 4 :
        quadIndex=getQuadIndex(h[2],h[3],h[7],h[6]);
        assert(quadIndex!= -1);
        m_hexahedronQuad[i][4]=quadIndex;
        // Quad 5 :
        quadIndex=getQuadIndex(h[3],h[0],h[4],h[7]);
        assert(quadIndex!= -1);
        m_hexahedronQuad[i][5]=quadIndex;
    }
}

void HexahedronSetTopologyContainer::createHexahedronVertexShellArray()
{
    if(hasHexahedronVertexShell())
        clearHexahedronVertexShell();

    m_hexahedronVertexShell.resize( getNbPoints() );

    for(unsigned int i=0; i<m_hexahedron.size(); ++i)
    {
        // adding vertex i in the vertex shell
        for(unsigned int j=0; j<8; ++j)
            m_hexahedronVertexShell[ m_hexahedron[i][j]  ].push_back( i );
    }
}

void HexahedronSetTopologyContainer::createHexahedronEdgeShellArray ()
{
    if(!hasHexahedronEdges())
        createHexahedronEdgeArray();

    if(hasHexahedronEdgeShell())
        clearHexahedronEdgeShell();

    m_hexahedronEdgeShell.resize(getNumberOfEdges());

    for(unsigned int i=0; i<m_hexahedron.size(); ++i)
    {
        // adding edge i in the edge shell
        for(unsigned int j=0; j<12; ++j)
        {
            m_hexahedronEdgeShell[ m_hexahedronEdge[i][j] ].push_back( i );
        }
    }
}

void HexahedronSetTopologyContainer::createHexahedronQuadShellArray()
{
    if(!hasHexahedronQuads())
        createHexahedronQuadArray();

    if(hasHexahedronQuadShell())
        clearHexahedronQuadShell();

    m_hexahedronQuadShell.resize( getNumberOfQuads());

    for(unsigned int i=0; i<m_hexahedron.size(); ++i)
    {
        // adding quad i in the edge shell of both points
        for(unsigned int j=0; j<6; ++j)
        {
            m_hexahedronQuadShell[ m_hexahedronQuad[i][j] ].push_back( i );
        }
    }
}

const sofa::helper::vector<Hexahedron> &HexahedronSetTopologyContainer::getHexahedronArray()
{
    if(!hasHexahedra() && getNbPoints()>0)
    {
#ifndef NDEBUG
        sout << "Warning. [HexahedronSetTopologyContainer::getHexahedronArray] creating hexahedron array." << endl;
#endif
        createHexahedronSetArray();
    }

    return m_hexahedron;
}

int HexahedronSetTopologyContainer::getHexahedronIndex(PointID v1, PointID v2, PointID v3, PointID v4,
        PointID v5, PointID v6, PointID v7, PointID v8)
{
    if(!hasHexahedronVertexShell())
        createHexahedronVertexShellArray();

    sofa::helper::vector<unsigned int> set1 = getHexahedronVertexShell(v1);
    sofa::helper::vector<unsigned int> set2 = getHexahedronVertexShell(v2);
    sofa::helper::vector<unsigned int> set3 = getHexahedronVertexShell(v3);
    sofa::helper::vector<unsigned int> set4 = getHexahedronVertexShell(v4);
    sofa::helper::vector<unsigned int> set5 = getHexahedronVertexShell(v5);
    sofa::helper::vector<unsigned int> set6 = getHexahedronVertexShell(v6);
    sofa::helper::vector<unsigned int> set7 = getHexahedronVertexShell(v7);
    sofa::helper::vector<unsigned int> set8 = getHexahedronVertexShell(v8);

    sort(set1.begin(), set1.end());
    sort(set2.begin(), set2.end());
    sort(set3.begin(), set3.end());
    sort(set4.begin(), set4.end());
    sort(set5.begin(), set5.end());
    sort(set6.begin(), set6.end());
    sort(set7.begin(), set7.end());
    sort(set8.begin(), set8.end());

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

    sofa::helper::vector<unsigned int> out4(set5.size()+out3.size());
    sofa::helper::vector<unsigned int>::iterator result4;
    result4 = std::set_intersection(set5.begin(),set5.end(),out3.begin(),out3.end(),out4.begin());
    out4.erase(result4,out4.end());

    sofa::helper::vector<unsigned int> out5(set6.size()+out4.size());
    sofa::helper::vector<unsigned int>::iterator result5;
    result5 = std::set_intersection(set6.begin(),set6.end(),out4.begin(),out4.end(),out5.begin());
    out5.erase(result5,out5.end());

    sofa::helper::vector<unsigned int> out6(set7.size()+out5.size());
    sofa::helper::vector<unsigned int>::iterator result6;
    result6 = std::set_intersection(set7.begin(),set7.end(),out5.begin(),out5.end(),out6.begin());
    out6.erase(result6,out6.end());

    sofa::helper::vector<unsigned int> out7(set8.size()+out6.size());
    sofa::helper::vector<unsigned int>::iterator result7;
    result7 = std::set_intersection(set8.begin(),set8.end(),out6.begin(),out6.end(),out7.begin());
    out7.erase(result7,out7.end());

    assert(out7.size()==0 || out7.size()==1);
    if(out7.size()==1)
        return (int) (out7[0]);
    else
        return -1;
}

const Hexahedron &HexahedronSetTopologyContainer::getHexahedron(const unsigned int i)
{
    if(!hasHexahedra())
        createHexahedronSetArray();

    return m_hexahedron[i];
}

unsigned int HexahedronSetTopologyContainer::getNumberOfHexahedra() const
{
    return m_hexahedron.size();
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &HexahedronSetTopologyContainer::getHexahedronVertexShellArray()
{
    if(!hasHexahedronVertexShell())
        createHexahedronVertexShellArray();

    return m_hexahedronVertexShell;
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &HexahedronSetTopologyContainer::getHexahedronEdgeShellArray()
{
    if(!hasHexahedronEdgeShell())
        createHexahedronEdgeShellArray();

    return m_hexahedronEdgeShell;
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &HexahedronSetTopologyContainer::getHexahedronQuadShellArray()
{
    if(!hasHexahedronQuadShell())
        createHexahedronQuadShellArray();

    return m_hexahedronQuadShell;
}

const sofa::helper::vector< HexahedronEdges> &HexahedronSetTopologyContainer::getHexahedronEdgeArray()
{
    if(!hasHexahedronEdges())
        createHexahedronEdgeArray();

    return m_hexahedronEdge;
}

Edge HexahedronSetTopologyContainer::getLocalHexahedronEdges (const unsigned int i) const
{
    assert(i<12);
    return Edge (hexahedronEdgeArray[i][0], hexahedronEdgeArray[i][1]);
}

const sofa::helper::vector< HexahedronQuads> &HexahedronSetTopologyContainer::getHexahedronQuadArray()
{
    if(!hasHexahedronQuads())
        createHexahedronQuadArray();

    return m_hexahedronQuad;
}

const sofa::helper::vector< unsigned int > &HexahedronSetTopologyContainer::getHexahedronVertexShell(const unsigned int i)
{
    if(!hasHexahedronVertexShell())
        createHexahedronVertexShellArray();

    assert(i < m_hexahedronVertexShell.size());

    return m_hexahedronVertexShell[i];
}

const sofa::helper::vector< unsigned int > &HexahedronSetTopologyContainer::getHexahedronEdgeShell(const unsigned int i)
{
    if(!hasHexahedronEdgeShell())
        createHexahedronEdgeShellArray();

    assert(i < m_hexahedronEdgeShell.size());

    return m_hexahedronEdgeShell[i];
}

const sofa::helper::vector< unsigned int > &HexahedronSetTopologyContainer::getHexahedronQuadShell(const unsigned int i)
{
    if(!hasHexahedronQuadShell())
        createHexahedronQuadShellArray();

    assert(i < m_hexahedronQuadShell.size());

    return m_hexahedronQuadShell[i];
}

const HexahedronEdges &HexahedronSetTopologyContainer::getHexahedronEdges(const unsigned int i)
{
    if(!hasHexahedronEdges())
        createHexahedronEdgeArray();

    assert(i < m_hexahedronEdge.size());

    return m_hexahedronEdge[i];
}

const HexahedronQuads &HexahedronSetTopologyContainer::getHexahedronQuads(const unsigned int i)
{
    if(!hasHexahedronQuads())
        createHexahedronQuadArray();

    assert(i < m_hexahedronQuad.size());

    return m_hexahedronQuad[i];
}

int HexahedronSetTopologyContainer::getVertexIndexInHexahedron(Hexahedron &t,unsigned int vertexIndex) const
{
    if(t[0]==vertexIndex)
        return 0;
    else if(t[1]==vertexIndex)
        return 1;
    else if(t[2]==vertexIndex)
        return 2;
    else if(t[3]==vertexIndex)
        return 3;
    else if(t[4]==vertexIndex)
        return 4;
    else if(t[5]==vertexIndex)
        return 5;
    else if(t[6]==vertexIndex)
        return 6;
    else if(t[7]==vertexIndex)
        return 7;
    else
        return -1;
}

int HexahedronSetTopologyContainer::getEdgeIndexInHexahedron(const HexahedronEdges &t,
        const unsigned int edgeIndex) const
{
    if(t[0]==edgeIndex)
        return 0;
    else if(t[1]==edgeIndex)
        return 1;
    else if(t[2]==edgeIndex)
        return 2;
    else if(t[3]==edgeIndex)
        return 3;
    else if(t[4]==edgeIndex)
        return 4;
    else if(t[5]==edgeIndex)
        return 5;
    else if(t[6]==edgeIndex)
        return 6;
    else if(t[7]==edgeIndex)
        return 7;
    else if(t[8]==edgeIndex)
        return 8;
    else if(t[9]==edgeIndex)
        return 9;
    else if(t[10]==edgeIndex)
        return 10;
    else if(t[11]==edgeIndex)
        return 11;
    else
        return -1;
}

int HexahedronSetTopologyContainer::getQuadIndexInHexahedron(const HexahedronQuads &t,
        const unsigned int quadIndex) const
{
    if(t[0]==quadIndex)
        return 0;
    else if(t[1]==quadIndex)
        return 1;
    else if(t[2]==quadIndex)
        return 2;
    else if(t[3]==quadIndex)
        return 3;
    else if(t[4]==quadIndex)
        return 4;
    else if(t[5]==quadIndex)
        return 5;
    else
        return -1;
}

sofa::helper::vector< unsigned int > &HexahedronSetTopologyContainer::getHexahedronEdgeShellForModification(const unsigned int i)
{
    if(!hasHexahedronEdgeShell())
        createHexahedronEdgeShellArray();

    assert( i < m_hexahedronEdgeShell.size());

    return m_hexahedronEdgeShell[i];
}

sofa::helper::vector< unsigned int > &HexahedronSetTopologyContainer::getHexahedronVertexShellForModification(const unsigned int i)
{
    if(!hasHexahedronVertexShell())
        createHexahedronVertexShellArray();

    assert( i < m_hexahedronVertexShell.size());

    return m_hexahedronVertexShell[i];
}

bool HexahedronSetTopologyContainer::checkTopology() const
{
#ifndef NDEBUG
    bool ret = true;

    if(hasHexahedronVertexShell())
    {
        for(unsigned int i=0; i<m_hexahedronVertexShell.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tvs = m_hexahedronVertexShell[i];
            for(unsigned int j=0; j<tvs.size(); ++j)
            {
                bool check_hexa_vertex_shell = (m_hexahedron[tvs[j]][0]==i)
                        || (m_hexahedron[tvs[j]][1]==i)
                        || (m_hexahedron[tvs[j]][2]==i)
                        || (m_hexahedron[tvs[j]][3]==i)
                        || (m_hexahedron[tvs[j]][4]==i)
                        ||  (m_hexahedron[tvs[j]][5]==i)
                        || (m_hexahedron[tvs[j]][6]==i)
                        || (m_hexahedron[tvs[j]][7]==i);

                if(!check_hexa_vertex_shell)
                {
                    std::cout << "*** CHECK FAILED : check_hexa_vertex_shell, i = " << i << " , j = " << j << std::endl;
                    ret = false;
                }
            }
        }
    }

    if(hasHexahedronEdgeShell())
    {
        for(unsigned int i=0; i<m_hexahedronEdgeShell.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tes=m_hexahedronEdgeShell[i];
            for(unsigned int j=0; j<tes.size(); ++j)
            {
                bool check_hexa_edge_shell =   (m_hexahedronEdge[tes[j]][0]==i)
                        || (m_hexahedronEdge[tes[j]][1]==i)
                        || (m_hexahedronEdge[tes[j]][2]==i)
                        || (m_hexahedronEdge[tes[j]][3]==i)
                        || (m_hexahedronEdge[tes[j]][4]==i)
                        || (m_hexahedronEdge[tes[j]][5]==i)
                        || (m_hexahedronEdge[tes[j]][6]==i)
                        || (m_hexahedronEdge[tes[j]][7]==i)
                        || (m_hexahedronEdge[tes[j]][8]==i)
                        || (m_hexahedronEdge[tes[j]][9]==i)
                        || (m_hexahedronEdge[tes[j]][10]==i)
                        || (m_hexahedronEdge[tes[j]][11]==i);
                if(!check_hexa_edge_shell)
                {
                    std::cout << "*** CHECK FAILED : check_hexa_edge_shell, i = " << i << " , j = " << j << std::endl;
                    ret = false;
                }
            }
        }
    }

    if(hasHexahedronQuadShell())
    {
        for(unsigned int i=0; i<m_hexahedronQuadShell.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tes=m_hexahedronQuadShell[i];
            for(unsigned int j=0; j<tes.size(); ++j)
            {
                bool check_hexa_quad_shell =   (m_hexahedronQuad[tes[j]][0]==i)
                        || (m_hexahedronQuad[tes[j]][1]==i)
                        || (m_hexahedronQuad[tes[j]][2]==i)
                        || (m_hexahedronQuad[tes[j]][3]==i)
                        || (m_hexahedronQuad[tes[j]][4]==i)
                        || (m_hexahedronQuad[tes[j]][5]==i);
                if(!check_hexa_quad_shell)
                {
                    std::cout << "*** CHECK FAILED : check_hexa_quad_shell, i = " << i << " , j = " << j << std::endl;
                    ret = false;
                }
            }
        }
    }

    return ret && QuadSetTopologyContainer::checkTopology();
#else
    return true;
#endif
}

bool HexahedronSetTopologyContainer::hasHexahedra() const
{
    return !m_hexahedron.empty();
}

bool HexahedronSetTopologyContainer::hasHexahedronEdges() const
{
    return !m_hexahedronEdge.empty();
}

bool HexahedronSetTopologyContainer::hasHexahedronQuads() const
{
    return !m_hexahedronQuad.empty();
}

bool HexahedronSetTopologyContainer::hasHexahedronVertexShell() const
{
    return !m_hexahedronVertexShell.empty();
}

bool HexahedronSetTopologyContainer::hasHexahedronEdgeShell() const
{
    return !m_hexahedronEdgeShell.empty();
}

bool HexahedronSetTopologyContainer::hasHexahedronQuadShell() const
{
    return !m_hexahedronQuadShell.empty();
}

void HexahedronSetTopologyContainer::clearHexahedra()
{
    d_hexahedron.beginEdit();
    m_hexahedron.clear();
    d_hexahedron.endEdit();
}

void HexahedronSetTopologyContainer::clearHexahedronEdges()
{
    m_hexahedronEdge.clear();
}

void HexahedronSetTopologyContainer::clearHexahedronQuads()
{
    m_hexahedronQuad.clear();
}

void HexahedronSetTopologyContainer::clearHexahedronVertexShell()
{
    m_hexahedronVertexShell.clear();
}

void HexahedronSetTopologyContainer::clearHexahedronEdgeShell()
{
    m_hexahedronEdgeShell.clear();
}

void HexahedronSetTopologyContainer::clearHexahedronQuadShell()
{
    m_hexahedronQuadShell.clear();
}

void HexahedronSetTopologyContainer::clear()
{
    clearHexahedronVertexShell();
    clearHexahedronEdgeShell();
    clearHexahedronQuadShell();
    clearHexahedronQuads();
    clearHexahedronEdges();
    clearHexahedra();

    QuadSetTopologyContainer::clear();
}

} // namespace topology

} // namespace component

} // namespace sofa

