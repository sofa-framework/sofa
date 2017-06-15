/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <SofaBaseTopology/HexahedronSetTopologyContainer.h>
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

SOFA_DECL_CLASS(HexahedronSetTopologyContainer)
int HexahedronSetTopologyContainerClass = core::RegisterObject("Hexahedron set topology container")
        .add< HexahedronSetTopologyContainer >()
        ;

const unsigned int edgesInHexahedronArray[12][2]= {{0,1},{0,3},{0,4},{1,2},{1,5},{2,3},{2,6},{3,7},{4,5},{4,7},{5,6},{6,7}};
///convention quads in hexa (orientation interior)
const unsigned int quadsInHexahedronArray[6][4]= {{0,1,2,3}, {4,7,6,5}, {1,0,4,5},{1,5,6,2},  {2,6,7,3}, {0,3,7,4}};

const unsigned int verticesInHexahedronArray[2][2][2]=  {{{0,4},{3,7}},{{1,5},{2,6}}};

HexahedronSetTopologyContainer::HexahedronSetTopologyContainer()
    : QuadSetTopologyContainer()
	, d_createQuadArray(initData(&d_createQuadArray, bool(false),"createQuadArray", "Force the creation of a set of quads associated with the hexahedra"))
    , d_hexahedron(initData(&d_hexahedron, "hexahedra", "List of hexahedron indices"))
{
    addAlias(&d_hexahedron, "hexas");
}


void HexahedronSetTopologyContainer::addHexa( int a, int b, int c, int d, int e, int f, int g, int h )
{
    helper::WriteAccessor< Data< sofa::helper::vector<Hexahedron> > > m_hexahedron = d_hexahedron;
    m_hexahedron.push_back(Hexahedron(a,b,c,d,e,f,g,h));
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
    d_hexahedron.updateIfDirty(); // make sure m_hexahedron is up to date
	// eventually force the creation of quads
	if (d_createQuadArray.getValue())
		createQuadSetArray();
}



void HexahedronSetTopologyContainer::createHexahedronSetArray()
{
#ifndef NDEBUG
    serr << "Error. [createHexahedronSetArray] This method must be implemented by a child topology." << sendl;
#endif
}

void HexahedronSetTopologyContainer::createEdgeSetArray()
{

    if(hasEdges())
    {
        EdgeSetTopologyContainer::clear();

        clearEdgesInQuad();
        clearQuadsAroundEdge();

        clearEdgesInHexahedron();
        clearHexahedraAroundEdge();
    }

    // create a temporary map to find redundant edges
    std::map<Edge,unsigned int> edgeMap;
    helper::WriteAccessor< Data< sofa::helper::vector<Edge> > > m_edge = d_edge;
    helper::ReadAccessor< Data< sofa::helper::vector<Hexahedron> > > m_hexahedron = d_hexahedron;

    /// create the m_edge array at the same time than it fills the m_edgesInHexahedron array
    for(unsigned int i=0; i<m_hexahedron.size(); ++i)
    {
        const Hexahedron &t = m_hexahedron[i];
        for(unsigned int j=0; j<12; ++j)
        {
            unsigned int v1 = t[edgesInHexahedronArray[j][0]];
            unsigned int v2 = t[edgesInHexahedronArray[j][1]];
            const Edge e((v1<v2) ? Edge(v1,v2) : Edge(v2,v1));

            if(edgeMap.find(e)==edgeMap.end())
            {
                // edge not in edgeMap so create a new one
                const unsigned int edgeIndex = (unsigned int)edgeMap.size();
                edgeMap[e] = edgeIndex;
                m_edge.push_back(e);
            }
        }
    }
}

void HexahedronSetTopologyContainer::createEdgesInHexahedronArray()
{
    if(!hasEdges())
        createEdgeSetArray();

    if(hasEdgesInHexahedron())
        clearEdgesInHexahedron();

    m_edgesInHexahedron.resize( getNumberOfHexahedra());
    helper::ReadAccessor< Data< sofa::helper::vector<Hexahedron> > > m_hexahedron = d_hexahedron;

    for(unsigned int i=0; i<m_hexahedron.size(); ++i)
    {
        const Hexahedron &t = m_hexahedron[i];

        // adding edge i in the edge shell of both points
        for(unsigned int j=0; j<12; ++j)
        {
            const int edgeIndex = getEdgeIndex(t[edgesInHexahedronArray[j][0]],
                    t[edgesInHexahedronArray[j][1]]);
            m_edgesInHexahedron[i][j] = edgeIndex;
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
        clearQuadsInHexahedron();
        clearHexahedraAroundQuad();
    }

    // create a temporary map to find redundant quads
    std::map<Quad,unsigned int> quadMap;
    helper::WriteAccessor< Data< sofa::helper::vector<Quad> > > m_quad = d_quad;
    helper::ReadAccessor< Data< sofa::helper::vector<Hexahedron> > > m_hexahedron = d_hexahedron;

    for(unsigned int i=0; i<m_hexahedron.size(); ++i)
    {
        const Hexahedron &h = m_hexahedron[i];

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
        unsigned int quadIndex;
        Quad qu(v[0],v[3],v[2],v[1]);
        std::map<Quad,unsigned int>::iterator itt = quadMap.find(qu);
        if(itt==quadMap.end())
        {
            // quad not in edgeMap so create a new one
            quadIndex=(unsigned int)m_quad.size();
            quadMap[qu]=quadIndex;
            qu=Quad(v[0],v[1],v[2],v[3]);
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
        qu=Quad(v[0],v[3],v[2],v[1]);
        itt=quadMap.find(qu);
        if(itt==quadMap.end())
        {
            // quad not in edgeMap so create a new one
            quadIndex=(unsigned int)m_quad.size();
            quadMap[qu]=quadIndex;
            qu=Quad(v[0],v[1],v[2],v[3]);
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
        qu=Quad(v[0],v[3],v[2],v[1]);
        itt=quadMap.find(qu);
        if(itt==quadMap.end())
        {
            // quad not in edgeMap so create a new one
            quadIndex=(unsigned int)m_quad.size();
            quadMap[qu]=quadIndex;
            qu=Quad(v[0],v[1],v[2],v[3]);
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
        qu=Quad(v[0],v[3],v[2],v[1]);
        itt=quadMap.find(qu);
        if(itt==quadMap.end())
        {
            // quad not in edgeMap so create a new one
            quadIndex=(unsigned int)m_quad.size();
            quadMap[qu]=quadIndex;
            qu=Quad(v[0],v[1],v[2],v[3]);
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
        qu=Quad(v[0],v[3],v[2],v[1]);
        itt=quadMap.find(qu);
        if(itt==quadMap.end())
        {
            // quad not in edgeMap so create a new one
            quadIndex=(unsigned int)m_quad.size();
            quadMap[qu]=quadIndex;
            qu=Quad(v[0],v[1],v[2],v[3]);
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
        qu=Quad(v[0],v[3],v[2],v[1]);
        itt=quadMap.find(qu);
        if(itt==quadMap.end())
        {
            // quad not in edgeMap so create a new one
            quadIndex=(unsigned int)m_quad.size();
            quadMap[qu]=quadIndex;
            qu=Quad(v[0],v[1],v[2],v[3]);
            quadMap[qu]=quadIndex;
            m_quad.push_back(qu);
        }
    }
    d_quad.endEdit();
}

void HexahedronSetTopologyContainer::createQuadsInHexahedronArray()
{
    if(!hasQuads())
        createQuadSetArray();

    if(hasQuadsInHexahedron())
        clearQuadsInHexahedron();

    m_quadsInHexahedron.resize( getNumberOfHexahedra());
    helper::ReadAccessor< Data< sofa::helper::vector<Hexahedron> > > m_hexahedron = d_hexahedron;

    for(unsigned int i = 0; i < getNumberOfHexahedra(); ++i)
    {
        const Hexahedron &h=m_hexahedron[i];
        int quadIndex;

        // adding the 6 quads in the quad list of the ith hexahedron  i
        // Quad 0 :
        quadIndex=getQuadIndex(h[0],h[3],h[2],h[1]);
        assert(quadIndex!= -1);
        m_quadsInHexahedron[i][0]=quadIndex;
        // Quad 1 :
        quadIndex=getQuadIndex(h[4],h[5],h[6],h[7]);
        assert(quadIndex!= -1);
        m_quadsInHexahedron[i][1]=quadIndex;
        // Quad 2 :
        quadIndex=getQuadIndex(h[0],h[1],h[5],h[4]);
        assert(quadIndex!= -1);
        m_quadsInHexahedron[i][2]=quadIndex;
        // Quad 3 :
        quadIndex=getQuadIndex(h[1],h[2],h[6],h[5]);
        assert(quadIndex!= -1);
        m_quadsInHexahedron[i][3]=quadIndex;
        // Quad 4 :
        quadIndex=getQuadIndex(h[2],h[3],h[7],h[6]);
        assert(quadIndex!= -1);
        m_quadsInHexahedron[i][4]=quadIndex;
        // Quad 5 :
        quadIndex=getQuadIndex(h[3],h[0],h[4],h[7]);
        assert(quadIndex!= -1);
        m_quadsInHexahedron[i][5]=quadIndex;
    }
}

void HexahedronSetTopologyContainer::createHexahedraAroundVertexArray()
{
    if(hasHexahedraAroundVertex())
        clearHexahedraAroundVertex();

    m_hexahedraAroundVertex.resize( getNbPoints() );
    helper::ReadAccessor< Data< sofa::helper::vector<Hexahedron> > > m_hexahedron = d_hexahedron;

    for(unsigned int i=0; i<m_hexahedron.size(); ++i)
    {
        // adding vertex i in the vertex shell
        for(unsigned int j=0; j<8; ++j)
            m_hexahedraAroundVertex[ m_hexahedron[i][j]  ].push_back( i );
    }
}

void HexahedronSetTopologyContainer::createHexahedraAroundEdgeArray ()
{
    if(!hasEdgesInHexahedron())
        createEdgesInHexahedronArray();

    if(hasHexahedraAroundEdge())
        clearHexahedraAroundEdge();

    m_hexahedraAroundEdge.resize(getNumberOfEdges());

    for(unsigned int i=0; i<getNumberOfHexahedra(); ++i)
    {
        // adding edge i in the edge shell
        for(unsigned int j=0; j<12; ++j)
        {
            m_hexahedraAroundEdge[ m_edgesInHexahedron[i][j] ].push_back( i );
        }
    }
}

void HexahedronSetTopologyContainer::createHexahedraAroundQuadArray()
{
    if(!hasQuadsInHexahedron())
        createQuadsInHexahedronArray();

    if(hasHexahedraAroundQuad())
        clearHexahedraAroundQuad();

    m_hexahedraAroundQuad.resize( getNumberOfQuads());

    for(unsigned int i=0; i<getNumberOfHexahedra(); ++i)
    {
        // adding quad i in the edge shell of both points
        for(unsigned int j=0; j<6; ++j)
        {
            m_hexahedraAroundQuad[ m_quadsInHexahedron[i][j] ].push_back( i );
        }
    }
}

const sofa::helper::vector<HexahedronSetTopologyContainer::Hexahedron> &HexahedronSetTopologyContainer::getHexahedronArray()
{
    if(!hasHexahedra() && getNbPoints()>0)
    {
#ifndef NDEBUG
        sout << "Warning. [HexahedronSetTopologyContainer::getHexahedronArray] creating hexahedron array." << sendl;
#endif
        createHexahedronSetArray();
    }

    return d_hexahedron.getValue ();
}

int HexahedronSetTopologyContainer::getHexahedronIndex(PointID v1, PointID v2, PointID v3, PointID v4,
        PointID v5, PointID v6, PointID v7, PointID v8)
{
    if(!hasHexahedraAroundVertex())
        createHexahedraAroundVertexArray();

    sofa::helper::vector<unsigned int> set1 = getHexahedraAroundVertex(v1);
    sofa::helper::vector<unsigned int> set2 = getHexahedraAroundVertex(v2);
    sofa::helper::vector<unsigned int> set3 = getHexahedraAroundVertex(v3);
    sofa::helper::vector<unsigned int> set4 = getHexahedraAroundVertex(v4);
    sofa::helper::vector<unsigned int> set5 = getHexahedraAroundVertex(v5);
    sofa::helper::vector<unsigned int> set6 = getHexahedraAroundVertex(v6);
    sofa::helper::vector<unsigned int> set7 = getHexahedraAroundVertex(v7);
    sofa::helper::vector<unsigned int> set8 = getHexahedraAroundVertex(v8);

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

const HexahedronSetTopologyContainer::Hexahedron HexahedronSetTopologyContainer::getHexahedron(HexaID i)
{
    if(!hasHexahedra())
        createHexahedronSetArray();

    return (d_hexahedron.getValue())[i];
}

unsigned int HexahedronSetTopologyContainer::getNumberOfHexahedra() const
{
    return (unsigned int)d_hexahedron.getValue().size();
}


unsigned int HexahedronSetTopologyContainer::getNumberOfElements() const
{
    return this->getNumberOfHexahedra();
}


const sofa::helper::vector< sofa::helper::vector<unsigned int> > &HexahedronSetTopologyContainer::getHexahedraAroundVertexArray()
{
    if(!hasHexahedraAroundVertex())
        createHexahedraAroundVertexArray();

    return m_hexahedraAroundVertex;
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &HexahedronSetTopologyContainer::getHexahedraAroundEdgeArray()
{
    if(!hasHexahedraAroundEdge())
        createHexahedraAroundEdgeArray();

    return m_hexahedraAroundEdge;
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &HexahedronSetTopologyContainer::getHexahedraAroundQuadArray()
{
    if(!hasHexahedraAroundQuad())
        createHexahedraAroundQuadArray();

    return m_hexahedraAroundQuad;
}

const sofa::helper::vector< HexahedronSetTopologyContainer::EdgesInHexahedron> &HexahedronSetTopologyContainer::getEdgesInHexahedronArray()
{
    if(!hasEdgesInHexahedron())
        createEdgesInHexahedronArray();

    return m_edgesInHexahedron;
}

HexahedronSetTopologyContainer::Edge HexahedronSetTopologyContainer::getLocalEdgesInHexahedron (const EdgeID i) const
{
    assert(i<12);
    return Edge (edgesInHexahedronArray[i][0], edgesInHexahedronArray[i][1]);
}


HexahedronSetTopologyContainer::Quad HexahedronSetTopologyContainer::getLocalQuadsInHexahedron (const QuadID i) const
{
    assert(i<6);
    return Quad (quadsInHexahedronArray[i][0],
            quadsInHexahedronArray[i][1],
            quadsInHexahedronArray[i][2],
            quadsInHexahedronArray[i][3]);
}

 unsigned int HexahedronSetTopologyContainer::getLocalIndexFromBinaryIndex(const HexahedronBinaryIndex bi) const 
 {
	 return(verticesInHexahedronArray[bi[0]][bi[1]][bi[2]]);
 }
 HexahedronSetTopologyContainer::HexahedronBinaryIndex HexahedronSetTopologyContainer::getBinaryIndexFromLocalIndex(const unsigned int li) const 
 {
	 HexahedronBinaryIndex bi;
	 if (li==0)
		 bi[0]=0;
	 else
		 bi[0]=1-(((li-1)&2)/2);
	 bi[1]=(li&2)/2;
	 bi[2]=(li&4)/4;
	return bi;
 }
QuadSetTopologyContainer::QuadID HexahedronSetTopologyContainer::getNextAdjacentQuad(const HexaID _hexaID, const QuadID _quadID, const EdgeID _edgeID)
{
    assert(_hexaID < d_hexahedron.getValue().size());
    assert(_quadID<6);
    assert(_edgeID<12);

    EdgeID the_edgeID = this->getEdgesInHexahedron(_hexaID)[_edgeID];
    const QuadsAroundEdge QaroundE = this->getQuadsAroundEdge(the_edgeID);

    const QuadsInHexahedron QinH = this->getQuadsInHexahedron(_hexaID);
    QuadID the_quadID = QinH[_quadID];
    QuadID nextQuad = 0;

    if (QaroundE.size() < 2)
    {
#ifndef NDEBUG
        serr << "Error: getNextAdjacentQuad: no quad around edge: " << the_edgeID << sendl;
#endif
        return nextQuad;
    }
    else if (QaroundE.size() == 2)
    {
        if (QaroundE[0] == the_quadID)
            return (this->getQuadIndexInHexahedron(QinH, QaroundE[1]));
        else
            return (this->getQuadIndexInHexahedron(QinH, QaroundE[0]));
    }
    else
    {
        for (unsigned int i=0; i<QaroundE.size(); ++i)
        {
            int res = this->getQuadIndexInHexahedron(QinH, QaroundE[i]);
            if (res != -1 && QaroundE[i] != the_quadID)
                return (unsigned int)res;
        }
    }

    return nextQuad;
}


const sofa::helper::vector< QuadSetTopologyContainer::QuadsInHexahedron> &HexahedronSetTopologyContainer::getQuadsInHexahedronArray()
{
    if(!hasQuadsInHexahedron())
        createQuadsInHexahedronArray();

    return m_quadsInHexahedron;
}

const sofa::helper::vector< unsigned int > &HexahedronSetTopologyContainer::getHexahedraAroundVertex(const unsigned int i)
{
    if(!hasHexahedraAroundVertex())
        createHexahedraAroundVertexArray();

    assert(i < m_hexahedraAroundVertex.size());

    return m_hexahedraAroundVertex[i];
}

const sofa::helper::vector< unsigned int > &HexahedronSetTopologyContainer::getHexahedraAroundEdge(const unsigned int i)
{
    if(!hasHexahedraAroundEdge())
        createHexahedraAroundEdgeArray();

    assert(i < m_hexahedraAroundEdge.size());

    return m_hexahedraAroundEdge[i];
}

const sofa::helper::vector< unsigned int > &HexahedronSetTopologyContainer::getHexahedraAroundQuad(const unsigned int i)
{
    if(!hasHexahedraAroundQuad())
        createHexahedraAroundQuadArray();

    assert(i < m_hexahedraAroundQuad.size());

    return m_hexahedraAroundQuad[i];
}

const QuadSetTopologyContainer::EdgesInHexahedron &HexahedronSetTopologyContainer::getEdgesInHexahedron(const QuadID i)
{
    if(!hasEdgesInHexahedron())
        createEdgesInHexahedronArray();

    assert(i < m_edgesInHexahedron.size());

    return m_edgesInHexahedron[i];
}

const QuadSetTopologyContainer::QuadsInHexahedron &HexahedronSetTopologyContainer::getQuadsInHexahedron(const QuadID i)
{
    if(!hasQuadsInHexahedron())
        createQuadsInHexahedronArray();

    assert(i < m_quadsInHexahedron.size());

    return m_quadsInHexahedron[i];
}

int HexahedronSetTopologyContainer::getVertexIndexInHexahedron(const Hexahedron &t,PointID vertexIndex) const
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

int HexahedronSetTopologyContainer::getEdgeIndexInHexahedron(const EdgesInHexahedron &t,
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

int HexahedronSetTopologyContainer::getQuadIndexInHexahedron(const QuadsInHexahedron &t,
        const QuadID quadIndex) const
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

HexahedronSetTopologyContainer::HexahedraAroundEdge &HexahedronSetTopologyContainer::getHexahedraAroundEdgeForModification(const EdgeID i)
{
    if(!hasHexahedraAroundEdge())
        createHexahedraAroundEdgeArray();

    assert( i < m_hexahedraAroundEdge.size());

    return m_hexahedraAroundEdge[i];
}

HexahedronSetTopologyContainer::HexahedraAroundVertex &HexahedronSetTopologyContainer::getHexahedraAroundVertexForModification(const PointID i)
{
    if(!hasHexahedraAroundVertex())
        createHexahedraAroundVertexArray();

    assert( i < m_hexahedraAroundVertex.size());

    return m_hexahedraAroundVertex[i];
}

HexahedronSetTopologyContainer::HexahedraAroundQuad &HexahedronSetTopologyContainer::getHexahedraAroundQuadForModification(const QuadID i)
{
    if(!hasHexahedraAroundQuad())
        createHexahedraAroundQuadArray();

    assert( i < m_hexahedraAroundQuad.size());

    return m_hexahedraAroundQuad[i];
}


bool HexahedronSetTopologyContainer::checkTopology() const
{
#ifndef NDEBUG
    bool ret = true;
    helper::ReadAccessor< Data< sofa::helper::vector<Hexahedron> > > m_hexahedron = d_hexahedron;
    if(hasHexahedraAroundVertex())
    {
        for(unsigned int i=0; i<m_hexahedraAroundVertex.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tvs = m_hexahedraAroundVertex[i];
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

    if(hasHexahedraAroundEdge())
    {
        for(unsigned int i=0; i<m_hexahedraAroundEdge.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tes=m_hexahedraAroundEdge[i];
            for(unsigned int j=0; j<tes.size(); ++j)
            {
                bool check_hexa_edge_shell =   (m_edgesInHexahedron[tes[j]][0]==i)
                        || (m_edgesInHexahedron[tes[j]][1]==i)
                        || (m_edgesInHexahedron[tes[j]][2]==i)
                        || (m_edgesInHexahedron[tes[j]][3]==i)
                        || (m_edgesInHexahedron[tes[j]][4]==i)
                        || (m_edgesInHexahedron[tes[j]][5]==i)
                        || (m_edgesInHexahedron[tes[j]][6]==i)
                        || (m_edgesInHexahedron[tes[j]][7]==i)
                        || (m_edgesInHexahedron[tes[j]][8]==i)
                        || (m_edgesInHexahedron[tes[j]][9]==i)
                        || (m_edgesInHexahedron[tes[j]][10]==i)
                        || (m_edgesInHexahedron[tes[j]][11]==i);
                if(!check_hexa_edge_shell)
                {
                    std::cout << "*** CHECK FAILED : check_hexa_edge_shell, i = " << i << " , j = " << j << std::endl;
                    ret = false;
                }
            }
        }
    }

    if(hasHexahedraAroundQuad())
    {
        for(unsigned int i=0; i<m_hexahedraAroundQuad.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tes=m_hexahedraAroundQuad[i];
            for(unsigned int j=0; j<tes.size(); ++j)
            {
                bool check_hexa_quad_shell =   (m_quadsInHexahedron[tes[j]][0]==i)
                        || (m_quadsInHexahedron[tes[j]][1]==i)
                        || (m_quadsInHexahedron[tes[j]][2]==i)
                        || (m_quadsInHexahedron[tes[j]][3]==i)
                        || (m_quadsInHexahedron[tes[j]][4]==i)
                        || (m_quadsInHexahedron[tes[j]][5]==i);
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




/// Get information about connexity of the mesh
/// @{

bool HexahedronSetTopologyContainer::checkConnexity()
{

    unsigned int nbr = this->getNbHexahedra();

    if (nbr == 0)
    {
#ifndef NDEBUG
        serr << "Warning. [HexahedronSetTopologyContainer::checkConnexity] Can't compute connexity as there are no Hexahedra" << sendl;
#endif
        return false;
    }

    VecHexaID elemAll = this->getConnectedElement(0);

    if (elemAll.size() != nbr)
    {
        serr << "Warning: in computing connexity, Hexahedra are missings. There is more than one connexe component." << sendl;
        return false;
    }

    return true;
}


unsigned int HexahedronSetTopologyContainer::getNumberOfConnectedComponent()
{
    unsigned int nbr = this->getNbHexahedra();

    if (nbr == 0)
    {
#ifndef NDEBUG
        serr << "Warning. [HexahedronSetTopologyContainer::getNumberOfConnectedComponent] Can't compute connexity as there are no Hexahedra" << sendl;
#endif
        return 0;
    }

    VecHexaID elemAll = this->getConnectedElement(0);
    unsigned int cpt = 1;

    while (elemAll.size() < nbr)
    {
        std::sort(elemAll.begin(), elemAll.end());
        HexaID other_HexaID = (HexaID)elemAll.size();

        for (HexaID i = 0; i<elemAll.size(); ++i)
            if (elemAll[i] != i)
            {
                other_HexaID = i;
                break;
            }

        VecHexaID elemTmp = this->getConnectedElement(other_HexaID);
        cpt++;

        elemAll.insert(elemAll.begin(), elemTmp.begin(), elemTmp.end());
    }

    return cpt;
}


const HexahedronSetTopologyContainer::VecHexaID HexahedronSetTopologyContainer::getConnectedElement(HexaID elem)
{
    if(!hasHexahedraAroundVertex())	// this method should only be called when the shell array exists
    {
#ifndef NDEBUG
        serr << "Warning. [HexahedronSetTopologyContainer::getConnectedElement] hexahedron vertex shell array is empty." << sendl;
#endif
        createHexahedraAroundVertexArray();
    }

    VecHexaID elemAll;
    VecHexaID elemOnFront, elemPreviousFront, elemNextFront;
    bool end = false;
    unsigned int cpt = 0;
    const unsigned int nbr = this->getNbHexahedra();

    // init algo
    elemAll.push_back(elem);
    elemOnFront.push_back(elem);
    elemPreviousFront.clear();
    cpt++;

    while (!end && cpt < nbr)
    {
        // First Step - Create new region
        elemNextFront = this->getElementAroundElements(elemOnFront); // for each HexaID on the propagation front

        // Second Step - Avoid backward direction
        for (unsigned int i = 0; i<elemNextFront.size(); ++i)
        {
            bool find = false;
            HexaID id = elemNextFront[i];

            for (HexaID j = 0; j<elemAll.size(); ++j)
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
        cpt += (unsigned int)elemPreviousFront.size();

        if (elemPreviousFront.empty())
        {
            end = true;
#ifndef NDEBUG
            serr << "Loop for computing connexity has reach end." << sendl;
#endif
        }

        // iterate
        elemOnFront = elemPreviousFront;
        elemPreviousFront.clear();
    }

    return elemAll;
}


const HexahedronSetTopologyContainer::VecHexaID HexahedronSetTopologyContainer::getElementAroundElement(HexaID elem)
{
    VecHexaID elems;

    if (!hasHexahedraAroundVertex())
    {
#ifndef NDEBUG
        serr << "Warning. [HexahedronSetTopologyContainer::getElementAroundElement] hexahedron vertex shell array is empty." << sendl;
#endif
        createHexahedraAroundVertexArray();
    }

    Hexahedron the_hexa = this->getHexahedron(elem);

    for(unsigned int i = 0; i<8; ++i) // for each node of the hexahedron
    {
        HexahedraAroundVertex hexaAV = this->getHexahedraAroundVertex(the_hexa[i]);

        for (unsigned int j = 0; j<hexaAV.size(); ++j) // for each hexahedron around the node
        {
            bool find = false;
            HexaID id = hexaAV[j];

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


const HexahedronSetTopologyContainer::VecHexaID HexahedronSetTopologyContainer::getElementAroundElements(VecHexaID elems)
{
    VecHexaID elemAll;
    VecHexaID elemTmp;

    if (!hasHexahedraAroundVertex())
    {
#ifndef NDEBUG
        serr << "Warning. [HexahedronSetTopologyContainer::getElementAroundElements] hexahedron vertex shell array is empty." << sendl;
#endif
        createHexahedraAroundVertexArray();
    }

    for (unsigned int i = 0; i <elems.size(); ++i) // for each HexaID of input vector
    {
        VecHexaID elemTmp2 = this->getElementAroundElement(elems[i]);

        elemTmp.insert(elemTmp.end(), elemTmp2.begin(), elemTmp2.end());
    }

    for (unsigned int i = 0; i<elemTmp.size(); ++i) // for each hexahedron Id found
    {
        bool find = false;
        HexaID id = elemTmp[i];

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



bool HexahedronSetTopologyContainer::hasHexahedra() const
{
    d_hexahedron.updateIfDirty();
    return !(d_hexahedron.getValue()).empty();
}

bool HexahedronSetTopologyContainer::hasEdgesInHexahedron() const
{
    return !m_edgesInHexahedron.empty();
}

bool HexahedronSetTopologyContainer::hasQuadsInHexahedron() const
{
    return !m_quadsInHexahedron.empty();
}

bool HexahedronSetTopologyContainer::hasHexahedraAroundVertex() const
{
    return !m_hexahedraAroundVertex.empty();
}

bool HexahedronSetTopologyContainer::hasHexahedraAroundEdge() const
{
    return !m_hexahedraAroundEdge.empty();
}

bool HexahedronSetTopologyContainer::hasHexahedraAroundQuad() const
{
    return !m_hexahedraAroundQuad.empty();
}

void HexahedronSetTopologyContainer::clearHexahedra()
{
    helper::WriteAccessor< Data< sofa::helper::vector<Hexahedron> > > m_hexahedron = d_hexahedron;
    m_hexahedron.clear();
}

void HexahedronSetTopologyContainer::clearEdgesInHexahedron()
{
    m_edgesInHexahedron.clear();
}

void HexahedronSetTopologyContainer::clearQuadsInHexahedron()
{
    m_quadsInHexahedron.clear();
}

void HexahedronSetTopologyContainer::clearHexahedraAroundVertex()
{
    m_hexahedraAroundVertex.clear();
}

void HexahedronSetTopologyContainer::clearHexahedraAroundEdge()
{
    m_hexahedraAroundEdge.clear();
}

void HexahedronSetTopologyContainer::clearHexahedraAroundQuad()
{
    m_hexahedraAroundQuad.clear();
}

void HexahedronSetTopologyContainer::clear()
{
    clearHexahedraAroundVertex();
    clearHexahedraAroundEdge();
    clearHexahedraAroundQuad();
    clearQuadsInHexahedron();
    clearEdgesInHexahedron();
    clearHexahedra();

    QuadSetTopologyContainer::clear();
}



void HexahedronSetTopologyContainer::updateTopologyEngineGraph()
{
    // calling real update Data graph function implemented once in PointSetTopologyModifier
    this->updateDataEngineGraph(this->d_hexahedron, this->m_enginesList);

    // will concatenate with edges one:
    QuadSetTopologyContainer::updateTopologyEngineGraph();
}

} // namespace topology

} // namespace component

} // namespace sofa

