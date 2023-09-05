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
#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyContainer.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/core/topology/TopologyHandler.h>

#include <sofa/core/ObjectFactory.h>

namespace sofa::component::topology::container::dynamic
{

using namespace std;
using namespace sofa::defaulttype;
using sofa::core::topology::edgesInHexahedronArray;
using sofa::core::topology::quadsOrientationInHexahedronArray;
using sofa::core::topology::verticesInHexahedronArray;

int HexahedronSetTopologyContainerClass = core::RegisterObject("Hexahedron set topology container")
        .add< HexahedronSetTopologyContainer >()
        ;

HexahedronSetTopologyContainer::HexahedronSetTopologyContainer()
    : QuadSetTopologyContainer()
    , d_createQuadArray(initData(&d_createQuadArray, bool(false),"createQuadArray", "Force the creation of a set of quads associated with the hexahedra"))
    , d_hexahedron(initData(&d_hexahedron, "hexahedra", "List of hexahedron indices"))
{
    addAlias(&d_hexahedron, "hexas");
}


void HexahedronSetTopologyContainer::addHexa(Index a, Index b, Index c, Index d, Index e, Index f, Index g, Index h )
{
    helper::WriteAccessor< Data< sofa::type::vector<Hexahedron> > > m_hexahedron = d_hexahedron;
    m_hexahedron.push_back(Hexahedron(a,b,c,d,e,f,g,h));
    if (a >= getNbPoints()) setNbPoints(a+1);
    if (b >= getNbPoints()) setNbPoints(b+1);
    if (c >= getNbPoints()) setNbPoints(c+1);
    if (d >= getNbPoints()) setNbPoints(d+1);
    if (e >= getNbPoints()) setNbPoints(e+1);
    if (f >= getNbPoints()) setNbPoints(f+1);
    if (g >= getNbPoints()) setNbPoints(g+1);
    if (h >= getNbPoints()) setNbPoints(h+1);
}

void HexahedronSetTopologyContainer::init()
{
    const helper::ReadAccessor< Data< sofa::type::vector<Hexahedron> > > m_hexahedron = d_hexahedron;

    if (d_initPoints.isSet())
    {
        setNbPoints(Size(d_initPoints.getValue().size()));
    }    
    if (!m_hexahedron.empty())
    {
        // Todo (epernod 2019-03-12): optimise by removing this loop or at least create tetrahedronAV at the same time.
        for (size_t i=0; i<m_hexahedron.size(); ++i)
        {
            for(PointID j=0; j<8; ++j)
            {
                const HexahedronID a = m_hexahedron[i][j];
                if (a >= getNbPoints()) setNbPoints(a+1);
            }
        }
    }

    if (!m_hexahedron.empty())
        initTopology();
}

void HexahedronSetTopologyContainer::initTopology()
{
    QuadSetTopologyContainer::initTopology();

    // Create tetrahedron cross element buffers.
    createQuadsInHexahedronArray();
    createEdgesInHexahedronArray();

    createHexahedraAroundQuadArray();
    createHexahedraAroundEdgeArray();
    createHexahedraAroundVertexArray();
}

void HexahedronSetTopologyContainer::createHexahedronSetArray()
{
    msg_error() << "createHexahedronSetArray method must be implemented by a child topology.";
}

void HexahedronSetTopologyContainer::createEdgeSetArray()
{
    if(!hasHexahedra()) // this method should only be called when hexa exist
        createHexahedronSetArray();

    if(hasEdges())
    {
        EdgeSetTopologyContainer::clear();

        clearEdgesInQuad();
        clearQuadsAroundEdge();

        clearEdgesInHexahedron();
        clearHexahedraAroundEdge();
    }

    // create a temporary map to find redundant edges
    std::map<Edge,EdgeID> edgeMap;
    helper::WriteAccessor< Data< sofa::type::vector<Edge> > > m_edge = d_edge;
    const helper::ReadAccessor< Data< sofa::type::vector<Hexahedron> > > m_hexahedron = d_hexahedron;

    /// create the m_edge array at the same time than it fills the m_edgesInHexahedron array
    for(size_t i=0; i<m_hexahedron.size(); ++i)
    {
        const Hexahedron &t = m_hexahedron[i];
        for(PointID j=0; j<12; ++j)
        {
            PointID v1 = t[edgesInHexahedronArray[j][0]];
            PointID v2 = t[edgesInHexahedronArray[j][1]];
            const Edge e((v1<v2) ? Edge(v1,v2) : Edge(v2,v1));

            if(edgeMap.find(e)==edgeMap.end())
            {
                // edge not in edgeMap so create a new one
                const size_t edgeIndex = edgeMap.size();
                edgeMap[e] = (EdgeID)edgeIndex;
                m_edge.push_back(e);
            }
        }
    }
}

void HexahedronSetTopologyContainer::createEdgesInHexahedronArray()
{
    // first clear potential previous buffer
    clearEdgesInHexahedron();

    if(!hasHexahedra()) // this method should only be called when Hexa exist
        createHexahedronSetArray();

    if (hasEdgesInHexahedron()) // created by upper topology
        return;

    m_edgesInHexahedron.resize( getNumberOfHexahedra());
    const helper::ReadAccessor< Data< sofa::type::vector<Hexahedron> > > m_hexahedron = d_hexahedron;

    for(size_t i=0; i<m_hexahedron.size(); ++i)
    {
        const Hexahedron &t = m_hexahedron[i];

        // adding edge i in the edge shell of both points
        for(PointID j=0; j<12; ++j)
        {
            const EdgeID edgeIndex = getEdgeIndex(t[edgesInHexahedronArray[j][0]],
                                                  t[edgesInHexahedronArray[j][1]]);
            m_edgesInHexahedron[i][j] = edgeIndex;
        }
    }
}

void HexahedronSetTopologyContainer::createQuadSetArray()
{
    if(!hasHexahedra()) // this method should only be called when hexa exist
        createHexahedronSetArray();

    if(hasQuads())
    {
        QuadSetTopologyContainer::clear();

        clearQuadsInHexahedron();
        clearHexahedraAroundQuad();
    }

    // create a temporary map to find redundant quads
    std::map<Quad, QuadID> quadMap;
    helper::WriteAccessor< Data< sofa::type::vector<Quad> > > m_quad = d_quad;
    const helper::ReadAccessor< Data< sofa::type::vector<Hexahedron> > > m_hexahedron = d_hexahedron;

    for(size_t i=0; i<m_hexahedron.size(); ++i)
    {
        const Hexahedron &h = m_hexahedron[i];

        PointID v[4], val;

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
        QuadID quadIndex;
        Quad qu(v[0],v[3],v[2],v[1]);
        std::map<Quad,QuadID>::iterator itt = quadMap.find(qu);
        if(itt==quadMap.end())
        {
            // quad not in edgeMap so create a new one
            quadIndex=(QuadID)m_quad.size();
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
            quadIndex=(QuadID)m_quad.size();
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
            quadIndex=(QuadID)m_quad.size();
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
            quadIndex=(QuadID)m_quad.size();
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
            quadIndex=(QuadID)m_quad.size();
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
            quadIndex=(QuadID)m_quad.size();
            quadMap[qu]=quadIndex;
            qu=Quad(v[0],v[1],v[2],v[3]);
            quadMap[qu]=quadIndex;
            m_quad.push_back(qu);
        }
    }
}

void HexahedronSetTopologyContainer::createQuadsInHexahedronArray()
{
    // first clear potential previous buffer
    clearQuadsInHexahedron();

    if(!hasQuads())
        createQuadSetArray();

    if(hasQuadsInHexahedron())// created by upper topology
        return;

    m_quadsInHexahedron.resize( getNumberOfHexahedra());
    const helper::ReadAccessor< Data< sofa::type::vector<Hexahedron> > > m_hexahedron = d_hexahedron;

    for(size_t i = 0; i < getNumberOfHexahedra(); ++i)
    {
        const Hexahedron &h=m_hexahedron[i];
        QuadID quadIndex;

        // adding the 6 quads in the quad list of the ith hexahedron  i
        // Quad 0 :
        quadIndex=getQuadIndex(h[0],h[3],h[2],h[1]);
        assert(quadIndex!= InvalidID);
        m_quadsInHexahedron[i][0]=quadIndex;
        // Quad 1 :
        quadIndex=getQuadIndex(h[4],h[5],h[6],h[7]);
        assert(quadIndex!= InvalidID);
        m_quadsInHexahedron[i][1]=quadIndex;
        // Quad 2 :
        quadIndex=getQuadIndex(h[0],h[1],h[5],h[4]);
        assert(quadIndex!= InvalidID);
        m_quadsInHexahedron[i][2]=quadIndex;
        // Quad 3 :
        quadIndex=getQuadIndex(h[1],h[2],h[6],h[5]);
        assert(quadIndex!= InvalidID);
        m_quadsInHexahedron[i][3]=quadIndex;
        // Quad 4 :
        quadIndex=getQuadIndex(h[2],h[3],h[7],h[6]);
        assert(quadIndex!= InvalidID);
        m_quadsInHexahedron[i][4]=quadIndex;
        // Quad 5 :
        quadIndex=getQuadIndex(h[3],h[0],h[4],h[7]);
        assert(quadIndex!= InvalidID);
        m_quadsInHexahedron[i][5]=quadIndex;
    }
}

void HexahedronSetTopologyContainer::createHexahedraAroundVertexArray()
{
    // first clear potential previous buffer
    clearHexahedraAroundVertex();

    if (getNbPoints() == 0) // in case only Data have been copied and not going thourgh AddTriangle methods.
        this->setNbPoints(Size(d_initPoints.getValue().size()));

    m_hexahedraAroundVertex.resize( getNbPoints() );
    const helper::ReadAccessor< Data< sofa::type::vector<Hexahedron> > > m_hexahedron = d_hexahedron;

    for(size_t i=0; i<m_hexahedron.size(); ++i)
    {
        HexahedronID hexaID = (HexahedronID)i;
        // adding vertex i in the vertex shell
        for(unsigned int j=0; j<8; ++j)
            m_hexahedraAroundVertex[ m_hexahedron[i][j]  ].push_back(hexaID);
    }
}

void HexahedronSetTopologyContainer::createHexahedraAroundEdgeArray ()
{
    clearHexahedraAroundEdge();

    if(!hasEdgesInHexahedron())
        createEdgesInHexahedronArray();

    m_hexahedraAroundEdge.resize(getNumberOfEdges());
    for(size_t i=0; i<getNumberOfHexahedra(); ++i)
    {
        HexahedronID hexaID = (HexahedronID)i;
        // adding edge i in the edge shell
        for(unsigned int j=0; j<12; ++j)
        {
            m_hexahedraAroundEdge[ m_edgesInHexahedron[i][j] ].push_back(hexaID);
        }
    }
}

void HexahedronSetTopologyContainer::createHexahedraAroundQuadArray()
{
    clearHexahedraAroundQuad();

    if(!hasQuadsInHexahedron())
        createQuadsInHexahedronArray();

    m_hexahedraAroundQuad.resize( getNumberOfQuads());
    for(size_t i=0; i<getNumberOfHexahedra(); ++i)
    {
        HexahedronID hexaID = (HexahedronID)i;
        // adding quad i in the edge shell of both points
        for(unsigned int j=0; j<6; ++j)
        {
            m_hexahedraAroundQuad[ m_quadsInHexahedron[i][j] ].push_back(hexaID);
        }
    }
}

const sofa::type::vector<HexahedronSetTopologyContainer::Hexahedron> &HexahedronSetTopologyContainer::getHexahedronArray()
{
    return d_hexahedron.getValue();
}

core::topology::Topology::HexahedronID HexahedronSetTopologyContainer::getHexahedronIndex(PointID v1, PointID v2, PointID v3, PointID v4,
        PointID v5, PointID v6, PointID v7, PointID v8)
{
    if(!hasHexahedraAroundVertex())
    {
        return InvalidID;
    }

    sofa::type::vector<HexahedronID> set1 = getHexahedraAroundVertex(v1);
    sofa::type::vector<HexahedronID> set2 = getHexahedraAroundVertex(v2);
    sofa::type::vector<HexahedronID> set3 = getHexahedraAroundVertex(v3);
    sofa::type::vector<HexahedronID> set4 = getHexahedraAroundVertex(v4);
    sofa::type::vector<HexahedronID> set5 = getHexahedraAroundVertex(v5);
    sofa::type::vector<HexahedronID> set6 = getHexahedraAroundVertex(v6);
    sofa::type::vector<HexahedronID> set7 = getHexahedraAroundVertex(v7);
    sofa::type::vector<HexahedronID> set8 = getHexahedraAroundVertex(v8);

    sort(set1.begin(), set1.end());
    sort(set2.begin(), set2.end());
    sort(set3.begin(), set3.end());
    sort(set4.begin(), set4.end());
    sort(set5.begin(), set5.end());
    sort(set6.begin(), set6.end());
    sort(set7.begin(), set7.end());
    sort(set8.begin(), set8.end());

    // The destination vector must be large enough to contain the result.
    sofa::type::vector<HexahedronID> out1(set1.size()+set2.size());
    sofa::type::vector<HexahedronID>::iterator result1;
    result1 = std::set_intersection(set1.begin(),set1.end(),set2.begin(),set2.end(),out1.begin());
    out1.erase(result1,out1.end());

    sofa::type::vector<HexahedronID> out2(set3.size()+out1.size());
    sofa::type::vector<HexahedronID>::iterator result2;
    result2 = std::set_intersection(set3.begin(),set3.end(),out1.begin(),out1.end(),out2.begin());
    out2.erase(result2,out2.end());

    sofa::type::vector<HexahedronID> out3(set4.size()+out2.size());
    sofa::type::vector<HexahedronID>::iterator result3;
    result3 = std::set_intersection(set4.begin(),set4.end(),out2.begin(),out2.end(),out3.begin());
    out3.erase(result3,out3.end());

    sofa::type::vector<HexahedronID> out4(set5.size()+out3.size());
    sofa::type::vector<HexahedronID>::iterator result4;
    result4 = std::set_intersection(set5.begin(),set5.end(),out3.begin(),out3.end(),out4.begin());
    out4.erase(result4,out4.end());

    sofa::type::vector<HexahedronID> out5(set6.size()+out4.size());
    sofa::type::vector<HexahedronID>::iterator result5;
    result5 = std::set_intersection(set6.begin(),set6.end(),out4.begin(),out4.end(),out5.begin());
    out5.erase(result5,out5.end());

    sofa::type::vector<HexahedronID> out6(set7.size()+out5.size());
    sofa::type::vector<HexahedronID>::iterator result6;
    result6 = std::set_intersection(set7.begin(),set7.end(),out5.begin(),out5.end(),out6.begin());
    out6.erase(result6,out6.end());

    sofa::type::vector<HexahedronID> out7(set8.size()+out6.size());
    sofa::type::vector<HexahedronID>::iterator result7;
    result7 = std::set_intersection(set8.begin(),set8.end(),out6.begin(),out6.end(),out7.begin());
    out7.erase(result7,out7.end());

    msg_warning_when(out7.size() > 1) << "More than one Hexahedron found for indices: [" << v1 << "; " << v2 << "; " << v3 << "; " << v4 << "; "
                         << v5 << "; " << v6 << "; " << v7 << "; " << v8 << "]";

    assert(out7.size()==0 || out7.size()==1);
    if(out7.size()==1)
        return (int) (out7[0]);

    return InvalidID;
}

const HexahedronSetTopologyContainer::Hexahedron HexahedronSetTopologyContainer::getHexahedron(HexaID i)
{
    if (i >= getNbHexahedra())
        return Hexahedron(InvalidID, InvalidID, InvalidID, InvalidID, InvalidID, InvalidID, InvalidID, InvalidID);
    else
        return (d_hexahedron.getValue())[i];
}

Size HexahedronSetTopologyContainer::getNumberOfHexahedra() const
{
    return Size(d_hexahedron.getValue().size());
}


Size HexahedronSetTopologyContainer::getNumberOfElements() const
{
    return this->getNumberOfHexahedra();
}


const sofa::type::vector< HexahedronSetTopologyContainer::HexahedraAroundVertex > &HexahedronSetTopologyContainer::getHexahedraAroundVertexArray()
{
    return m_hexahedraAroundVertex;
}

const sofa::type::vector< HexahedronSetTopologyContainer::HexahedraAroundEdge > &HexahedronSetTopologyContainer::getHexahedraAroundEdgeArray()
{
    return m_hexahedraAroundEdge;
}

const sofa::type::vector< HexahedronSetTopologyContainer::HexahedraAroundQuad > &HexahedronSetTopologyContainer::getHexahedraAroundQuadArray()
{
    return m_hexahedraAroundQuad;
}

const sofa::type::vector< HexahedronSetTopologyContainer::EdgesInHexahedron> &HexahedronSetTopologyContainer::getEdgesInHexahedronArray()
{
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
    return Quad (quadsOrientationInHexahedronArray[i][0],
        quadsOrientationInHexahedronArray[i][1],
        quadsOrientationInHexahedronArray[i][2],
        quadsOrientationInHexahedronArray[i][3]);
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

    const EdgeID the_edgeID = this->getEdgesInHexahedron(_hexaID)[_edgeID];
    const QuadsAroundEdge QaroundE = this->getQuadsAroundEdge(the_edgeID);

    const QuadsInHexahedron QinH = this->getQuadsInHexahedron(_hexaID);
    const QuadID the_quadID = QinH[_quadID];
    const QuadID nextQuad = 0;

    if (QaroundE.size() < 2)
    {
    	msg_error() << "GetNextAdjacentQuad: no quad around edge: " << the_edgeID;
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
        for (size_t i=0; i<QaroundE.size(); ++i)
        {
            const int res = this->getQuadIndexInHexahedron(QinH, QaroundE[i]);
            if (res != -1 && QaroundE[i] != the_quadID)
                return (unsigned int)res;
        }
    }

    return nextQuad;
}


const sofa::type::vector< QuadSetTopologyContainer::QuadsInHexahedron> &HexahedronSetTopologyContainer::getQuadsInHexahedronArray()
{
    return m_quadsInHexahedron;
}

const HexahedronSetTopologyContainer::HexahedraAroundVertex &HexahedronSetTopologyContainer::getHexahedraAroundVertex(PointID id)
{
    if (id < m_hexahedraAroundVertex.size())
        return m_hexahedraAroundVertex[id];

    return InvalidSet;
}

const HexahedronSetTopologyContainer::HexahedraAroundEdge &HexahedronSetTopologyContainer::getHexahedraAroundEdge(EdgeID id)
{
    if (id < m_hexahedraAroundEdge.size())
        return m_hexahedraAroundEdge[id];

    return InvalidSet;
}

const HexahedronSetTopologyContainer::HexahedraAroundQuad &HexahedronSetTopologyContainer::getHexahedraAroundQuad(QuadID id)
{
    if (id < m_hexahedraAroundQuad.size())
        return m_hexahedraAroundQuad[id];

    return InvalidSet;
}

const QuadSetTopologyContainer::EdgesInHexahedron &HexahedronSetTopologyContainer::getEdgesInHexahedron(HexaID id)
{
    if (id < m_edgesInHexahedron.size())
        return m_edgesInHexahedron[id];

    return InvalidEdgesInHexahedron;
}

const QuadSetTopologyContainer::QuadsInHexahedron &HexahedronSetTopologyContainer::getQuadsInHexahedron(HexaID id)
{
    if (id < m_quadsInHexahedron.size())
        return m_quadsInHexahedron[id];

    return InvalidQuadsInHexahedron;
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
        const EdgeID edgeIndex) const
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
    {
        dmsg_warning() << "getHexahedraAroundEdgeForModification: HexahedraAroundEdgeArray is empty. Be sure to call createHexahedraAroundEdgeArray first.";
        createHexahedraAroundEdgeArray();
    }

    assert( i < m_hexahedraAroundEdge.size());

    //TODO epernod (2020-04): this method should be removed as it can create a seg fault.
    return m_hexahedraAroundEdge[i];
}

HexahedronSetTopologyContainer::HexahedraAroundVertex &HexahedronSetTopologyContainer::getHexahedraAroundVertexForModification(const PointID i)
{
    if(!hasHexahedraAroundVertex())
    {
        dmsg_warning() << "getHexahedraAroundVertexForModification: HexahedraAroundVertexArray is empty. Be sure to call createHexahedraAroundVertexArray first.";
        createHexahedraAroundVertexArray();
    }

    assert( i < m_hexahedraAroundVertex.size());

    //TODO epernod (2020-04): this method should be removed as it can create a seg fault.
    return m_hexahedraAroundVertex[i];
}

HexahedronSetTopologyContainer::HexahedraAroundQuad &HexahedronSetTopologyContainer::getHexahedraAroundQuadForModification(const QuadID i)
{
    if(!hasHexahedraAroundQuad())
    {
        dmsg_warning() << "getHexahedraAroundQuadForModification: HexahedraAroundQuadArray is empty. Be sure to call createHexahedraAroundQuadArray first.";
        createHexahedraAroundQuadArray();
    }

    assert( i < m_hexahedraAroundQuad.size());

    //TODO epernod (2020-04): this method should be removed as it can create a seg fault.
    return m_hexahedraAroundQuad[i];
}


bool HexahedronSetTopologyContainer::checkTopology() const
{
    if (!d_checkTopology.getValue()) {
        return true;
    }

	bool ret = true;
    const helper::ReadAccessor< Data< sofa::type::vector<Hexahedron> > > m_hexahedron = d_hexahedron;
	if (hasHexahedraAroundVertex())
	{
		for (size_t i = 0; i < m_hexahedraAroundVertex.size(); ++i)
		{
			const sofa::type::vector<HexahedronID> &tvs = m_hexahedraAroundVertex[i];
			for (size_t j = 0; j < tvs.size(); ++j)
			{
                const bool check_hexa_vertex_shell = (m_hexahedron[tvs[j]][0] == i)
					|| (m_hexahedron[tvs[j]][1] == i)
					|| (m_hexahedron[tvs[j]][2] == i)
					|| (m_hexahedron[tvs[j]][3] == i)
					|| (m_hexahedron[tvs[j]][4] == i)
					|| (m_hexahedron[tvs[j]][5] == i)
					|| (m_hexahedron[tvs[j]][6] == i)
					|| (m_hexahedron[tvs[j]][7] == i);

				if (!check_hexa_vertex_shell)
				{
					msg_error() << "*** CHECK FAILED : check_hexa_vertex_shell, i = " << i << " , j = " << j;
					ret = false;
				}
			}
		}
	}

	if (hasHexahedraAroundEdge())
	{
		for (size_t i = 0; i < m_hexahedraAroundEdge.size(); ++i)
		{
			const sofa::type::vector<HexahedronID> &tes = m_hexahedraAroundEdge[i];
			for (size_t j = 0; j < tes.size(); ++j)
			{
                const bool check_hexa_edge_shell = (m_edgesInHexahedron[tes[j]][0] == i)
					|| (m_edgesInHexahedron[tes[j]][1] == i)
					|| (m_edgesInHexahedron[tes[j]][2] == i)
					|| (m_edgesInHexahedron[tes[j]][3] == i)
					|| (m_edgesInHexahedron[tes[j]][4] == i)
					|| (m_edgesInHexahedron[tes[j]][5] == i)
					|| (m_edgesInHexahedron[tes[j]][6] == i)
					|| (m_edgesInHexahedron[tes[j]][7] == i)
					|| (m_edgesInHexahedron[tes[j]][8] == i)
					|| (m_edgesInHexahedron[tes[j]][9] == i)
					|| (m_edgesInHexahedron[tes[j]][10] == i)
					|| (m_edgesInHexahedron[tes[j]][11] == i);
				if (!check_hexa_edge_shell)
				{
					msg_error() << "*** CHECK FAILED : check_hexa_edge_shell, i = " << i << " , j = " << j;
					ret = false;
				}
			}
		}
	}

	if (hasHexahedraAroundQuad())
	{
		for (size_t i = 0; i < m_hexahedraAroundQuad.size(); ++i)
		{
			const sofa::type::vector<HexahedronID> &tes = m_hexahedraAroundQuad[i];
			for (size_t j = 0; j < tes.size(); ++j)
			{
                const bool check_hexa_quad_shell = (m_quadsInHexahedron[tes[j]][0] == i)
					|| (m_quadsInHexahedron[tes[j]][1] == i)
					|| (m_quadsInHexahedron[tes[j]][2] == i)
					|| (m_quadsInHexahedron[tes[j]][3] == i)
					|| (m_quadsInHexahedron[tes[j]][4] == i)
					|| (m_quadsInHexahedron[tes[j]][5] == i);
				if (!check_hexa_quad_shell)
				{
					msg_error() << "*** CHECK FAILED : check_hexa_quad_shell, i = " << i << " , j = " << j;
					ret = false;
				}
			}
		}
	}

	return ret && QuadSetTopologyContainer::checkTopology();
}




/// Get information about connexity of the mesh
/// @{

bool HexahedronSetTopologyContainer::checkConnexity()
{
    const size_t nbr = this->getNbHexahedra();

    if (nbr == 0)
    {
        msg_error() << "CheckConnexity: Can't compute connexity as there are no Hexahedra";
        return false;
    }

    const VecHexaID elemAll = this->getConnectedElement(0);

    if (elemAll.size() != nbr)
    {
        msg_warning() << "CheckConnexity: Hexahedra are missings. There is more than one connexe component.";
        return false;
    }

    return true;
}


Size HexahedronSetTopologyContainer::getNumberOfConnectedComponent()
{
    const auto nbr = this->getNbHexahedra();

    if (nbr == 0)
    {
        msg_error() << "Can't getNumberOfConnectedComponent as there are no Hexahedra";
        return 0;
    }

    VecHexaID elemAll = this->getConnectedElement(0);
    Size cpt = 1;

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
    VecHexaID elemAll;
    if(!hasHexahedraAroundVertex())	// this method should only be called when the shell array exists
    {
        dmsg_warning() << "getConnectedElement: HexahedraAroundVertexArray is empty. Be sure to call createHexahedraAroundVertexArray first.";
        return elemAll;
    }

    VecHexaID elemOnFront, elemPreviousFront, elemNextFront;
    bool end = false;
    size_t cpt = 0;
    const size_t nbr = this->getNbHexahedra();

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
        for (size_t i = 0; i<elemNextFront.size(); ++i)
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
        cpt += elemPreviousFront.size();

        if (elemPreviousFront.empty())
        {
            end = true;
			msg_error() << "Loop for computing connexity has reach end.";
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
        dmsg_warning() << "getElementAroundElement: HexahedraAroundVertexArray is empty. Be sure to call createHexahedraAroundVertexArray first.";
        return elems;
    }

    Hexahedron the_hexa = this->getHexahedron(elem);

    for(unsigned int i = 0; i<8; ++i) // for each node of the hexahedron
    {
        HexahedraAroundVertex hexaAV = this->getHexahedraAroundVertex(the_hexa[i]);

        for (size_t j = 0; j<hexaAV.size(); ++j) // for each hexahedron around the node
        {
            bool find = false;
            HexaID id = hexaAV[j];

            if (id == elem)
                continue;

            for (size_t k = 0; k<elems.size(); ++k) // check no redundancy
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
    if (!hasHexahedraAroundVertex())
    {
        dmsg_warning() << "getElementAroundElements: HexahedraAroundVertexArray is empty. Be sure to call createHexahedraAroundVertexArray first.";
        return elemAll;
    }

    VecHexaID elemTmp;
    for (size_t i = 0; i <elems.size(); ++i) // for each HexaID of input vector
    {
        VecHexaID elemTmp2 = this->getElementAroundElement(elems[i]);

        elemTmp.insert(elemTmp.end(), elemTmp2.begin(), elemTmp2.end());
    }

    for (size_t i = 0; i<elemTmp.size(); ++i) // for each hexahedron Id found
    {
        bool find = false;
        HexaID id = elemTmp[i];

        for (size_t j = 0; j<elems.size(); ++j) // check no redundancy with input vector
            if (id == elems[j])
            {
                find = true;
                break;
            }

        if (!find)
        {
            for (size_t j = 0; j<elemAll.size(); ++j) // check no redundancy in output vector
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
    helper::WriteAccessor< Data< sofa::type::vector<Hexahedron> > > m_hexahedron = d_hexahedron;
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

void HexahedronSetTopologyContainer::setHexahedronTopologyToDirty()
{
    // set this container to dirty
    m_hexahedronTopologyDirty = true;

    // set all engines link to this container to dirty
    auto& hexaTopologyHandlerList = getTopologyHandlerList(sofa::geometry::ElementType::HEXAHEDRON);
    for (const auto topoHandler : hexaTopologyHandlerList)
    {
        topoHandler->setDirtyValue();
        msg_info() << "Hexahedron Topology Set dirty engine: " << topoHandler->getName();
    }
}

void HexahedronSetTopologyContainer::cleanHexahedronTopologyFromDirty()
{
    m_hexahedronTopologyDirty = false;

    // security, clean all engines to avoid loops
    auto& hexaTopologyHandlerList = getTopologyHandlerList(sofa::geometry::ElementType::HEXAHEDRON);
    for (const auto topoHandler : hexaTopologyHandlerList)
    {
        if (topoHandler->isDirty())
        {
            msg_warning() << "Hexahedron Topology update did not clean engine: " << topoHandler->getName();
            topoHandler->cleanDirty();
        }
    }
}

bool HexahedronSetTopologyContainer::linkTopologyHandlerToData(core::topology::TopologyHandler* topologyHandler, sofa::geometry::ElementType elementType)
{
    if (elementType == sofa::geometry::ElementType::HEXAHEDRON)
    {
        d_hexahedron.addOutput(topologyHandler);
        return true;
    }
    else
    {
        return QuadSetTopologyContainer::linkTopologyHandlerToData(topologyHandler, elementType);
    }
}

bool HexahedronSetTopologyContainer::unlinkTopologyHandlerToData(core::topology::TopologyHandler* topologyHandler, sofa::geometry::ElementType elementType)
{
    if (elementType == sofa::geometry::ElementType::HEXAHEDRON)
    {
        d_hexahedron.delOutput(topologyHandler);
        return true;
    }
    else
    {
        return QuadSetTopologyContainer::unlinkTopologyHandlerToData(topologyHandler, elementType);
    }
}

} //namespace sofa::component::topology::container::dynamic
