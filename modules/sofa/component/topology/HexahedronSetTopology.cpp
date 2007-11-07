#include <sofa/component/topology/HexahedronSetTopology.h>
#include <sofa/component/topology/HexahedronSetTopology.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(HexahedronSetTopology)


template class HexahedronSetTopology<Vec3dTypes>;
template class HexahedronSetTopology<Vec3fTypes>;
template class HexahedronSetTopology<Vec2dTypes>;
template class HexahedronSetTopology<Vec2fTypes>;
template class HexahedronSetTopology<Vec1dTypes>;
template class HexahedronSetTopology<Vec1fTypes>;

template class HexahedronSetTopologyAlgorithms<Vec3fTypes>;
template class HexahedronSetTopologyAlgorithms<Vec3dTypes>;
template class HexahedronSetTopologyAlgorithms<Vec2fTypes>;
template class HexahedronSetTopologyAlgorithms<Vec2dTypes>;
template class HexahedronSetTopologyAlgorithms<Vec1fTypes>;
template class HexahedronSetTopologyAlgorithms<Vec1dTypes>;

template class HexahedronSetGeometryAlgorithms<Vec3fTypes>;
template class HexahedronSetGeometryAlgorithms<Vec3dTypes>;
template class HexahedronSetGeometryAlgorithms<Vec2fTypes>;
template class HexahedronSetGeometryAlgorithms<Vec2dTypes>;
template class HexahedronSetGeometryAlgorithms<Vec1fTypes>;
template class HexahedronSetGeometryAlgorithms<Vec1dTypes>;


// implementation HexahedronSetTopologyContainer


void HexahedronSetTopologyContainer::createHexahedronEdgeArray ()
{
    m_hexahedronEdge.resize( getNumberOfHexahedra());
    unsigned int j;
    int edgeIndex;
    unsigned int edgeHexahedronDescriptionArray[12][2]= {{0,1},{0,3},{0,4},{1,2},{1,5},{2,3},{2,6},{3,7},{4,5},{4,7},{5,6},{6,7}};

    if (m_edge.size()>0)
    {
        for (unsigned int i = 0; i < m_hexahedron.size(); ++i)
        {
            Hexahedron &t=m_hexahedron[i];
            // adding edge i in the edge shell of both points
            for (j=0; j<12; ++j)
            {
                edgeIndex=getEdgeIndex(t[edgeHexahedronDescriptionArray[j][0]],
                        t[edgeHexahedronDescriptionArray[j][1]]);
                assert(edgeIndex!= -1);
                m_hexahedronEdge[i][j]=edgeIndex;
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
        /// create the m_edge array at the same time than it fills the m_hexahedronEdge array
        for (unsigned int i = 0; i < m_hexahedron.size(); ++i)
        {
            Hexahedron &t=m_hexahedron[i];
            for (j=0; j<12; ++j)
            {
                v1=t[edgeHexahedronDescriptionArray[j][0]];
                v2=t[edgeHexahedronDescriptionArray[j][1]];
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
                m_hexahedronEdge[i][j]=edgeIndex;
            }
        }
    }
}


void HexahedronSetTopologyContainer::createHexahedronQuadArray ()
{
    m_hexahedronQuad.resize( getNumberOfHexahedra());
    int quadIndex;

    if (m_quad.size()>0)
    {
        for (unsigned int i = 0; i < m_hexahedron.size(); ++i)
        {
            Hexahedron &h=m_hexahedron[i];
            // adding the 6 quads in the quad list of the ith hexahedron  i
            // Quad 0 :
            quadIndex=getQuadIndex(h[0],h[1],h[2],h[3]);
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
    else
    {
        // create a temporary map to find redundant quads
        std::map<Quad,unsigned int> quadMap;
        std::map<Quad,unsigned int>::iterator itt;
        Quad qu;
        unsigned int v[4];
        /// create the m_edge array at the same time than it fills the m_hexahedronEdge array
        for (unsigned int i = 0; i < m_hexahedron.size(); ++i)
        {
            Hexahedron &h=m_hexahedron[i];

            // Quad 0 :
            v[0]=h[0]; v[1]=h[1]; v[2]=h[2]; v[3]=h[3];
            std::sort(v,v+1); std::sort(v,v+2); std::sort(v,v+3);
            std::sort(v+1,v+2); std::sort(v+1,v+3);
            std::sort(v+2,v+3);
            // sort vertices in lexicographics order
            qu=helper::make_array<unsigned int>(v[0],v[1],v[2],v[3]);
            itt=quadMap.find(qu);
            if (itt==quadMap.end())
            {
                // quad not in edgeMap so create a new one
                quadIndex=quadMap.size();
                quadMap[qu]=quadIndex;
                m_quad.push_back(qu);
            }
            else
            {
                quadIndex=(*itt).second;
            }
            m_hexahedronQuad[i][0]=quadIndex;

            // Quad 1 :
            v[0]=h[4]; v[1]=h[5]; v[2]=h[6]; v[3]=h[7];
            std::sort(v,v+1); std::sort(v,v+2); std::sort(v,v+3);
            std::sort(v+1,v+2); std::sort(v+1,v+3);
            std::sort(v+2,v+3);
            // sort vertices in lexicographics order
            qu=helper::make_array<unsigned int>(v[0],v[1],v[2],v[3]);
            itt=quadMap.find(qu);
            if (itt==quadMap.end())
            {
                // quad not in edgeMap so create a new one
                quadIndex=quadMap.size();
                quadMap[qu]=quadIndex;
                m_quad.push_back(qu);
            }
            else
            {
                quadIndex=(*itt).second;
            }
            m_hexahedronQuad[i][1]=quadIndex;

            // Quad 2 :
            v[0]=h[0]; v[1]=h[1]; v[2]=h[5]; v[3]=h[4];
            std::sort(v,v+1); std::sort(v,v+2); std::sort(v,v+3);
            std::sort(v+1,v+2); std::sort(v+1,v+3);
            std::sort(v+2,v+3);
            // sort vertices in lexicographics order
            qu=helper::make_array<unsigned int>(v[0],v[1],v[2],v[3]);
            itt=quadMap.find(qu);
            if (itt==quadMap.end())
            {
                // quad not in edgeMap so create a new one
                quadIndex=quadMap.size();
                quadMap[qu]=quadIndex;
                m_quad.push_back(qu);
            }
            else
            {
                quadIndex=(*itt).second;
            }
            m_hexahedronQuad[i][2]=quadIndex;

            // Quad 3 :
            v[0]=h[1]; v[1]=h[2]; v[2]=h[6]; v[3]=h[5];
            std::sort(v,v+1); std::sort(v,v+2); std::sort(v,v+3);
            std::sort(v+1,v+2); std::sort(v+1,v+3);
            std::sort(v+2,v+3);
            // sort vertices in lexicographics order
            qu=helper::make_array<unsigned int>(v[0],v[1],v[2],v[3]);
            itt=quadMap.find(qu);
            if (itt==quadMap.end())
            {
                // quad not in edgeMap so create a new one
                quadIndex=quadMap.size();
                quadMap[qu]=quadIndex;
                m_quad.push_back(qu);
            }
            else
            {
                quadIndex=(*itt).second;
            }
            m_hexahedronQuad[i][3]=quadIndex;

            // Quad 4 :
            v[0]=h[2]; v[1]=h[3]; v[2]=h[7]; v[3]=h[6];
            std::sort(v,v+1); std::sort(v,v+2); std::sort(v,v+3);
            std::sort(v+1,v+2); std::sort(v+1,v+3);
            std::sort(v+2,v+3);
            // sort vertices in lexicographics order
            qu=helper::make_array<unsigned int>(v[0],v[1],v[2],v[3]);
            itt=quadMap.find(qu);
            if (itt==quadMap.end())
            {
                // quad not in edgeMap so create a new one
                quadIndex=quadMap.size();
                quadMap[qu]=quadIndex;
                m_quad.push_back(qu);
            }
            else
            {
                quadIndex=(*itt).second;
            }
            m_hexahedronQuad[i][4]=quadIndex;

            // Quad 5 :
            v[0]=h[3]; v[1]=h[0]; v[2]=h[4]; v[3]=h[7];
            std::sort(v,v+1); std::sort(v,v+2); std::sort(v,v+3);
            std::sort(v+1,v+2); std::sort(v+1,v+3);
            std::sort(v+2,v+3);
            // sort vertices in lexicographics order
            qu=helper::make_array<unsigned int>(v[0],v[1],v[2],v[3]);
            itt=quadMap.find(qu);
            if (itt==quadMap.end())
            {
                // quad not in edgeMap so create a new one
                quadIndex=quadMap.size();
                quadMap[qu]=quadIndex;
                m_quad.push_back(qu);
            }
            else
            {
                quadIndex=(*itt).second;
            }
            m_hexahedronQuad[i][5]=quadIndex;
        }
    }
}

void HexahedronSetTopologyContainer::createHexahedronVertexShellArray ()
{
    m_hexahedronVertexShell.resize( m_basicTopology->getDOFNumber() );
    unsigned int j;

    for (unsigned int i = 0; i < m_hexahedron.size(); ++i)
    {
        // adding vertex i in the vertex shell
        for (j=0; j<8; ++j)
            m_hexahedronVertexShell[ m_hexahedron[i][j]  ].push_back( i );
    }
}

void HexahedronSetTopologyContainer::createHexahedronEdgeShellArray ()
{
    m_hexahedronEdgeShell.resize( getNumberOfEdges());
    unsigned int j;
    const sofa::helper::vector< HexahedronEdges > &tea=getHexahedronEdgeArray();


    for (unsigned int i = 0; i < m_hexahedron.size(); ++i)
    {
        // adding edge i in the edge shell
        for (j=0; j<12; ++j)
        {
            m_hexahedronEdgeShell[ tea[i][j] ].push_back( i );
        }
    }
}

void HexahedronSetTopologyContainer::createHexahedronQuadShellArray ()
{
    m_hexahedronQuadShell.resize( getNumberOfQuads());
    unsigned int j;
    const sofa::helper::vector< HexahedronQuads > &tta=getHexahedronQuadArray();


    for (unsigned int i = 0; i < m_hexahedron.size(); ++i)
    {
        // adding quad i in the edge shell of both points
        for (j=0; j<6; ++j)
        {
            m_hexahedronQuadShell[ tta[i][j] ].push_back( i );
        }
    }
}
const sofa::helper::vector<Hexahedron> &HexahedronSetTopologyContainer::getHexahedronArray()
{
    if (!m_hexahedron.size())
        createHexahedronSetArray();
    return m_hexahedron;
}


int HexahedronSetTopologyContainer::getHexahedronIndex(const unsigned int v1, const unsigned int v2, const unsigned int v3, const unsigned int v4, const unsigned int v5, const unsigned int v6, const unsigned int v7, const unsigned int v8)
{
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tvs=getHexahedronVertexShellArray();
    const sofa::helper::vector<unsigned int> &set1=tvs[v1];
    const sofa::helper::vector<unsigned int> &set2=tvs[v2];
    const sofa::helper::vector<unsigned int> &set3=tvs[v3];
    const sofa::helper::vector<unsigned int> &set4=tvs[v4];
    const sofa::helper::vector<unsigned int> &set5=tvs[v5];
    const sofa::helper::vector<unsigned int> &set6=tvs[v6];
    const sofa::helper::vector<unsigned int> &set7=tvs[v7];
    const sofa::helper::vector<unsigned int> &set8=tvs[v8];

    sofa::helper::vector<unsigned int> out1,out2,out3,out4,out5,out6,out7;
    std::set_intersection(set1.begin(),set1.end(),set2.begin(),set2.end(),out1.begin());
    std::set_intersection(set3.begin(),set3.end(),out1.begin(),out1.end(),out2.begin());
    std::set_intersection(set4.begin(),set4.end(),out2.begin(),out2.end(),out3.begin());
    std::set_intersection(set5.begin(),set5.end(),out3.begin(),out3.end(),out4.begin());
    std::set_intersection(set6.begin(),set6.end(),out4.begin(),out4.end(),out5.begin());
    std::set_intersection(set7.begin(),set7.end(),out5.begin(),out5.end(),out6.begin());
    std::set_intersection(set8.begin(),set8.end(),out6.begin(),out6.end(),out7.begin());

    assert(out7.size()==0 || out7.size()==1);
    if (out7.size()==1)
        return (int) (out7[0]);
    else
        return -1;
}

const Hexahedron &HexahedronSetTopologyContainer::getHexahedron(const unsigned int i)
{
    if (!m_hexahedron.size())
        createHexahedronSetArray();
    return m_hexahedron[i];
}



unsigned int HexahedronSetTopologyContainer::getNumberOfHexahedra()
{
    if (!m_hexahedron.size())
        createHexahedronSetArray();
    return m_hexahedron.size();
}



const sofa::helper::vector< sofa::helper::vector<unsigned int> > &HexahedronSetTopologyContainer::getHexahedronVertexShellArray()
{
    if (!m_hexahedronVertexShell.size())
        createHexahedronVertexShellArray();
    return m_hexahedronVertexShell;
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &HexahedronSetTopologyContainer::getHexahedronEdgeShellArray()
{
    if (!m_hexahedronEdgeShell.size())
        createHexahedronEdgeShellArray();
    return m_hexahedronEdgeShell;
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &HexahedronSetTopologyContainer::getHexahedronQuadShellArray()
{
    if (!m_hexahedronQuadShell.size())
        createHexahedronQuadShellArray();
    return m_hexahedronQuadShell;
}
const sofa::helper::vector< HexahedronEdges> &HexahedronSetTopologyContainer::getHexahedronEdgeArray()
{
    if (!m_hexahedronEdge.size())
        createHexahedronEdgeArray();
    return m_hexahedronEdge;
}

const sofa::helper::vector< HexahedronQuads> &HexahedronSetTopologyContainer::getHexahedronQuadArray()
{
    if (!m_hexahedronQuad.size())
        createHexahedronQuadArray();
    return m_hexahedronQuad;
}



const sofa::helper::vector< unsigned int > &HexahedronSetTopologyContainer::getHexahedronVertexShell(const unsigned int i)
{
    if (!m_hexahedronVertexShell.size())
        createHexahedronVertexShellArray();
    return m_hexahedronVertexShell[i];
}


const sofa::helper::vector< unsigned int > &HexahedronSetTopologyContainer::getHexahedronEdgeShell(const unsigned int i)
{
    if (!m_hexahedronEdgeShell.size())
        createHexahedronEdgeShellArray();
    return m_hexahedronEdgeShell[i];
}

const sofa::helper::vector< unsigned int > &HexahedronSetTopologyContainer::getHexahedronQuadShell(const unsigned int i)
{
    if (!m_hexahedronQuadShell.size())
        createHexahedronQuadShellArray();
    return m_hexahedronQuadShell[i];
}

const HexahedronEdges &HexahedronSetTopologyContainer::getHexahedronEdges(const unsigned int i)
{
    if (!m_hexahedronEdge.size())
        createHexahedronEdgeArray();
    return m_hexahedronEdge[i];
}

const HexahedronQuads &HexahedronSetTopologyContainer::getHexahedronQuads(const unsigned int i)
{
    if (!m_hexahedronQuad.size())
        createHexahedronQuadArray();
    return m_hexahedronQuad[i];
}

int HexahedronSetTopologyContainer::getVertexIndexInHexahedron(Hexahedron &t,unsigned int vertexIndex) const
{

    if (t[0]==vertexIndex)
        return 0;
    else if (t[1]==vertexIndex)
        return 1;
    else if (t[2]==vertexIndex)
        return 2;
    else if (t[3]==vertexIndex)
        return 3;
    else if (t[4]==vertexIndex)
        return 4;
    else if (t[5]==vertexIndex)
        return 5;
    else if (t[6]==vertexIndex)
        return 6;
    else if (t[7]==vertexIndex)
        return 7;
    else
        return -1;
}

sofa::helper::vector< unsigned int > &HexahedronSetTopologyContainer::getHexahedronEdgeShellForModification(const unsigned int i)
{
    if (!m_hexahedronEdgeShell.size())
        createHexahedronEdgeShellArray();
    return m_hexahedronEdgeShell[i];
}

sofa::helper::vector< unsigned int > &HexahedronSetTopologyContainer::getHexahedronVertexShellForModification(const unsigned int i)
{
    if (!m_hexahedronVertexShell.size())
        createHexahedronVertexShellArray();
    return m_hexahedronVertexShell[i];
}




HexahedronSetTopologyContainer::HexahedronSetTopologyContainer(core::componentmodel::topology::BaseTopology *top, const sofa::helper::vector< unsigned int > &DOFIndex,
        const sofa::helper::vector< Hexahedron >         &hexahedra )
    : QuadSetTopologyContainer( top,DOFIndex), m_hexahedron( hexahedra )
{

}



// factory related stuff

template<class DataTypes>
void create(HexahedronSetTopology<DataTypes>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    simulation::tree::xml::createWithParent< HexahedronSetTopology<DataTypes>, component::MechanicalObject<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        if (arg->getAttribute("filename"))
            obj->load(arg->getAttribute("filename"));
    }
}

Creator<simulation::tree::xml::ObjectFactory, HexahedronSetTopology<Vec3dTypes> >
HexahedronSetTopologyVec3dClass("HexahedronSetTopology", true);

Creator<simulation::tree::xml::ObjectFactory, HexahedronSetTopology<Vec3fTypes> >
HexahedronSetTopologyVec3fClass("HexahedronSetTopology", true);


} // namespace topology

} // namespace component

} // namespace sofa

