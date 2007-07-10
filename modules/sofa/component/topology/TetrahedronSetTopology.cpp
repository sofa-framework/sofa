#include <sofa/component/topology/TetrahedronSetTopology.h>
#include <sofa/component/topology/TetrahedronSetTopology.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(TetrahedronSetTopology)


template class TetrahedronSetTopology<Vec3dTypes>;
template class TetrahedronSetTopology<Vec3fTypes>;
template class TetrahedronSetTopology<Vec2dTypes>;
template class TetrahedronSetTopology<Vec2fTypes>;
template class TetrahedronSetTopology<Vec1dTypes>;
template class TetrahedronSetTopology<Vec1fTypes>;

template class TetrahedronSetTopologyAlgorithms<Vec3fTypes>;
template class TetrahedronSetTopologyAlgorithms<Vec3dTypes>;
template class TetrahedronSetTopologyAlgorithms<Vec2fTypes>;
template class TetrahedronSetTopologyAlgorithms<Vec2dTypes>;
template class TetrahedronSetTopologyAlgorithms<Vec1fTypes>;
template class TetrahedronSetTopologyAlgorithms<Vec1dTypes>;

template class TetrahedronSetGeometryAlgorithms<Vec3fTypes>;
template class TetrahedronSetGeometryAlgorithms<Vec3dTypes>;
template class TetrahedronSetGeometryAlgorithms<Vec2fTypes>;
template class TetrahedronSetGeometryAlgorithms<Vec2dTypes>;
template class TetrahedronSetGeometryAlgorithms<Vec1fTypes>;
template class TetrahedronSetGeometryAlgorithms<Vec1dTypes>;


// implementation TetrahedronSetTopologyContainer


void TetrahedronSetTopologyContainer::createTetrahedronEdgeArray ()
{
    m_tetrahedronEdge.resize( getNumberOfTetrahedra());
    unsigned int j;
    int edgeIndex;
    unsigned int edgeTetrahedronDescriptionArray[6][2]= {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

    if (m_edge.size()>0)
    {
        for (unsigned int i = 0; i < m_tetrahedron.size(); ++i)
        {
            Tetrahedron &t=m_tetrahedron[i];
            // adding edge i in the edge shell of both points
            for (j=0; j<6; ++j)
            {
                edgeIndex=getEdgeIndex(t[edgeTetrahedronDescriptionArray[j][0]],
                        t[edgeTetrahedronDescriptionArray[j][1]]);
                assert(edgeIndex!= -1);
                m_tetrahedronEdge[i][j]=edgeIndex;
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
        /// create the m_edge array at the same time than it fills the m_tetrahedronEdge array
        for (unsigned int i = 0; i < m_tetrahedron.size(); ++i)
        {
            Tetrahedron &t=m_tetrahedron[i];
            for (j=0; j<6; ++j)
            {
                v1=t[edgeTetrahedronDescriptionArray[j][0]];
                v2=t[edgeTetrahedronDescriptionArray[j][1]];
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
                m_tetrahedronEdge[i][j]=edgeIndex;
            }
        }
    }
}


void TetrahedronSetTopologyContainer::createTetrahedronTriangleArray ()
{
    m_tetrahedronTriangle.resize( getNumberOfTetrahedra());
    unsigned int j;
    int triangleIndex;

    if (m_triangle.size()>0)
    {
        for (unsigned int i = 0; i < m_tetrahedron.size(); ++i)
        {
            Tetrahedron &t=m_tetrahedron[i];
            // adding triangles in the triangle list of the ith tetrahedron  i
            for (j=0; j<4; ++j)
            {
                triangleIndex=getTriangleIndex(t[(j+1)%4],t[(j+2)%4],t[(j+3)%4]);
                assert(triangleIndex!= -1);
                m_tetrahedronTriangle[i][j]=triangleIndex;
            }
        }
    }
    else
    {
        // create a temporary map to find redundant triangles
        std::map<Triangle,unsigned int> triangleMap;
        std::map<Triangle,unsigned int>::iterator itt;
        Triangle tr;
        unsigned int v[3];
        /// create the m_edge array at the same time than it fills the m_tetrahedronEdge array
        for (unsigned int i = 0; i < m_tetrahedron.size(); ++i)
        {
            Tetrahedron &t=m_tetrahedron[i];
            for (j=0; j<4; ++j)
            {
                v[0]=t[(j+1)%4]; v[1]=t[(j+2)%4]; v[2]=t[(j+3)%4];
                std::sort(v,v+2);
                // sort vertices in lexicographics order
                tr=make_array<unsigned int>(v[0],v[1],v[2]);
                itt=triangleMap.find(tr);
                if (itt==triangleMap.end())
                {
                    // edge not in edgeMap so create a new one
                    triangleIndex=triangleMap.size();
                    triangleMap[tr]=triangleIndex;
                    m_triangle.push_back(tr);
                }
                else
                {
                    triangleIndex=(*itt).second;
                }
                m_tetrahedronTriangle[i][j]=triangleIndex;
            }
        }
    }
}

void TetrahedronSetTopologyContainer::createTetrahedronVertexShellArray ()
{
    m_tetrahedronVertexShell.resize( m_basicTopology->getDOFNumber() );
    unsigned int j;

    for (unsigned int i = 0; i < m_tetrahedron.size(); ++i)
    {
        // adding edge i in the edge shell of both points
        for (j=0; j<4; ++j)
            m_tetrahedronVertexShell[ m_tetrahedron[i][j]  ].push_back( i );
    }
}

void TetrahedronSetTopologyContainer::createTetrahedronEdgeShellArray ()
{
    m_tetrahedronEdgeShell.resize( getNumberOfEdges());
    unsigned int j;
    const std::vector< TetrahedronEdges > &tea=getTetrahedronEdgeArray();


    for (unsigned int i = 0; i < m_tetrahedron.size(); ++i)
    {
        // adding edge i in the edge shell of both points
        for (j=0; j<6; ++j)
        {
            m_tetrahedronEdgeShell[ tea[i][j] ].push_back( i );
        }
    }
}

void TetrahedronSetTopologyContainer::createTetrahedronTriangleShellArray ()
{
    m_tetrahedronTriangleShell.resize( getNumberOfTriangles());
    unsigned int j;
    const std::vector< TetrahedronTriangles > &tta=getTetrahedronTriangleArray();


    for (unsigned int i = 0; i < m_tetrahedron.size(); ++i)
    {
        // adding edge i in the edge shell of both points
        for (j=0; j<4; ++j)
        {
            m_tetrahedronTriangleShell[ tta[i][j] ].push_back( i );
        }
    }
}
const std::vector<Tetrahedron> &TetrahedronSetTopologyContainer::getTetrahedronArray()
{
    if (!m_tetrahedron.size())
        createTetrahedronSetArray();
    return m_tetrahedron;
}


int TetrahedronSetTopologyContainer::getTetrahedronIndex(const unsigned int v1, const unsigned int v2, const unsigned int v3, const unsigned int v4)
{
    const std::vector< std::vector<unsigned int> > &tvs=getTetrahedronVertexShellArray();
    const std::vector<unsigned int> &set1=tvs[v1];
    const std::vector<unsigned int> &set2=tvs[v2];
    const std::vector<unsigned int> &set3=tvs[v3];
    const std::vector<unsigned int> &set4=tvs[v4];

    std::vector<unsigned int> out1,out2,out3;
    std::set_intersection(set1.begin(),set1.end(),set2.begin(),set2.end(),out1.begin());
    std::set_intersection(set3.begin(),set3.end(),out1.begin(),out1.end(),out2.begin());
    std::set_intersection(set4.begin(),set4.end(),out2.begin(),out2.end(),out3.begin());

    assert(out3.size()==0 || out3.size()==1);
    if (out3.size()==1)
        return (int) (out3[0]);
    else
        return -1;
}

const Tetrahedron &TetrahedronSetTopologyContainer::getTetrahedron(const unsigned int i)
{
    if (!m_tetrahedron.size())
        createTetrahedronSetArray();
    return m_tetrahedron[i];
}



unsigned int TetrahedronSetTopologyContainer::getNumberOfTetrahedra()
{
    if (!m_tetrahedron.size())
        createTetrahedronSetArray();
    return m_tetrahedron.size();
}



const std::vector< std::vector<unsigned int> > &TetrahedronSetTopologyContainer::getTetrahedronVertexShellArray()
{
    if (!m_tetrahedronVertexShell.size())
        createTetrahedronVertexShellArray();
    return m_tetrahedronVertexShell;
}

const std::vector< std::vector<unsigned int> > &TetrahedronSetTopologyContainer::getTetrahedronEdgeShellArray()
{
    if (!m_tetrahedronEdgeShell.size())
        createTetrahedronEdgeShellArray();
    return m_tetrahedronEdgeShell;
}

const std::vector< std::vector<unsigned int> > &TetrahedronSetTopologyContainer::getTetrahedronTriangleShellArray()
{
    if (!m_tetrahedronTriangleShell.size())
        createTetrahedronTriangleShellArray();
    return m_tetrahedronTriangleShell;
}
const std::vector< TetrahedronEdges> &TetrahedronSetTopologyContainer::getTetrahedronEdgeArray()
{
    if (!m_tetrahedronEdge.size())
        createTetrahedronEdgeArray();
    return m_tetrahedronEdge;
}

const std::vector< TetrahedronTriangles> &TetrahedronSetTopologyContainer::getTetrahedronTriangleArray()
{
    if (!m_tetrahedronTriangle.size())
        createTetrahedronTriangleArray();
    return m_tetrahedronTriangle;
}



const std::vector< unsigned int > &TetrahedronSetTopologyContainer::getTetrahedronVertexShell(const unsigned int i)
{
    if (!m_tetrahedronVertexShell.size())
        createTetrahedronVertexShellArray();
    return m_tetrahedronVertexShell[i];
}


const std::vector< unsigned int > &TetrahedronSetTopologyContainer::getTetrahedronEdgeShell(const unsigned int i)
{
    if (!m_tetrahedronEdgeShell.size())
        createTetrahedronEdgeShellArray();
    return m_tetrahedronEdgeShell[i];
}

const std::vector< unsigned int > &TetrahedronSetTopologyContainer::getTetrahedronTriangleShell(const unsigned int i)
{
    if (!m_tetrahedronTriangleShell.size())
        createTetrahedronTriangleShellArray();
    return m_tetrahedronTriangleShell[i];
}

const TetrahedronEdges &TetrahedronSetTopologyContainer::getTetrahedronEdges(const unsigned int i)
{
    if (!m_tetrahedronEdge.size())
        createTetrahedronEdgeArray();
    return m_tetrahedronEdge[i];
}

const TetrahedronTriangles &TetrahedronSetTopologyContainer::getTetrahedronTriangles(const unsigned int i)
{
    if (!m_tetrahedronTriangle.size())
        createTetrahedronTriangleArray();
    return m_tetrahedronTriangle[i];
}

int TetrahedronSetTopologyContainer::getVertexIndexInTetrahedron(Tetrahedron &t,unsigned int vertexIndex) const
{

    if (t[0]==vertexIndex)
        return 0;
    else if (t[1]==vertexIndex)
        return 1;
    else if (t[2]==vertexIndex)
        return 2;
    else if (t[3]==vertexIndex)
        return 3;
    else
        return -1;
}

std::vector< unsigned int > &TetrahedronSetTopologyContainer::getTetrahedronEdgeShellForModification(const unsigned int i)
{
    if (!m_tetrahedronEdgeShell.size())
        createTetrahedronEdgeShellArray();
    return m_tetrahedronEdgeShell[i];
}

std::vector< unsigned int > &TetrahedronSetTopologyContainer::getTetrahedronVertexShellForModification(const unsigned int i)
{
    if (!m_tetrahedronVertexShell.size())
        createTetrahedronVertexShellArray();
    return m_tetrahedronVertexShell[i];
}




TetrahedronSetTopologyContainer::TetrahedronSetTopologyContainer(core::componentmodel::topology::BaseTopology *top, const std::vector< unsigned int > &DOFIndex,
        const std::vector< Tetrahedron >         &tetrahedra )
    : TriangleSetTopologyContainer( top,DOFIndex), m_tetrahedron( tetrahedra )
{

}



// factory related stuff

template<class DataTypes>
void create(TetrahedronSetTopology<DataTypes>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    simulation::tree::xml::createWithParent< TetrahedronSetTopology<DataTypes>, component::MechanicalObject<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        if (arg->getAttribute("filename"))
            obj->load(arg->getAttribute("filename"));
    }
}

Creator<simulation::tree::xml::ObjectFactory, TetrahedronSetTopology<Vec3dTypes> >
TetrahedronSetTopologyVec3dClass("TetrahedronSetTopology", true);

Creator<simulation::tree::xml::ObjectFactory, TetrahedronSetTopology<Vec3fTypes> >
TetrahedronSetTopologyVec3fClass("TetrahedronSetTopology", true);


} // namespace topology

} // namespace component

} // namespace sofa

