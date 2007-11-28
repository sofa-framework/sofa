#include <sofa/component/topology/TetrahedronSetTopology.h>
#include <sofa/component/topology/TetrahedronSetTopology.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(TetrahedronSetTopology)

int TetrahedronSetTopologyClass = core::RegisterObject("Tetrahedron set topology")
        .add< TetrahedronSetTopology<Vec3dTypes> >()
        .add< TetrahedronSetTopology<Vec3fTypes> >()
        .add< TetrahedronSetTopology<Vec2dTypes> >()
        .add< TetrahedronSetTopology<Vec2fTypes> >()
        .add< TetrahedronSetTopology<Vec1dTypes> >()
        .add< TetrahedronSetTopology<Vec1fTypes> >();

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

    if (m_edge.size()>0)
    {
        for (unsigned int i = 0; i < m_tetrahedron.size(); ++i)
        {
            Tetrahedron &t=m_tetrahedron[i];
            // adding edge i in the edge shell of both points
            for (j=0; j<6; ++j)
            {
                edgeIndex=getEdgeIndex(t[tetrahedronEdgeArray[j][0]],
                        t[tetrahedronEdgeArray[j][1]]);
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
                v1=t[tetrahedronEdgeArray[j][0]];
                v2=t[tetrahedronEdgeArray[j][1]];
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
        unsigned int v[3],val;
        /// create the m_edge array at the same time than it fills the m_tetrahedronEdge array
        for (unsigned int i = 0; i < m_tetrahedron.size(); ++i)
        {
            Tetrahedron &t=m_tetrahedron[i];
            for (j=0; j<4; ++j)
            {
                if (j%2)
                {
                    v[0]=t[(j+1)%4]; v[1]=t[(j+2)%4]; v[2]=t[(j+3)%4];
                }
                else
                {
                    v[0]=t[(j+1)%4]; v[2]=t[(j+2)%4]; v[1]=t[(j+3)%4];
                }
                //		std::sort(v,v+2);
                // sort v such that v[0] is the smallest one
                while ((v[0]>v[1]) || (v[0]>v[2]))
                {
                    val=v[0]; v[0]=v[1]; v[1]=v[2]; v[2]=val;
                }
                // check if a triangle with an opposite orientation already exists
                tr=helper::make_array<unsigned int>(v[0],v[2],v[1]);
                itt=triangleMap.find(tr);
                if (itt==triangleMap.end())
                {
                    // edge not in edgeMap so create a new one
                    triangleIndex=triangleMap.size();
                    tr=helper::make_array<unsigned int>(v[0],v[1],v[2]);
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
    const sofa::helper::vector< TetrahedronEdges > &tea=getTetrahedronEdgeArray();


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
    const sofa::helper::vector< TetrahedronTriangles > &tta=getTetrahedronTriangleArray();


    for (unsigned int i = 0; i < m_tetrahedron.size(); ++i)
    {
        // adding edge i in the edge shell of both points
        for (j=0; j<4; ++j)
        {
            m_tetrahedronTriangleShell[ tta[i][j] ].push_back( i );
        }
    }
}
const sofa::helper::vector<Tetrahedron> &TetrahedronSetTopologyContainer::getTetrahedronArray()
{
    if (!m_tetrahedron.size())
        createTetrahedronSetArray();
    return m_tetrahedron;
}


int TetrahedronSetTopologyContainer::getTetrahedronIndex(const unsigned int v1, const unsigned int v2, const unsigned int v3, const unsigned int v4)
{
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tvs=getTetrahedronVertexShellArray();
    const sofa::helper::vector<unsigned int> &set1=tvs[v1];
    const sofa::helper::vector<unsigned int> &set2=tvs[v2];
    const sofa::helper::vector<unsigned int> &set3=tvs[v3];
    const sofa::helper::vector<unsigned int> &set4=tvs[v4];

    sofa::helper::vector<unsigned int> out1,out2,out3;
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



const sofa::helper::vector< sofa::helper::vector<unsigned int> > &TetrahedronSetTopologyContainer::getTetrahedronVertexShellArray()
{
    if (!m_tetrahedronVertexShell.size())
        createTetrahedronVertexShellArray();
    return m_tetrahedronVertexShell;
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &TetrahedronSetTopologyContainer::getTetrahedronEdgeShellArray()
{
    if (!m_tetrahedronEdgeShell.size())
        createTetrahedronEdgeShellArray();
    return m_tetrahedronEdgeShell;
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &TetrahedronSetTopologyContainer::getTetrahedronTriangleShellArray()
{
    if (!m_tetrahedronTriangleShell.size())
        createTetrahedronTriangleShellArray();
    return m_tetrahedronTriangleShell;
}
const sofa::helper::vector< TetrahedronEdges> &TetrahedronSetTopologyContainer::getTetrahedronEdgeArray()
{
    if (!m_tetrahedronEdge.size())
        createTetrahedronEdgeArray();
    return m_tetrahedronEdge;
}
std::pair<unsigned int,unsigned int> TetrahedronSetTopologyContainer::getLocalTetrahedronEdges (const unsigned int i) const
{
    assert(i<6);
    return std::pair<unsigned int,unsigned int> (tetrahedronEdgeArray[i][0],tetrahedronEdgeArray[i][1]);
}

const sofa::helper::vector< TetrahedronTriangles> &TetrahedronSetTopologyContainer::getTetrahedronTriangleArray()
{
    if (!m_tetrahedronTriangle.size())
        createTetrahedronTriangleArray();
    return m_tetrahedronTriangle;
}



const sofa::helper::vector< unsigned int > &TetrahedronSetTopologyContainer::getTetrahedronVertexShell(const unsigned int i)
{
    if (!m_tetrahedronVertexShell.size())
        createTetrahedronVertexShellArray();
    return m_tetrahedronVertexShell[i];
}


const sofa::helper::vector< unsigned int > &TetrahedronSetTopologyContainer::getTetrahedronEdgeShell(const unsigned int i)
{
    if (!m_tetrahedronEdgeShell.size())
        createTetrahedronEdgeShellArray();
    return m_tetrahedronEdgeShell[i];
}

const sofa::helper::vector< unsigned int > &TetrahedronSetTopologyContainer::getTetrahedronTriangleShell(const unsigned int i)
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

int TetrahedronSetTopologyContainer::getVertexIndexInTetrahedron(const Tetrahedron &t,unsigned int vertexIndex) const
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

int TetrahedronSetTopologyContainer::getEdgeIndexInTetrahedron(const TetrahedronEdges &t,const unsigned int edgeIndex) const
{

    if (t[0]==edgeIndex)
        return 0;
    else if (t[1]==edgeIndex)
        return 1;
    else if (t[2]==edgeIndex)
        return 2;
    else if (t[3]==edgeIndex)
        return 3;
    else if (t[4]==edgeIndex)
        return 4;
    else if (t[5]==edgeIndex)
        return 5;
    else
        return -1;
}

int TetrahedronSetTopologyContainer::getTriangleIndexInTetrahedron(const TetrahedronTriangles &t,const unsigned int triangleIndex) const
{

    if (t[0]==triangleIndex)
        return 0;
    else if (t[1]==triangleIndex)
        return 1;
    else if (t[2]==triangleIndex)
        return 2;
    else if (t[3]==triangleIndex)
        return 3;
    else
        return -1;
}

sofa::helper::vector< unsigned int > &TetrahedronSetTopologyContainer::getTetrahedronEdgeShellForModification(const unsigned int i)
{
    if (!m_tetrahedronEdgeShell.size())
        createTetrahedronEdgeShellArray();
    return m_tetrahedronEdgeShell[i];
}

sofa::helper::vector< unsigned int > &TetrahedronSetTopologyContainer::getTetrahedronVertexShellForModification(const unsigned int i)
{
    if (!m_tetrahedronVertexShell.size())
        createTetrahedronVertexShellArray();
    return m_tetrahedronVertexShell[i];
}




TetrahedronSetTopologyContainer::TetrahedronSetTopologyContainer(core::componentmodel::topology::BaseTopology *top, const sofa::helper::vector< unsigned int > &DOFIndex,
        const sofa::helper::vector< Tetrahedron >         &tetrahedra )
    : TriangleSetTopologyContainer( top,DOFIndex), m_tetrahedron( tetrahedra )
{

}

bool TetrahedronSetTopologyContainer::checkTopology() const
{
    //std::cout << "*** CHECK TetrahedronSetTopologyContainer ***" << std::endl;

    TriangleSetTopologyContainer::checkTopology();
    if (m_tetrahedronVertexShell.size()>0)
    {
        unsigned int i,j;
        for (i=0; i<m_tetrahedronVertexShell.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tvs=m_tetrahedronVertexShell[i];
            for (j=0; j<tvs.size(); ++j)
            {
                bool check_tetra_vertex_shell = (m_tetrahedron[tvs[j]][0]==i) ||  (m_tetrahedron[tvs[j]][1]==i) || (m_tetrahedron[tvs[j]][2]==i) || (m_tetrahedron[tvs[j]][3]==i);
                if(!check_tetra_vertex_shell)
                {
                    std::cout << "*** CHECK FAILED : check_tetra_vertex_shell, i = " << i << " , j = " << j << std::endl;
                }
                assert(check_tetra_vertex_shell);
            }
        }
        //std::cout << "******** DONE : check_tetra_vertex_shell" << std::endl;
    }

    if (m_tetrahedronEdgeShell.size()>0)
    {
        unsigned int i,j;
        for (i=0; i<m_tetrahedronEdgeShell.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tes=m_tetrahedronEdgeShell[i];
            for (j=0; j<tes.size(); ++j)
            {
                bool check_tetra_edge_shell = (m_tetrahedronEdge[tes[j]][0]==i) ||  (m_tetrahedronEdge[tes[j]][1]==i) || (m_tetrahedronEdge[tes[j]][2]==i) || (m_tetrahedronEdge[tes[j]][3]==i) || (m_tetrahedronEdge[tes[j]][4]==i) || (m_tetrahedronEdge[tes[j]][5]==i);
                if(!check_tetra_edge_shell)
                {
                    std::cout << "*** CHECK FAILED : check_tetra_edge_shell, i = " << i << " , j = " << j << std::endl;
                }
                assert(check_tetra_edge_shell);
            }
        }
        //std::cout << "******** DONE : check_tetra_edge_shell" << std::endl;
    }

    if (m_tetrahedronTriangleShell.size()>0)
    {
        unsigned int i,j;
        for (i=0; i<m_tetrahedronTriangleShell.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tes=m_tetrahedronTriangleShell[i];
            for (j=0; j<tes.size(); ++j)
            {
                bool check_tetra_triangle_shell = (m_tetrahedronTriangle[tes[j]][0]==i) ||  (m_tetrahedronTriangle[tes[j]][1]==i) || (m_tetrahedronTriangle[tes[j]][2]==i) || (m_tetrahedronTriangle[tes[j]][3]==i);
                if(!check_tetra_triangle_shell)
                {
                    std::cout << "*** CHECK FAILED : check_tetra_triangle_shell, i = " << i << " , j = " << j << std::endl;
                }
                assert(check_tetra_triangle_shell);
            }
        }
        //std::cout << "******** DONE : check_tetra_triangle_shell" << std::endl;
    }
    return true;
}


// factory related stuff
/*
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
*/

} // namespace topology

} // namespace component

} // namespace sofa

