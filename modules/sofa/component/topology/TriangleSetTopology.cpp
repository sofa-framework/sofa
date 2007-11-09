#include <sofa/component/topology/TriangleSetTopology.h>
#include <sofa/component/topology/TriangleSetTopology.inl>
#include <sofa/defaulttype/Vec3Types.h>
//#include <sofa/simulation/tree/xml/ObjectFactory.h>
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
        .add< TriangleSetTopology<Vec3dTypes> >()
        .add< TriangleSetTopology<Vec3fTypes> >()
        .add< TriangleSetTopology<Vec2dTypes> >()
        .add< TriangleSetTopology<Vec2fTypes> >()
        .add< TriangleSetTopology<Vec1dTypes> >()
        .add< TriangleSetTopology<Vec1fTypes> >();

template class TriangleSetTopology<Vec3dTypes>;
template class TriangleSetTopology<Vec3fTypes>;
template class TriangleSetTopology<Vec2dTypes>;
template class TriangleSetTopology<Vec2fTypes>;
template class TriangleSetTopology<Vec1dTypes>;
template class TriangleSetTopology<Vec1fTypes>;

template class TriangleSetTopologyAlgorithms<Vec3fTypes>;
template class TriangleSetTopologyAlgorithms<Vec3dTypes>;
template class TriangleSetTopologyAlgorithms<Vec2fTypes>;
template class TriangleSetTopologyAlgorithms<Vec2dTypes>;
template class TriangleSetTopologyAlgorithms<Vec1fTypes>;
template class TriangleSetTopologyAlgorithms<Vec1dTypes>;

template class TriangleSetGeometryAlgorithms<Vec3fTypes>;
template class TriangleSetGeometryAlgorithms<Vec3dTypes>;
template class TriangleSetGeometryAlgorithms<Vec2fTypes>;
template class TriangleSetGeometryAlgorithms<Vec2dTypes>;
template class TriangleSetGeometryAlgorithms<Vec1fTypes>;
template class TriangleSetGeometryAlgorithms<Vec1dTypes>;

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
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tvs=getTriangleVertexShellArray();

    const sofa::helper::vector<unsigned int> &set1=tvs[v1];
    const sofa::helper::vector<unsigned int> &set2=tvs[v2];
    const sofa::helper::vector<unsigned int> &set3=tvs[v3];

    sofa::helper::vector<unsigned int> out1,out2;
    std::set_intersection(set1.begin(),set1.end(),set2.begin(),set2.end(),out1.begin());
    std::set_intersection(set3.begin(),set3.end(),out1.begin(),out1.end(),out2.begin());

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
    if (!m_triangleVertexShell.size())
        createTriangleVertexShellArray();
    return m_triangleVertexShell[i];
}


const sofa::helper::vector< unsigned int > &TriangleSetTopologyContainer::getTriangleEdgeShell(const unsigned int i)
{
    if (!m_triangleEdgeShell.size())
        createTriangleEdgeShellArray();
    return m_triangleEdgeShell[i];
}

const TriangleEdges &TriangleSetTopologyContainer::getTriangleEdge(const unsigned int i)
{
    if (!m_triangleEdge.size())
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
    if (!m_triangleEdgeShell.size())
        createTriangleEdgeShellArray();
    return m_triangleEdgeShell[i];
}
sofa::helper::vector< unsigned int > &TriangleSetTopologyContainer::getTriangleVertexShellForModification(const unsigned int i)
{
    if (!m_triangleVertexShell.size())
        createTriangleVertexShellArray();
    return m_triangleVertexShell[i];
}



TriangleSetTopologyContainer::TriangleSetTopologyContainer(core::componentmodel::topology::BaseTopology *top, const sofa::helper::vector< unsigned int > &DOFIndex,
        const sofa::helper::vector< Triangle >         &triangles )
    : EdgeSetTopologyContainer( top,DOFIndex), m_triangle( triangles )
{

}
bool TriangleSetTopologyContainer::checkTopology() const
{
    EdgeSetTopologyContainer::checkTopology();
    if (m_triangleVertexShell.size()>0)
    {
        unsigned int i,j;
        for (i=0; i<m_triangleVertexShell.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tvs=m_triangleVertexShell[i];
            for (j=0; j<tvs.size(); ++j)
                assert((m_triangle[tvs[j]][0]==i) ||  (m_triangle[tvs[j]][1]==i) || (m_triangle[tvs[j]][2]==i));
        }
    }

    if (m_triangleEdgeShell.size()>0)
    {
        unsigned int i,j;
        for (i=0; i<m_triangleEdgeShell.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tes=m_triangleEdgeShell[i];
            for (j=0; j<tes.size(); ++j)
                assert((m_triangleEdge[tes[j]][0]==i) ||  (m_triangleEdge[tes[j]][1]==i) || (m_triangleEdge[tes[j]][2]==i));
        }
    }
    return true;
}


// factory related stuff
/*
template<class DataTypes>
void create(TriangleSetTopology<DataTypes>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
	simulation::tree::xml::createWithParent< TriangleSetTopology<DataTypes>, component::MechanicalObject<DataTypes> >(obj, arg);
	if (obj!=NULL)
	{
		if (arg->getAttribute("filename"))
			obj->load(arg->getAttribute("filename"));
	}
}

*/

} // namespace topology

} // namespace component

} // namespace sofa

