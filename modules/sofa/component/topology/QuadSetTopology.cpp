#include <sofa/component/topology/QuadSetTopology.h>
#include <sofa/component/topology/QuadSetTopology.inl>
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


SOFA_DECL_CLASS(QuadSetTopology)

int QuadSetTopologyClass = core::RegisterObject("Quad set topology")
        .add< QuadSetTopology<Vec3dTypes> >()
        .add< QuadSetTopology<Vec3fTypes> >()
        .add< QuadSetTopology<Vec2dTypes> >()
        .add< QuadSetTopology<Vec2fTypes> >()
        .add< QuadSetTopology<Vec1dTypes> >()
        .add< QuadSetTopology<Vec1fTypes> >();

template class QuadSetTopology<Vec3dTypes>;
template class QuadSetTopology<Vec3fTypes>;
template class QuadSetTopology<Vec2dTypes>;
template class QuadSetTopology<Vec2fTypes>;
template class QuadSetTopology<Vec1dTypes>;
template class QuadSetTopology<Vec1fTypes>;

template class QuadSetTopologyAlgorithms<Vec3fTypes>;
template class QuadSetTopologyAlgorithms<Vec3dTypes>;
template class QuadSetTopologyAlgorithms<Vec2fTypes>;
template class QuadSetTopologyAlgorithms<Vec2dTypes>;
template class QuadSetTopologyAlgorithms<Vec1fTypes>;
template class QuadSetTopologyAlgorithms<Vec1dTypes>;

template class QuadSetGeometryAlgorithms<Vec3fTypes>;
template class QuadSetGeometryAlgorithms<Vec3dTypes>;
template class QuadSetGeometryAlgorithms<Vec2fTypes>;
template class QuadSetGeometryAlgorithms<Vec2dTypes>;
template class QuadSetGeometryAlgorithms<Vec1fTypes>;
template class QuadSetGeometryAlgorithms<Vec1dTypes>;

// implementation QuadSetTopologyContainer

void QuadSetTopologyContainer::createQuadVertexShellArray ()
{
    m_quadVertexShell.resize( m_basicTopology->getDOFNumber() );
    unsigned int j;

    for (unsigned int i = 0; i < m_quad.size(); ++i)
    {
        // adding edge i in the edge shell of both points
        for (j=0; j<4; ++j)
            m_quadVertexShell[ m_quad[i][j]  ].push_back( i );
    }
}

void QuadSetTopologyContainer::createQuadEdgeShellArray ()
{
    m_quadEdgeShell.resize( getNumberOfEdges());
    unsigned int j;
    const sofa::helper::vector< QuadEdges > &tea=getQuadEdgeArray();


    for (unsigned int i = 0; i < m_quad.size(); ++i)
    {
        // adding edge i in the edge shell of both points
        for (j=0; j<4; ++j)
        {
            m_quadEdgeShell[ tea[i][j] ].push_back( i );
        }
    }
}


void QuadSetTopologyContainer::createQuadEdgeArray ()
{
    m_quadEdge.resize( getNumberOfQuads());
    unsigned int j;
    int edgeIndex;

    if (m_edge.size()>0)
    {

        for (unsigned int i = 0; i < m_quad.size(); ++i)
        {
            Quad &t=m_quad[i];
            // adding edge i in the edge shell of both points
            for (j=0; j<4; ++j)
            {
                edgeIndex=getEdgeIndex(t[(j+1)%4],t[(j+2)%4]);
                assert(edgeIndex!= -1);
                m_quadEdge[i][j]=edgeIndex;
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
        /// create the m_edge array at the same time than it fills the m_hexahedronEdges array
        for (unsigned int i = 0; i < m_quad.size(); ++i)
        {
            Quad &t=m_quad[i];
            for (j=0; j<4; ++j)
            {
                v1=t[(j+1)%4];
                v2=t[(j+2)%4];
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
                m_quadEdge[i][j]=edgeIndex;
            }
        }
    }
}


const sofa::helper::vector<Quad> &QuadSetTopologyContainer::getQuadArray()
{
    if (!m_quad.size())
        createQuadSetArray();
    return m_quad;
}


int QuadSetTopologyContainer::getQuadIndex(const unsigned int v1, const unsigned int v2, const unsigned int v3,  const unsigned int v4)
{
    const sofa::helper::vector< sofa::helper::vector<unsigned int> > &tvs=getQuadVertexShellArray();

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

const Quad &QuadSetTopologyContainer::getQuad(const unsigned int i)
{
    if (!m_quad.size())
        createQuadSetArray();
    return m_quad[i];
}



unsigned int QuadSetTopologyContainer::getNumberOfQuads()
{
    if (!m_quad.size())
        createQuadSetArray();
    return m_quad.size();
}



const sofa::helper::vector< sofa::helper::vector<unsigned int> > &QuadSetTopologyContainer::getQuadVertexShellArray()
{
    if (!m_quadVertexShell.size())
        createQuadVertexShellArray();
    return m_quadVertexShell;
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &QuadSetTopologyContainer::getQuadEdgeShellArray()
{
    if (!m_quadEdgeShell.size())
        createQuadEdgeShellArray();
    return m_quadEdgeShell;
}

const sofa::helper::vector< QuadEdges> &QuadSetTopologyContainer::getQuadEdgeArray()
{
    if (!m_quadEdge.size())
        createQuadEdgeArray();
    return m_quadEdge;
}




const sofa::helper::vector< unsigned int > &QuadSetTopologyContainer::getQuadVertexShell(const unsigned int i)
{
    if (!m_quadVertexShell.size())
        createQuadVertexShellArray();
    return m_quadVertexShell[i];
}


const sofa::helper::vector< unsigned int > &QuadSetTopologyContainer::getQuadEdgeShell(const unsigned int i)
{
    if (!m_quadEdgeShell.size())
        createQuadEdgeShellArray();
    return m_quadEdgeShell[i];
}

const QuadEdges &QuadSetTopologyContainer::getQuadEdge(const unsigned int i)
{
    if (!m_quadEdge.size())
        createQuadEdgeArray();
    return m_quadEdge[i];
}

int QuadSetTopologyContainer::getVertexIndexInQuad(Quad &t,unsigned int vertexIndex) const
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
int QuadSetTopologyContainer::getEdgeIndexInQuad(QuadEdges &t,unsigned int edgeIndex) const
{

    if (t[0]==edgeIndex)
        return 0;
    else if (t[1]==edgeIndex)
        return 1;
    else if (t[2]==edgeIndex)
        return 2;
    else if (t[3]==edgeIndex)
        return 3;
    else
        return -1;
}

sofa::helper::vector< unsigned int > &QuadSetTopologyContainer::getQuadEdgeShellForModification(const unsigned int i)
{
    if (!m_quadEdgeShell.size())
        createQuadEdgeShellArray();
    return m_quadEdgeShell[i];
}
sofa::helper::vector< unsigned int > &QuadSetTopologyContainer::getQuadVertexShellForModification(const unsigned int i)
{
    if (!m_quadVertexShell.size())
        createQuadVertexShellArray();
    return m_quadVertexShell[i];
}



QuadSetTopologyContainer::QuadSetTopologyContainer(core::componentmodel::topology::BaseTopology *top, /* const sofa::helper::vector< unsigned int > &DOFIndex, */
        const sofa::helper::vector< Quad >         &quads )
    : EdgeSetTopologyContainer( top /*,DOFIndex*/), m_quad( quads )
{

}
bool QuadSetTopologyContainer::checkTopology() const
{
    EdgeSetTopologyContainer::checkTopology();
    if (m_quadVertexShell.size()>0)
    {
        unsigned int i,j;
        for (i=0; i<m_quadVertexShell.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tvs=m_quadVertexShell[i];
            for (j=0; j<tvs.size(); ++j)
                assert((m_quad[tvs[j]][0]==i) ||  (m_quad[tvs[j]][1]==i) || (m_quad[tvs[j]][2]==i) || (m_quad[tvs[j]][3]==i));
        }
    }

    if (m_quadEdgeShell.size()>0)
    {
        unsigned int i,j;
        for (i=0; i<m_quadEdgeShell.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tes=m_quadEdgeShell[i];
            for (j=0; j<tes.size(); ++j)
                assert((m_quadEdge[tes[j]][0]==i) ||  (m_quadEdge[tes[j]][1]==i) || (m_quadEdge[tes[j]][2]==i) || (m_quadEdge[tes[j]][3]==i));
        }
    }
    return true;
}


// factory related stuff
/*
template<class DataTypes>
void create(QuadSetTopology<DataTypes>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
	simulation::tree::xml::createWithParent< QuadSetTopology<DataTypes>, component::MechanicalObject<DataTypes> >(obj, arg);
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

