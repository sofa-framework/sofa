#include <sofa/component/topology/EdgeSetTopology.h>
#include <sofa/component/topology/EdgeSetTopology.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(EdgeSetTopology)


template class EdgeSetTopology<Vec3dTypes>;
template class EdgeSetTopology<Vec3fTypes>;
template class EdgeSetTopology<Vec2dTypes>;
template class EdgeSetTopology<Vec2fTypes>;
template class EdgeSetTopology<Vec1dTypes>;
template class EdgeSetTopology<Vec1fTypes>;

template class EdgeSetTopologyAlgorithms<Vec3fTypes>;
template class EdgeSetTopologyAlgorithms<Vec3dTypes>;
template class EdgeSetTopologyAlgorithms<Vec2dTypes>;
template class EdgeSetTopologyAlgorithms<Vec2fTypes>;
template class EdgeSetTopologyAlgorithms<Vec1dTypes>;
template class EdgeSetTopologyAlgorithms<Vec1fTypes>;

template class EdgeSetGeometryAlgorithms<Vec3fTypes>;
template class EdgeSetGeometryAlgorithms<Vec3dTypes>;
template class EdgeSetGeometryAlgorithms<Vec2dTypes>;
template class EdgeSetGeometryAlgorithms<Vec2fTypes>;
template class EdgeSetGeometryAlgorithms<Vec1dTypes>;
template class EdgeSetGeometryAlgorithms<Vec1fTypes>;
// implementation EdgeSetTopologyContainer

void EdgeSetTopologyContainer::createEdgeVertexShellArray ()
{
    m_edgeVertexShell.resize( m_basicTopology->getDOFNumber() );

    for (unsigned int i = 0; i < m_edge.size(); ++i)
    {
        // adding edge i in the edge shell of both points
        m_edgeVertexShell[ m_edge[i].first  ].push_back( i );
        m_edgeVertexShell[ m_edge[i].second ].push_back( i );
    }
}



const std::vector<Edge> &EdgeSetTopologyContainer::getEdgeArray()
{
    if (!m_edge.size())
        createEdgeSetArray();
    return m_edge;
}


int EdgeSetTopologyContainer::getEdgeIndex(const unsigned int v1, const unsigned int v2)
{
    const std::vector< unsigned int > &es1=getEdgeVertexShell(v1) ;
    const std::vector<Edge> &ea=getEdgeArray();
    unsigned int i=0;
    int result= -1;
    while ((i<es1.size()) && (result== -1))
    {
        const Edge &e=ea[es1[i]];
        if ((e.first==v2)|| (e.second==v2))
            result=(int) es1[i];

        i++;
    }
    return result;
}

const Edge &EdgeSetTopologyContainer::getEdge(const unsigned int i)
{
    if (!m_edge.size())
        createEdgeSetArray();
    return m_edge[i];
}
bool EdgeSetTopologyContainer::checkTopology() const
{
    PointSetTopologyContainer::checkTopology();
    if (m_edgeVertexShell.size()>0)
    {
        unsigned int i,j;
        for (i=0; i<m_edgeVertexShell.size(); ++i)
        {
            const std::vector<unsigned int> &es=m_edgeVertexShell[i];
            for (j=0; j<es.size(); ++j)
                assert((m_edge[es[j]].first==i) ||  (m_edge[es[j]].second==i));
        }
    }
    return true;
}



unsigned int EdgeSetTopologyContainer::getNumberOfEdges()
{
    if (!m_edge.size())
        createEdgeSetArray();
    return m_edge.size();
}



const std::vector< std::vector<unsigned int> > &EdgeSetTopologyContainer::getEdgeVertexShellArray()
{
    if (!m_edgeVertexShell.size())
        createEdgeVertexShellArray();
    return m_edgeVertexShell;
}






const std::vector< unsigned int > &EdgeSetTopologyContainer::getEdgeVertexShell(const unsigned int i)
{
    if (!m_edgeVertexShell.size())
        createEdgeVertexShellArray();
    return m_edgeVertexShell[i];
}


std::vector< unsigned int > &EdgeSetTopologyContainer::getEdgeVertexShellForModification(const unsigned int i)
{
    if (!m_edgeVertexShell.size())
        createEdgeVertexShellArray();
    return m_edgeVertexShell[i];
}



/*EdgeSetTopologyContainer::EdgeSetTopologyContainer(core::componentmodel::topology::BaseTopology *top) : PointSetTopologyContainer( top )
{

}
*/


EdgeSetTopologyContainer::EdgeSetTopologyContainer(core::componentmodel::topology::BaseTopology *top, const std::vector< unsigned int > &DOFIndex,
        const std::vector< Edge >         &edges )
    : PointSetTopologyContainer( top, DOFIndex ), m_edge( edges )
{

}



// factory related stuff

int EdgeSetTopologyClass = core::RegisterObject("Dynamic topology handling point sets")
        .add< EdgeSetTopology<Vec3dTypes> >()
        .add< EdgeSetTopology<Vec3fTypes> >()
        ;


} // namespace topology

} // namespace component

} // namespace sofa

