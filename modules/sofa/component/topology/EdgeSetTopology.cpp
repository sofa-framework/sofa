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
template class EdgeSetTopologyAlgorithms<Vec3fTypes>;
template class EdgeSetTopologyAlgorithms<Vec3dTypes>;
template class EdgeSetGeometryAlgorithms<Vec3fTypes>;
template class EdgeSetGeometryAlgorithms<Vec3dTypes>;


// implementation EdgeSetTopologyContainer

void EdgeSetTopologyContainer::createEdgeShellArray ()
{
    m_edgeShell.resize( m_basicTopology->getDOFNumber() );

    for (unsigned int i = 0; i < m_edge.size(); ++i)
    {
        // adding edge i in the edge shell of both points
        m_edgeShell[ m_edge[i].first  ].push_back( i );
        m_edgeShell[ m_edge[i].second ].push_back( i );
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
    const std::vector< unsigned int > &es1=getEdgeShell(v1) ;
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



unsigned int EdgeSetTopologyContainer::getNumberOfEdges()
{
    if (!m_edge.size())
        createEdgeSetArray();
    return m_edge.size();
}



const std::vector< std::vector<unsigned int> > &EdgeSetTopologyContainer::getEdgeShellsArray()
{
    if (!m_edgeShell.size())
        createEdgeShellArray();
    return m_edgeShell;
}



//std::vector < std::vector<unsigned int> > &EdgeSetTopologyContainer::getEdgeShellsArray();



const std::vector< unsigned int > &EdgeSetTopologyContainer::getEdgeShell(const unsigned int i)
{
    if (!m_edgeShell.size())
        createEdgeShellArray();
    return m_edgeShell[i];
}


std::vector< unsigned int > &EdgeSetTopologyContainer::getEdgeShellForModification(const unsigned int i)
{
    if (!m_edgeShell.size())
        createEdgeShellArray();
    return m_edgeShell[i];
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

