#include "EdgeSetTopology.h"
#include "EdgeSetTopology.inl"
#include "Sofa/Components/Common/Vec3Types.h"
#include "Sofa/Components/Common/ObjectFactory.h"

namespace Sofa
{

namespace Components
{

using namespace Common;


SOFA_DECL_CLASS(EdgeSetTopology)


template class EdgeSetTopology<Vec3dTypes>;
template class EdgeSetTopology<Vec3fTypes>;
template class EdgeSetGeometryAlgorithms<Vec3fTypes>;
template class EdgeSetGeometryAlgorithms<Vec3dTypes>;


// implementation EdgeSetTopologyContainer

void EdgeSetTopologyContainer::createEdgeShellsArray ()
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
    return m_edge;
}



Edge &EdgeSetTopologyContainer::getEdge(const unsigned int i)
{
    return m_edge[i];
}



unsigned int EdgeSetTopologyContainer::getNumberOfEdges() const
{
    return m_edge.size();
}



const std::vector< std::vector<unsigned int> > &EdgeSetTopologyContainer::getEdgeShellsArray() const
{
    return m_edgeShell;
}



//std::vector < std::vector<unsigned int> > &EdgeSetTopologyContainer::getEdgeShellsArray();



const std::vector< unsigned int > &EdgeSetTopologyContainer::getEdgeShell(const unsigned int i) const
{
    return m_edgeShell[i];
}



/*EdgeSetTopologyContainer::EdgeSetTopologyContainer(Core::BasicTopology *top) : PointSetTopologyContainer( top )
{

}
*/


EdgeSetTopologyContainer::EdgeSetTopologyContainer(Core::BasicTopology *top, const std::vector< unsigned int > &DOFIndex,
        const std::vector< Edge >         &edges )
    : PointSetTopologyContainer( top, DOFIndex ), m_edge( edges )
{

}



// factory related stuff

template<class DataTypes>
void create(EdgeSetTopology<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< EdgeSetTopology<DataTypes>, Core::MechanicalObject<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {

    }
}

Creator<ObjectFactory, EdgeSetTopology<Vec3dTypes> >
EdgeSetTopologyVec3dClass("EdgeSetTopology", true);

Creator<ObjectFactory, EdgeSetTopology<Vec3fTypes> >
EdgeSetTopologyVec3fClass("EdgeSetTopology", true);


} // namespace Components

} // namespace Sofa
