
#include <sofa/component/topology/PointSetTopology.h>
#include <sofa/component/topology/PointSetTopology.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(PointSetTopology)


template class PointSetTopologyModifier<Vec3dTypes>;
template class PointSetTopologyModifier<Vec3fTypes>;
template class PointSetTopology<Vec3dTypes>;
template class PointSetTopology<Vec3fTypes>;
template class PointSetGeometryAlgorithms<Vec3fTypes>;
template class PointSetGeometryAlgorithms<Vec3dTypes>;

PointSetTopologyContainer::PointSetTopologyContainer(core::componentmodel::topology::BaseTopology *top)
    : core::componentmodel::topology::TopologyContainer(top)
{
}


unsigned int PointSetTopologyContainer::getNumberOfVertices() const
{

    return m_basicTopology->getDOFNumber();
}


bool PointSetTopologyContainer::checkTopology() const
{
    //std::cout << "*** CHECK PointSetTopologyContainer ***" << std::endl;
    return true;
}

int PointSetTopologyClass = core::RegisterObject("Topology consisting of a set of points")
        .add< PointSetTopology<Vec3dTypes> >()
        .add< PointSetTopology<Vec3fTypes> >()
        ;



} // namespace topology

} // namespace component

} // namespace sofa

