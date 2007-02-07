
#include <sofa/component/topology/PointSetTopology.h>
#include <sofa/component/topology/PointSetTopology.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/simulation/tree/xml/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(PointSetTopology)


template class PointSetTopology<Vec3dTypes>;
template class PointSetTopology<Vec3fTypes>;
template class PointSetGeometryAlgorithms<Vec3fTypes>;
template class PointSetGeometryAlgorithms<Vec3dTypes>;

PointSetTopologyContainer::PointSetTopologyContainer(core::componentmodel::topology::BaseTopology *top)
    : Core::TopologyContainer(top)
{
}

PointSetTopologyContainer::PointSetTopologyContainer(core::componentmodel::topology::BaseTopology *top, const std::vector<unsigned int>& DOFIndex)
    : Core::TopologyContainer(top), m_DOFIndex(DOFIndex)
{
}


void PointSetTopologyContainer::createPointSetIndex()
{
    // resizing
    m_PointSetIndex.resize( m_basicTopology->getDOFNumber() );

    // initializing
    for (unsigned int i = 0; i < m_PointSetIndex.size(); ++i)
        m_PointSetIndex[i] = -1;

    // overwriting defined DOFs indices
    for (unsigned int i = 0; i < m_DOFIndex.size(); ++i)
    {
        m_PointSetIndex[ m_DOFIndex[i] ] = i;
    }
}



const std::vector<int> &PointSetTopologyContainer::getPointSetIndexArray()
{
    if (!m_PointSetIndex.size())
        createPointSetIndex();
    return m_PointSetIndex;
}

unsigned int PointSetTopologyContainer::getPointSetIndexSize() const
{
    return m_PointSetIndex.size();
}

std::vector<int> &PointSetTopologyContainer::getPointSetIndexArrayForModification()
{
    if (!m_PointSetIndex.size())
        createPointSetIndex();
    return m_PointSetIndex;
}



int PointSetTopologyContainer::getPointSetIndex(const unsigned int i)
{
    if (!m_PointSetIndex.size())
        createPointSetIndex();
    return m_PointSetIndex[i];
}

unsigned int PointSetTopologyContainer::getNumberOfVertices() const
{
    return m_DOFIndex.size();
}

const std::vector<unsigned int> &PointSetTopologyContainer::getDOFIndexArray() const
{
    return m_DOFIndex;
}

std::vector<unsigned int> &PointSetTopologyContainer::getDOFIndexArrayForModification()
{
    return m_DOFIndex;
}

unsigned int PointSetTopologyContainer::getDOFIndex(const int i) const
{
    return m_DOFIndex[i];
}



template<class DataTypes>
void create(PointSetTopology<DataTypes>*& obj, simulation::tree::xml::ObjectDescription* arg)
{
    XML::createWithParent< PointSetTopology<DataTypes>, component::MechanicalObject<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {

    }
}

Creator<simulation::tree::xml::ObjectFactory, PointSetTopology<Vec3dTypes> >
PointSetTopologyVec3dClass("PointSetTopology", true);

Creator<simulation::tree::xml::ObjectFactory, PointSetTopology<Vec3fTypes> >
PointSetTopologyVec3fClass("PointSetTopology", true);


} // namespace topology

} // namespace component

} // namespace sofa

