
#include "PointSetTopology.h"
#include "PointSetTopology.inl"
#include "Sofa/Components/Common/Vec3Types.h"
#include "Sofa/Components/Common/ObjectFactory.h"

namespace Sofa
{

namespace Components
{

using namespace Common;


SOFA_DECL_CLASS(PointSetTopology)


template class PointSetTopology<Vec3dTypes>;
template class PointSetTopology<Vec3fTypes>;
template class PointSetGeometryAlgorithms<Vec3fTypes>;
template class PointSetGeometryAlgorithms<Vec3dTypes>;



void PointSetTopologyContainer::createPointSetIndex()
{
    // resizing
    m_PointSetIndex.resize( m_basicTopology->getDOFNumber() );

    // initializing
    for (int i = 0; i < m_PointSetIndex.size(); ++i)
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

std::vector<unsigned int> &PointSetTopologyContainer::getDOFIndexArray()
{
    return m_DOFIndex;
}

unsigned int PointSetTopologyContainer::getDOFIndex(const int i) const
{
    return m_DOFIndex[i];
}



template<class DataTypes>
void create(PointSetTopology<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< PointSetTopology<DataTypes>, Core::MechanicalObject<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {

    }
}

Creator<ObjectFactory, PointSetTopology<Vec3dTypes> >
PointSetTopologyVec3dClass("PointSetTopology", true);

Creator<ObjectFactory, PointSetTopology<Vec3fTypes> >
PointSetTopologyVec3fClass("PointSetTopology", true);


} // namespace Components

} // namespace Sofa
