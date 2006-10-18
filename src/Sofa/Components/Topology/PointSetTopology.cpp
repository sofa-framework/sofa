
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


/// give a read-only access to the edge array
const PointSetTopologyContainer::VertexArray &PointSetTopologyContainer::getVertexArray() const
{
    return vertexArray;
}
PointSetTopologyContainer::indexType PointSetTopologyContainer::getVertex(const PointSetTopologyContainer::indexType i) const
{
    return vertexArray[i];
}

unsigned int PointSetTopologyContainer::getNumberOfVertices() const
{
    return vertexArray.size();
}

const PointSetTopologyContainer::InSetArray &PointSetTopologyContainer::getVertexInSetArray() const
{
    return vertexInSetArray;
}
bool PointSetTopologyContainer::isVertexInSet(const PointSetTopologyContainer::indexType i) const\
{
    return vertexInSetArray[i];
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
