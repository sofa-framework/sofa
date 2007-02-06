#include "ReadState.inl"
#include "Common/Vec3Types.h"
#include "Common/RigidTypes.h"
#include "Common/ObjectFactory.h"

//#include <typeinfo>

namespace Sofa
{

namespace Components
{

SOFA_DECL_CLASS(ReadState)

using namespace Common;

template class ReadState<Vec3dTypes>;
template class ReadState<Vec3fTypes>;
template class ReadState<RigidTypes>;

template<class DataTypes>
void create(ReadState<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< ReadState<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        obj->parseFields( arg->getAttributeMap() );
    }
}

Creator< ObjectFactory, ReadState<Vec3dTypes> > ReadStateVec3dClass("ReadState", true);
Creator< ObjectFactory, ReadState<Vec3fTypes> > ReadStateVec3fClass("ReadState", true);
Creator< ObjectFactory, ReadState<RigidTypes> > ReadStateRigidClass("ReadState", true);

} // namespace Components

} // namespace Sofa
