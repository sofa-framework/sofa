#include "WriteState.inl"
#include "Common/Vec3Types.h"
#include "Common/RigidTypes.h"
#include "Common/ObjectFactory.h"

//#include <typeinfo>

namespace Sofa
{

namespace Components
{

SOFA_DECL_CLASS(WriteState)

using namespace Common;

template class WriteState<Vec3dTypes>;
template class WriteState<Vec3fTypes>;
template class WriteState<RigidTypes>;

template<class DataTypes>
void create(WriteState<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< WriteState<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj!=NULL)
    {
        obj->parseFields( arg->getAttributeMap() );
    }
}

Creator< ObjectFactory, WriteState<Vec3dTypes> > WriteStateVec3dClass("WriteState", true);
Creator< ObjectFactory, WriteState<Vec3fTypes> > WriteStateVec3fClass("WriteState", true);
Creator< ObjectFactory, WriteState<RigidTypes> > WriteStateRigidClass("WriteState", true);

} // namespace Components

} // namespace Sofa
