#include "WashingMachineForceField.inl"
#include "Common/Vec3Types.h"
#include "Sofa-old/Core/MechanicalObject.h"
#include "Common/ObjectFactory.h"

namespace Sofa
{

namespace Components
{
using namespace Common;

template class WashingMachineForceField<Vec3dTypes>;
template class WashingMachineForceField<Vec3fTypes>;

SOFA_DECL_CLASS(WashingMachineForceField)


template<class DataTypes>
void create(WashingMachineForceField<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< WashingMachineForceField<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
    obj->parseFields( arg->getAttributeMap() );
}

Creator< ObjectFactory, WashingMachineForceField<Vec3dTypes> > WashingMachineForceFieldVec3dClass("WashingMachineForceField", true);
Creator< ObjectFactory, WashingMachineForceField<Vec3fTypes> > WashingMachineForceFieldVec3fClass("WashingMachineForceField", true);

} // namespace Components

} // namespace Sofa
