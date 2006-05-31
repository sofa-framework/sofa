#include "RepulsiveSpringForceField.inl"
#include "Common/Vec3Types.h"
#include "Sofa/Core/MechanicalObject.h"
#include "Common/ObjectFactory.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

SOFA_DECL_CLASS(RepulsiveSpringForceField)

template class RepulsiveSpringForceField<Vec3dTypes>;
template class RepulsiveSpringForceField<Vec3fTypes>;

template<class DataTypes>
void create(RepulsiveSpringForceField<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParentAndFilename< RepulsiveSpringForceField<DataTypes>, Core::MechanicalObject<DataTypes> >(obj, arg);
    if (obj == NULL) // try the InteractionForceField initialization
        XML::createWith2ObjectsAndFilename< RepulsiveSpringForceField<DataTypes>, Core::MechanicalObject<DataTypes>, Core::MechanicalObject<DataTypes> >(obj, arg);
}

Creator< ObjectFactory, RepulsiveSpringForceField<Vec3dTypes> > RepulsiveSpringInteractionForceFieldVec3dClass("RepulsiveSpringForceField", true);
Creator< ObjectFactory, RepulsiveSpringForceField<Vec3fTypes> > RepulsiveSpringInteractionForceFieldVec3fClass("RepulsiveSpringForceField", true);

} // namespace Components

} // namespace Sofa
