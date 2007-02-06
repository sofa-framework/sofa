#include "RepulsiveSpringForceField.inl"
#include "Common/Vec3Types.h"
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
    XML::createWithParent< RepulsiveSpringForceField<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj == NULL) // try the InteractionForceField initialization
        XML::createWith2Objects< RepulsiveSpringForceField<DataTypes>, Core::MechanicalModel<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj != NULL)
    {
        if (arg->getAttribute("filename"))
            obj->load(arg->getAttribute("filename"));
    }
}

Creator< ObjectFactory, RepulsiveSpringForceField<Vec3dTypes> > RepulsiveSpringInteractionForceFieldVec3dClass("RepulsiveSpringForceField", true);
Creator< ObjectFactory, RepulsiveSpringForceField<Vec3fTypes> > RepulsiveSpringInteractionForceFieldVec3fClass("RepulsiveSpringForceField", true);

} // namespace Components

} // namespace Sofa
