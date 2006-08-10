#include "LagrangianMultiplierAttachConstraint.inl"
#include "Common/Vec3Types.h"
#include "Sofa/Core/MechanicalObject.h"
#include "Common/ObjectFactory.h"

namespace Sofa
{

namespace Components
{

SOFA_DECL_CLASS(LagrangianMultiplierAttachConstraint)

using namespace Common;

template class LagrangianMultiplierAttachConstraint<Vec3dTypes>;
template class LagrangianMultiplierAttachConstraint<Vec3fTypes>;

template<class DataTypes>
void create(LagrangianMultiplierAttachConstraint<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< LagrangianMultiplierAttachConstraint<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj == NULL) // try the InteractionForceField initialization
        XML::createWith2Objects< LagrangianMultiplierAttachConstraint<DataTypes>, Core::MechanicalModel<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
}

Creator< ObjectFactory, LagrangianMultiplierAttachConstraint<Vec3dTypes> > LagrangianMultiplierAttachConstraintVec3dClass("LagrangianMultiplierAttachConstraint", true);
Creator< ObjectFactory, LagrangianMultiplierAttachConstraint<Vec3fTypes> > LagrangianMultiplierAttachConstraintVec3fClass("LagrangianMultiplierAttachConstraint", true);

} // namespace Components

} // namespace Sofa
