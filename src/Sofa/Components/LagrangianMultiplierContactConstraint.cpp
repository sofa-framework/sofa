#include "LagrangianMultiplierContactConstraint.inl"
#include "Common/Vec3Types.h"
#include "Sofa/Core/MechanicalObject.h"
#include "Common/ObjectFactory.h"

namespace Sofa
{

namespace Components
{

SOFA_DECL_CLASS(LagrangianMultiplierContactConstraint)

using namespace Common;

template class LagrangianMultiplierContactConstraint<Vec3dTypes>;
template class LagrangianMultiplierContactConstraint<Vec3fTypes>;

template<class DataTypes>
void create(LagrangianMultiplierContactConstraint<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< LagrangianMultiplierContactConstraint<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
    if (obj == NULL) // try the InteractionForceField initialization
        XML::createWith2Objects< LagrangianMultiplierContactConstraint<DataTypes>, Core::MechanicalModel<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
}

Creator< ObjectFactory, LagrangianMultiplierContactConstraint<Vec3dTypes> > LagrangianMultiplierContactConstraintVec3dClass("LagrangianMultiplierContactConstraint", true);
Creator< ObjectFactory, LagrangianMultiplierContactConstraint<Vec3fTypes> > LagrangianMultiplierContactConstraintVec3fClass("LagrangianMultiplierContactConstraint", true);

} // namespace Components

} // namespace Sofa
