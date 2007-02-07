#include "LagrangianMultiplierFixedConstraint.inl"
#include "Common/Vec3Types.h"
#include "Common/ObjectFactory.h"

namespace Sofa
{

namespace Components
{

SOFA_DECL_CLASS(LagrangianMultiplierFixedConstraint)

using namespace Common;

template class LagrangianMultiplierFixedConstraint<Vec3dTypes>;
template class LagrangianMultiplierFixedConstraint<Vec3fTypes>;

namespace Common   // \todo Why this must be inside Common namespace
{

template<class DataTypes>
void create(LagrangianMultiplierFixedConstraint<DataTypes>*& obj, ObjectDescription* arg)
{
    XML::createWithParent< LagrangianMultiplierFixedConstraint<DataTypes>, Core::MechanicalModel<DataTypes> >(obj, arg);
}

}

Creator< ObjectFactory, LagrangianMultiplierFixedConstraint<Vec3dTypes> > LagrangianMultiplierFixedConstraintVec3dClass("LagrangianMultiplierFixedConstraint", true);
Creator< ObjectFactory, LagrangianMultiplierFixedConstraint<Vec3fTypes> > LagrangianMultiplierFixedConstraintVec3fClass("LagrangianMultiplierFixedConstraint", true);

} // namespace Components

} // namespace Sofa
